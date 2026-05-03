#!/usr/bin/env python3
import math
import heapq
import numpy as np
from dataclasses import dataclass, field
import cvxpy
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from scipy.spatial import transform
import os

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from utils import nearest_point


from scipy.interpolate import CubicSpline


@dataclass
class mpc_config:
    NXK: int = 4  # length of kinematic state vector: z = [x, y, v, yaw]
    NU: int = 2  # length of input vector: u = [steering speed, acceleration]
    TK: int = 20  # finite time horizon length - kinematic

    # ---------------------------------------------------
    # TODO: you may need to tune the following matrices
    Rk: list = field(
        #default_factory=lambda: np.diag([10.0, 5.0])
        default_factory=lambda: np.diag([5.0, 35.0])
    )  # input cost matrix, penalty for inputs - [accel, steering_speed]
    Rdk: list = field(
        #default_factory=lambda: np.diag([10.0, 5.0])
        default_factory=lambda: np.diag([1.0, 35.0])
    )  # input difference cost matrix, penalty for change of inputs - [accel, steering_speed]
    Qk: list = field(
        #default_factory=lambda: np.diag([30.0, 25., 35.0, 20.0])  # levine sim
        default_factory=lambda: np.diag([40., 40., 10.0, 50.0])
    )  # state error cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
    Qfk: list = field(
        #default_factory=lambda: np.diag([33.5, 25., 35.0, 20.0])  # levine sim
        default_factory=lambda: np.diag([40., 40., 10.0, 50.0])
    )  # final state error matrix, penalty  for the final state constraints: [x, y, v, yaw]
    # ---------------------------------------------------


    N_IND_SEARCH: int = 20  # Search index number
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 0.03  # dist step [m] kinematic
    LENGTH: float = 0.5  # Length of the vehicle [m]
    WIDTH: float = 0.280  # Width of the vehicle [m]
    WB: float = 0.330  # Wheelbase [m]
    MIN_STEER: float = -0.353 #$-0.4236,5236  # maximum steering angle [rad]
    MAX_STEER: float = 0.353 #0.4236,5236  # maximum steering angle [rad]
    MAX_DSTEER: float = np.deg2rad(180.0)  # maximum steering speed [rad/s]    
    MAX_SPEED: float = 1.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 5.0  # maximum acceleration [m/ss]

    # obstacle / CBF tuning
    # OBS_RADIUS is the default physical radius for each obstacle.
    # If one obstacle is larger/smaller, change OBS_1_RADIUS, OBS_2_RADIUS, OBS_3_RADIUS.
    OBS_RADIUS: float = 0.20      # [m] default obstacle radius approximation
    OBS_1_RADIUS: float = 0.20    # [m] physical radius of obstacle 1
    OBS_2_RADIUS: float = 0.20    # [m] physical radius of obstacle 2
    OBS_3_RADIUS: float = 0.20    # [m] physical radius of obstacle 3
    SAFETY_MARGIN: float = 0.15   # [m] extra clearance
    OBSTACLE_MARKER_HEIGHT: float = 0.08  # [m] RViz cylinder height

    # ---------- soft-min composite CBF tuning ----------
    SOFTMIN_KAPPA: float = 10.0
    BETA_1: float = 1e3
    SOFTMIN_TOPK: int = 2
    FAR_OBS_THRESHOLD: float = 1.0e5
    SOFTMIN_INACTIVE_VALUE: float = 1.0e3

    MIN_PREVIEW_SPEED: float = 0.8
    MIN_PREVIEW_DIND: float = 1.0


    # ---------- SCP tuning ----------
    # SCP_TRUST_XY: float = 0.30      # meters
    # SCP_TRUST_YAW: float = 0.70     # radians
    # SCP_TRUST_V: float = 0.50       # m/s

    # Number of SCP iterations.
    # Use 2 for real time. Use 3 for debugging.
    SCP_MAX_ITER: int = 2

    # Damping prevents large jumps between SCP iterations.
    # 1.0 means fully accept new QP solution.
    # 0.5 means blend old and new.
    SCP_ALPHA: float = 0.6


    # ---------- Hybrid A* local planner ----------
    USE_HYBRID_ASTAR: bool = True
    HA_TRIGGER_BUFFER: float = 0.50       # [m] extra trigger distance around CBF radius
    HA_GOAL_AHEAD_INDEX: int = 50        # waypoint index ahead on global square path
    HA_GRID_RES: float = 0.06            # [m] grid resolution for closed-set keys
    HA_YAW_BINS: int = 36                # 10 deg yaw bins
    HA_STEP: float = 0.10                # [m] forward simulation step
    HA_GOAL_TOL: float = 0.20            # [m] goal tolerance
    HA_MAX_EXPANSIONS: int = 12000
    HA_LOCAL_SIZE_X: float = 4.0         # [m] local planning box size
    HA_LOCAL_SIZE_Y: float = 4.0         # [m] local planning box size
    HA_LOCAL_MARGIN: float = 0.60        # [m] ensure goal has margin inside box
    HA_OBS_INFLATION_EXTRA: float = 0.03 # [m] extra planner inflation
    HA_PATH_SPEED: float = 0.40          # [m/s] speed assigned to planned path
    HA_STEER_PENALTY: float = 0.03
    HA_REVERSE_ALLOWED: bool = False     # keep False first for safer F1TENTH behavior
    HA_FORCE_PLAN: bool = False          # False: plan only when the square path is blocked

    # Keep the local planner close to the main square trajectory.
    # Without this corridor, A*/Hybrid A* will take the shortest shortcut.
    HA_USE_PATH_CORRIDOR: bool = True
    HA_CORRIDOR_WIDTH: float = 0.65      # [m] allowed distance from current square-path segment
    HA_CORRIDOR_EXTRA_AHEAD: int = 15    # extra global waypoints after the local goal


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0
    v: float = 0.0
    yaw: float = 0.0
    yawrate: float = 0.0
    beta: float = 0.0


class MPC(Node):
    """ 
    Implement Kinematic MPC on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('mpc_node')
        # use the MPC as a tracker (similar to pure pursuit)
        self.is_real = True
        #self.map_name = 'levine_2nd'
        #self.map_name = 'Spielberg_raceline'
        #self.map_name = 'map_2_f1tenth'
        #self.map_name = 'levine_centerline'
        # self.map_name = 'siccs_first_floor_1'
        #self.map_name = 'square_trajectory_small'
        self.map_name = 'square_trajectory'
        
        self.enable_drive = True  # enable drive message publishing

        # create ROS subscribers and publishers
        # pose_topic = "/pf/viz/inferred_pose" if self.is_real else "/ego_racecar/odom"
        pose_topic = "/optitrack/object_529/pose" if self.is_real else "/ego_racecar/odom"
        drive_topic = "/drive"
        vis_ref_traj_topic = "/ref_traj_marker"
        vis_waypoints_topic = "/waypoints_marker"
        vis_pred_path_topic = "/pred_path_marker"
        vis_obstacles_topic = "/obstacle_markers"
        vis_hybrid_path_topic = "/hybrid_astar_path_marker"

        self.pose_sub = self.create_subscription(PoseStamped if self.is_real else Odometry, pose_topic, self.pose_callback, 1)
        self.pose_sub  # prevent unused variable warning

        self.obs_1 = self.create_subscription(Odometry, "/optitrack/object_527/odom", self.obs_1_callback, 10)
        self.obs_2 = self.create_subscription(Odometry, "/optitrack/object_528/odom", self.obs_2_callback, 10)
        #self.obs_3 = self.create_subscription(Odometry, "/optitrack/object_526/odom", self.obs_3_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()

        self.vis_waypoints_pub = self.create_publisher(Marker, vis_waypoints_topic, 1)
        self.vis_waypoints_msg = Marker()
        self.vis_ref_traj_pub = self.create_publisher(Marker, vis_ref_traj_topic, 1)
        self.vis_ref_traj_msg = Marker()
        self.vis_pred_path_pub = self.create_publisher(Marker, vis_pred_path_topic, 1)
        self.vis_pred_path_msg = Marker()
        self.vis_obstacles_pub = self.create_publisher(MarkerArray, vis_obstacles_topic, 1)
        self.vis_hybrid_path_pub = self.create_publisher(Marker, vis_hybrid_path_topic, 1)
        self.vis_hybrid_path_msg = Marker()

        #map_path = os.path.abspath(os.path.join('src', 'f1tenth-software-stack','csv_data'))

        #print("[DEBUG] :", map_path)
        
        #self.waypoints = np.loadtxt(map_path + '/' + self.map_name + '.csv', delimiter=';', skiprows=0)  # csv data
        #self.waypoints = np.loadtxt(map_path + '/' + self.map_name + '_centerline.csv', delimiter=',', skiprows=1)  # csv data F1tenth centerline
        self.waypoints = np.loadtxt(self.map_name + '.csv', delimiter=',', skiprows=1)  # csv data F1tenth 
        #self.waypoints = np.loadtxt(map_path + '/' + self.map_name + '.csv', delimiter=';', skiprows=3)
        
        self.ref_pos_x = self.waypoints[:, 0]  # x position
        self.ref_pos_y = self.waypoints[:, 1]  # y position
        #self.ref_speed = self.waypoints[:, 5] * 1.5  # speed profile
        
        # for those who do not have speed profile in the csv file
        self.ref_speed = np.ones(len(self.waypoints)) * 0.7  # speed profile
        
        dx = np.gradient(self.ref_pos_x) # for levine is 1 colunm ahead [1]
        dy = np.gradient(self.ref_pos_y) # for levine is 1 colunm ahead [2]
        yaw = np.arctan2(dy, dx)
        self.heading_yaw = np.unwrap(yaw)
        

        
        
        if self.map_name == 'levine_2nd':
            self.heading_yaw += math.pi / 2
        self.visualize_waypoints_in_rviz()

        self.config = mpc_config()
        self.car_radius = 0.5 * math.hypot(self.config.LENGTH, self.config.WIDTH)
        # Safety radius used by the CBF is computed per obstacle:
        # r_safe_i = car_radius + obstacle_radius_i + SAFETY_MARGIN.

        self.odelta_v = None
        self.odelta = None
        self.oa = None
        self.init_flag = 0

        # store the last successful MPC solution
        self.last_ok_oa = np.zeros(self.config.TK)
        self.last_ok_odelta = np.zeros(self.config.TK)
        self.last_mpc_ok = False
        self.mpc_fail_count = 0

        # initialize MPC problem
        self.mpc_prob_init()

        # init state - avoid unknown variables for scan callback
        self.curr_pos = np.array([0.0, 0.0, 0.0])
        self.rot_mat = np.identity(3)
    
        # Initialize as invalid until OptiTrack publishes.
        # Important: this must be larger than FAR_OBS_THRESHOLD, otherwise the
        # obstacle will be treated as valid before receiving a real measurement.
        far_obs = 2.0 * self.config.FAR_OBS_THRESHOLD
        self.obs_1_x = far_obs
        self.obs_1_y = far_obs
        self.obs_2_x = far_obs
        self.obs_2_y = far_obs
        self.obs_3_x = far_obs
        self.obs_3_y = far_obs

        # Physical radius of each obstacle. Update these values if your obstacles have different sizes.
        self.obs_1_r = self.config.OBS_1_RADIUS
        self.obs_2_r = self.config.OBS_2_RADIUS
        self.obs_3_r = self.config.OBS_3_RADIUS

        # Publish obstacle and CBF safety-radius markers at 10 Hz.
        self.obs_marker_timer = self.create_timer(0.1, self.visualize_obstacles_in_rviz)

        self.last_hybrid_astar_path = None
        self.planner_corridor_xy = None
        self.active_corridor_width = self.config.HA_CORRIDOR_WIDTH
        self.planner_start_ind = None
        self.planner_goal_ind = None

        self.prev_pose_time = None
        self.prev_pose_x = None
        self.prev_pose_y = None


    def obs_1_callback(self, msg):
        self.obs_1_x = msg.pose.pose.position.x
        self.obs_1_y = msg.pose.pose.position.y

    def obs_2_callback(self, msg):
        self.obs_2_x = msg.pose.pose.position.x
        self.obs_2_y = msg.pose.pose.position.y

    def obs_3_callback(self, msg):
        self.obs_3_x = msg.pose.pose.position.x
        self.obs_3_y = msg.pose.pose.position.y

    def pose_callback(self, pose_msg):
        # extract pose from ROS msg
        self.update_rotation_matrix(pose_msg)
        vehicle_state = self.update_vehicle_state(pose_msg)

        if self.is_real:
            # vehicle_state.v = -1 * vehicle_state.v  # negate the monitoring speed
            vehicle_state.v = 1 * vehicle_state.v  # negate the monitoring speed

        # TODO: Calculate the next reference trajectory for the next T steps with current vehicle pose.
        # ref_x, ref_y, ref_yaw, ref_v are columns of self.waypoints
        ref_path = self.calc_ref_trajectory(vehicle_state, self.ref_pos_x, self.ref_pos_y, self.heading_yaw, self.ref_speed)
        #ref_path = self.calc_ref_trajectory(vehicle_state, self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2], self.waypoints[:, 3]) # f1tenth AutoDrive
        # print(ref_path)

        # Hybrid A* only changes the reference when the nominal square-path horizon is blocked.
        ref_path = self.apply_hybrid_astar_if_needed(vehicle_state, ref_path)

        self.visualize_ref_traj_in_rviz(ref_path)
        
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]
        # print(vehicle_state.v)

        # solve the MPC control problem
        (
            self.oa,
            self.odelta_v,
            ox,
            oy,
            oyaw,
            ov,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.oa, self.odelta_v)

        # publish drive message.
        if self.odelta_v is not None and self.oa is not None:
            steer_output = self.odelta_v[0]
            #speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
            speed_output = vehicle_state.v + self.oa[0] * self.config.DTK
            speed_output = np.clip(speed_output, 0.0, self.config.MAX_SPEED)

            self.drive_msg.drive.steering_angle = steer_output
            self.drive_msg.drive.speed = 1.0 * speed_output
            #self.drive_msg.drive.speed = (-1.0 if self.is_real else 1.0) * speed_output
        
            ######## PUBLISH DRIVE FOR CAR ##############################
            if self.enable_drive:
                self.drive_pub.publish(self.drive_msg)
        
        print("steering ={}, speed ={}".format(self.drive_msg.drive.steering_angle, self.drive_msg.drive.speed))

        self.vis_waypoints_pub.publish(self.vis_waypoints_msg)
        self.visualize_obstacles_in_rviz()

    # toolkits
    def update_rotation_matrix(self, pose_msg):
        # get rotation matrix from the car frame to the world frame
        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        self.rot_mat = (transform.Rotation.from_quat(quat)).as_matrix()
        # print("rotation matrix = {}".format(self.rot_mat))


    def update_vehicle_state(self, pose_msg):
        vehicle_state = State()

        vehicle_state.x = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        vehicle_state.y = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y

        curr_orien = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        q = [curr_orien.x, curr_orien.y, curr_orien.z, curr_orien.w]
        vehicle_state.yaw = math.atan2(
            2 * (q[3] * q[2] + q[0] * q[1]),
            1 - 2 * (q[1] ** 2 + q[2] ** 2)
        )

        # ---------- speed estimation from pose ----------
        stamp = pose_msg.header.stamp if self.is_real else pose_msg.header.stamp
        curr_time = float(stamp.sec) + 1e-9 * float(stamp.nanosec)

        if self.prev_pose_time is None:
            vehicle_state.v = self.drive_msg.drive.speed
        else:
            dt = curr_time - self.prev_pose_time
            if dt <= 1e-4:
                vehicle_state.v = self.drive_msg.drive.speed
            else:
                dx = vehicle_state.x - self.prev_pose_x
                dy = vehicle_state.y - self.prev_pose_y
                #vehicle_state.v = math.sqrt(dx * dx + dy * dy) / dt
                raw_v = math.sqrt(dx * dx + dy * dy) / dt
                raw_v = np.clip(raw_v, 0.0, self.config.MAX_SPEED)

                alpha = 0.3
                if not hasattr(self, "v_filt"):
                    self.v_filt = raw_v
                else:
                    self.v_filt = alpha * raw_v + (1.0 - alpha) * self.v_filt

                vehicle_state.v = self.v_filt

        self.prev_pose_time = curr_time
        self.prev_pose_x = vehicle_state.x
        self.prev_pose_y = vehicle_state.y

        return vehicle_state





    # ------------------------------------------------------------------
    # Hybrid A* local planner
    # ------------------------------------------------------------------
    def wrap_to_pi(self, angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def yaw_to_bin(self, yaw):
        yaw_wrapped = self.wrap_to_pi(yaw)
        yaw_norm = (yaw_wrapped + math.pi) / (2.0 * math.pi)
        return int(math.floor(yaw_norm * self.config.HA_YAW_BINS)) % self.config.HA_YAW_BINS

    def hybrid_state_key(self, x, y, yaw, bounds):
        xmin, xmax, ymin, ymax = bounds
        ix = int(math.floor((x - xmin) / self.config.HA_GRID_RES))
        iy = int(math.floor((y - ymin) / self.config.HA_GRID_RES))
        iyaw = self.yaw_to_bin(yaw)
        return (ix, iy, iyaw)

    def get_planner_obstacle_radius(self, obs_radius):
        """
        Radius used by the planner.

        Important: this must be LESS conservative than the CBF radius.
        The CBF can use SAFETY_MARGIN, but the planner should only avoid
        the physical collision region plus a small buffer.
        """
        return self.car_radius + obs_radius + self.config.HA_OBS_INFLATION_EXTRA

    def ref_path_blocked_by_obstacle(self, ref_path):
        """
        Return True if any OptiTrack obstacle is close enough to the nominal MPC
        reference horizon. This decides when Hybrid A* should replace the reference.
        """
        obstacles = self.get_valid_obstacles()
        if len(obstacles) == 0:
            return False

        path_xy = ref_path[0:2, :].T
        for ox, oy, obs_radius in obstacles:
            obs_xy = np.array([ox, oy])
            dmin = np.min(np.linalg.norm(path_xy - obs_xy, axis=1))
            trigger_radius = self.get_safe_radius(obs_radius) + self.config.HA_TRIGGER_BUFFER
            if dmin < trigger_radius:
                return True

        return False

    def prepare_planner_context(self, state):
        """
        Build a local corridor around the CURRENT part of the square trajectory.

        This is important: the planner is not allowed to use the whole lab as free
        space. It should only make a local detour and then reconnect to the square.
        """
        _, _, _, ind = nearest_point(
            np.array([state.x, state.y]),
            np.array([self.ref_pos_x, self.ref_pos_y]).T,
        )

        n = len(self.ref_pos_x)
        start_ind = int(ind)
        goal_ind = (start_ind + int(self.config.HA_GOAL_AHEAD_INDEX)) % n

        # Store these so get_hybrid_astar_goal() uses the same goal.
        self.planner_start_ind = start_ind
        self.planner_goal_ind = goal_ind

        # Corridor follows only the local forward segment of the square path,
        # not the whole square. This prevents shortcuts across the rectangle.
        num = int(self.config.HA_GOAL_AHEAD_INDEX + self.config.HA_CORRIDOR_EXTRA_AHEAD)
        idxs = [(start_ind + k) % n for k in range(num + 1)]
        self.planner_corridor_xy = np.column_stack((
            self.ref_pos_x[idxs],
            self.ref_pos_y[idxs],
        ))

        # If the car is slightly off the path, do not reject it immediately.
        start_d = self.distance_to_planner_corridor(state.x, state.y)
        self.active_corridor_width = max(
            self.config.HA_CORRIDOR_WIDTH,
            start_d + 0.15,
        )

    def distance_to_planner_corridor(self, x, y):
        if self.planner_corridor_xy is None or len(self.planner_corridor_xy) == 0:
            return 0.0
        p = np.array([x, y], dtype=float)
        return float(np.min(np.linalg.norm(self.planner_corridor_xy - p, axis=1)))

    def get_hybrid_astar_goal(self, state):
        """
        Pick a local goal ahead on the original square trajectory.
        This is what prevents the car from stopping at the obstacle.
        """
        _, _, _, ind = nearest_point(
            np.array([state.x, state.y]),
            np.array([self.ref_pos_x, self.ref_pos_y]).T,
        )

        n = len(self.ref_pos_x)
        if self.planner_goal_ind is not None:
            goal_ind = int(self.planner_goal_ind) % n
        else:
            goal_ind = (int(ind) + int(self.config.HA_GOAL_AHEAD_INDEX)) % n

        gx = float(self.ref_pos_x[goal_ind])
        gy = float(self.ref_pos_y[goal_ind])
        gyaw = float(self.heading_yaw[goal_ind])

        # Keep the goal yaw numerically close to the current yaw.
        while gyaw - state.yaw > math.pi:
            gyaw -= 2.0 * math.pi
        while state.yaw - gyaw > math.pi:
            gyaw += 2.0 * math.pi

        return gx, gy, gyaw

    def get_hybrid_astar_bounds(self, start, goal):
        sx, sy, _ = start
        gx, gy, _ = goal

        xmin = sx - 0.5 * self.config.HA_LOCAL_SIZE_X
        xmax = sx + 0.5 * self.config.HA_LOCAL_SIZE_X
        ymin = sy - 0.5 * self.config.HA_LOCAL_SIZE_Y
        ymax = sy + 0.5 * self.config.HA_LOCAL_SIZE_Y

        # Make sure the goal is inside the planning box.
        m = self.config.HA_LOCAL_MARGIN
        xmin = min(xmin, gx - m)
        xmax = max(xmax, gx + m)
        ymin = min(ymin, gy - m)
        ymax = max(ymax, gy + m)

        return xmin, xmax, ymin, ymax

    def is_xy_inside_bounds(self, x, y, bounds):
        xmin, xmax, ymin, ymax = bounds
        return xmin <= x <= xmax and ymin <= y <= ymax

    def is_collision_free_xy(self, x, y, bounds, obstacles):
        if not self.is_xy_inside_bounds(x, y, bounds):
            return False

        # Do not let the local planner shortcut across the square.
        # It must stay near the current forward segment of the main trajectory.
        if self.config.HA_USE_PATH_CORRIDOR:
            if self.distance_to_planner_corridor(x, y) > self.active_corridor_width:
                return False

        for ox, oy, obs_radius in obstacles:
            rr = self.get_planner_obstacle_radius(obs_radius)
            dx = x - ox
            dy = y - oy
            if dx * dx + dy * dy <= rr * rr:
                return False

        return True

    def is_motion_collision_free(self, x0, y0, x1, y1, bounds, obstacles):
        """Check the short motion primitive with a few samples."""
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(math.ceil(dist / (0.5 * self.config.HA_GRID_RES))))
        for i in range(n + 1):
            s = i / float(n)
            x = (1.0 - s) * x0 + s * x1
            y = (1.0 - s) * y0 + s * y1
            if not self.is_collision_free_xy(x, y, bounds, obstacles):
                return False
        return True

    def reconstruct_hybrid_path(self, goal_key, parents, node_data):
        path = []
        key = goal_key
        while key is not None:
            path.append(node_data[key])
            key = parents[key]
        path.reverse()
        return np.asarray(path, dtype=float)

    def hybrid_astar_plan(self, state):
        """
        Lightweight Hybrid A* planner.

        State: (x, y, yaw). Motion primitives use the same bicycle geometry as the MPC.
        Obstacles are OptiTrack circles inflated by car radius + margin.
        """
        obstacles = self.get_valid_obstacles()
        if len(obstacles) == 0:
            return None

        start = (float(state.x), float(state.y), float(state.yaw))
        goal = self.get_hybrid_astar_goal(state)
        bounds = self.get_hybrid_astar_bounds(start, goal)

        # Do not reject the start. In real experiments, the car may already be
        # inside the CBF safety margin. The planner should still try to escape.
        if not self.is_collision_free_xy(goal[0], goal[1], bounds, obstacles):
            print("[Hybrid A*] Goal is inside planner obstacle inflation.")
            return None

        steer_set = [self.config.MIN_STEER, 0.0, self.config.MAX_STEER]
        directions = [1.0]
        if self.config.HA_REVERSE_ALLOWED:
            directions.append(-1.0)

        start_key = self.hybrid_state_key(start[0], start[1], start[2], bounds)
        parents = {start_key: None}
        node_data = {start_key: start}
        g_cost = {start_key: 0.0}

        def heuristic(x, y):
            return math.hypot(goal[0] - x, goal[1] - y)

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (heuristic(start[0], start[1]), counter, start_key))

        best_key = start_key
        best_dist = heuristic(start[0], start[1])
        expansions = 0

        while open_heap and expansions < self.config.HA_MAX_EXPANSIONS:
            _, _, current_key = heapq.heappop(open_heap)
            cx, cy, cyaw = node_data[current_key]
            expansions += 1

            dist_to_goal = heuristic(cx, cy)
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_key = current_key

            if dist_to_goal <= self.config.HA_GOAL_TOL:
                print(f"[Hybrid A*] Success. expansions={expansions}, goal_dist={dist_to_goal:.3f}")
                return self.reconstruct_hybrid_path(current_key, parents, node_data)

            for direction in directions:
                for delta in steer_set:
                    ds = direction * self.config.HA_STEP
                    nx = cx + ds * math.cos(cyaw)
                    ny = cy + ds * math.sin(cyaw)
                    nyaw = self.wrap_to_pi(cyaw + (ds / self.config.WB) * math.tan(delta))

                    if not self.is_motion_collision_free(cx, cy, nx, ny, bounds, obstacles):
                        continue

                    nkey = self.hybrid_state_key(nx, ny, nyaw, bounds)

                    step_cost = abs(ds)
                    step_cost += self.config.HA_STEER_PENALTY * abs(delta) / max(abs(self.config.MAX_STEER), 1e-6)
                    if direction < 0.0:
                        step_cost += 0.50

                    new_g = g_cost[current_key] + step_cost
                    if nkey not in g_cost or new_g < g_cost[nkey]:
                        g_cost[nkey] = new_g
                        parents[nkey] = current_key
                        node_data[nkey] = (nx, ny, nyaw)
                        counter += 1
                        f = new_g + heuristic(nx, ny)
                        heapq.heappush(open_heap, (f, counter, nkey))

        if best_dist < 0.50:
            print(f"[Hybrid A*] Partial path accepted. best_dist={best_dist:.3f}")
            return self.reconstruct_hybrid_path(best_key, parents, node_data)

        print(f"[Hybrid A*] Failed. best_dist={best_dist:.3f}, expansions={expansions}")
        return None

    def grid_astar_plan(self, state):
        """
        Robust fallback planner on a local 2D grid.

        This is not the final controller. It is a debugging/planning backup that
        almost always produces a geometric path around OptiTrack obstacles. The
        MPC then tracks this path.
        """
        obstacles = self.get_valid_obstacles()
        if len(obstacles) == 0:
            return None

        start = (float(state.x), float(state.y), float(state.yaw))
        goal = self.get_hybrid_astar_goal(state)
        bounds = self.get_hybrid_astar_bounds(start, goal)

        xmin, xmax, ymin, ymax = bounds
        res = self.config.HA_GRID_RES
        nx = int(math.ceil((xmax - xmin) / res)) + 1
        ny = int(math.ceil((ymax - ymin) / res)) + 1

        def world_to_grid(x, y):
            ix = int(round((x - xmin) / res))
            iy = int(round((y - ymin) / res))
            ix = int(np.clip(ix, 0, nx - 1))
            iy = int(np.clip(iy, 0, ny - 1))
            return ix, iy

        def grid_to_world(ix, iy):
            return xmin + ix * res, ymin + iy * res

        start_idx = world_to_grid(start[0], start[1])
        goal_idx = world_to_grid(goal[0], goal[1])

        def free_cell(ix, iy):
            # Always allow the start cell, otherwise no planner can escape when
            # the car begins slightly inside the inflated safety region.
            if (ix, iy) == start_idx:
                return True
            x, y = grid_to_world(ix, iy)
            if not self.is_xy_inside_bounds(x, y, bounds):
                return False
            for ox, oy, obs_radius in obstacles:
                rr = self.get_planner_obstacle_radius(obs_radius)
                dx = x - ox
                dy = y - oy
                if dx * dx + dy * dy <= rr * rr:
                    return False
            return True

        # If the selected goal is occupied, move farther along the square path.
        if not free_cell(goal_idx[0], goal_idx[1]):
            _, _, _, ind = nearest_point(
                np.array([state.x, state.y]),
                np.array([self.ref_pos_x, self.ref_pos_y]).T,
            )
            found_goal = False
            nwp = len(self.ref_pos_x)
            for offset in range(self.config.HA_GOAL_AHEAD_INDEX, self.config.HA_GOAL_AHEAD_INDEX + 180, 10):
                gi = (int(ind) + offset) % nwp
                cand = (float(self.ref_pos_x[gi]), float(self.ref_pos_y[gi]), float(self.heading_yaw[gi]))
                goal_idx_tmp = world_to_grid(cand[0], cand[1])
                if free_cell(goal_idx_tmp[0], goal_idx_tmp[1]):
                    goal = cand
                    goal_idx = goal_idx_tmp
                    found_goal = True
                    break
            if not found_goal:
                print("[Grid A*] Could not find a free goal ahead.")
                return None

        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        def heuristic(idx):
            return math.hypot(idx[0] - goal_idx[0], idx[1] - goal_idx[1])

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start_idx), 0, start_idx))
        parents = {start_idx: None}
        g_cost = {start_idx: 0.0}
        counter = 0
        best_idx = start_idx
        best_h = heuristic(start_idx)

        while open_heap and counter < self.config.HA_MAX_EXPANSIONS:
            _, _, current = heapq.heappop(open_heap)

            hcur = heuristic(current)
            if hcur < best_h:
                best_h = hcur
                best_idx = current

            if current == goal_idx or hcur * res <= self.config.HA_GOAL_TOL:
                best_idx = current
                break

            cx, cy = current
            for dx, dy, cost_mult in neighbors:
                ni = cx + dx
                nj = cy + dy
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                    continue
                if not free_cell(ni, nj):
                    continue

                nidx = (ni, nj)
                new_g = g_cost[current] + cost_mult * res
                if nidx not in g_cost or new_g < g_cost[nidx]:
                    g_cost[nidx] = new_g
                    parents[nidx] = current
                    counter += 1
                    heapq.heappush(open_heap, (new_g + heuristic(nidx) * res, counter, nidx))

        if best_idx == start_idx:
            print("[Grid A*] Failed: best path stayed at start.")
            return None

        # Reconstruct path.
        cells = []
        k = best_idx
        while k is not None:
            cells.append(k)
            k = parents[k]
        cells.reverse()

        xy = np.array([grid_to_world(ix, iy) for ix, iy in cells], dtype=float)
        if xy.shape[0] < 2:
            return None

        yaw = np.zeros(xy.shape[0])
        yaw[:-1] = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
        yaw[-1] = yaw[-2]

        path = np.column_stack((xy[:, 0], xy[:, 1], np.unwrap(yaw)))
        print(f"[Grid A*] Success. points={path.shape[0]}, best_goal_error={best_h * res:.3f}")
        return path


    def hybrid_path_to_ref_path(self, hybrid_path, nominal_ref_path):
        """Convert the Hybrid A* geometric path into the MPC ref_path format."""
        if hybrid_path is None or len(hybrid_path) < 2:
            return nominal_ref_path

        x = hybrid_path[:, 0]
        y = hybrid_path[:, 1]

        ds = np.hypot(np.diff(x), np.diff(y))
        s = np.insert(np.cumsum(ds), 0, 0.0)
        total = float(s[-1])
        if total < 1e-6:
            return nominal_ref_path

        s_new = np.linspace(0.0, total, self.config.TK + 1)
        x_new = np.interp(s_new, s, x)
        y_new = np.interp(s_new, s, y)

        ref_path = nominal_ref_path.copy()
        ref_path[0, :] = x_new
        ref_path[1, :] = y_new
        ref_path[2, :] = self.config.HA_PATH_SPEED

        dx = np.gradient(x_new)
        dy = np.gradient(y_new)
        yaw = np.arctan2(dy, dx)
        ref_path[3, :] = np.unwrap(yaw)

        return ref_path

    def apply_hybrid_astar_if_needed(self, state, nominal_ref_path):
        """
        Main integration point.

        While debugging, HA_FORCE_PLAN=True makes the planner run whenever
        OptiTrack obstacles exist. After it works, you can set it False so
        it only activates when the reference is blocked.
        """
        if not self.config.USE_HYBRID_ASTAR:
            return nominal_ref_path

        obstacles = self.get_valid_obstacles()
        if len(obstacles) == 0:
            self.last_hybrid_astar_path = None
            self.clear_hybrid_astar_path_in_rviz()
            return nominal_ref_path

        blocked = self.ref_path_blocked_by_obstacle(nominal_ref_path)
        if (not self.config.HA_FORCE_PLAN) and (not blocked):
            self.last_hybrid_astar_path = None
            self.clear_hybrid_astar_path_in_rviz()
            return nominal_ref_path

        # Build the local corridor and the local goal on the square path.
        self.prepare_planner_context(state)

        # 1) Try Hybrid A*.
        hybrid_path = self.hybrid_astar_plan(state)

        # 2) If Hybrid A* fails, use a grid A* fallback.
        # This makes sure you still see a path in RViz while debugging.
        if hybrid_path is None:
            print("[Planner] Hybrid A* failed. Trying grid A* fallback.")
            hybrid_path = self.grid_astar_plan(state)

        if hybrid_path is None:
            print("[Planner] No local path found. Keeping nominal MPC reference.")
            self.last_hybrid_astar_path = None
            self.clear_hybrid_astar_path_in_rviz()
            return nominal_ref_path

        self.last_hybrid_astar_path = hybrid_path
        self.visualize_hybrid_astar_path_in_rviz(hybrid_path)
        return self.hybrid_path_to_ref_path(hybrid_path, nominal_ref_path)

    # mpc functions
    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html
        More QP example in CVXPY here: https://www.cvxpy.org/examples/basic/quadratic_program.html
        """
        # Initialize and create vectors for the optimization problem
        # Vehicle State Vector
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)  # 4 x 9
        )
        # Control Input vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)  # 2 x 8
        )
        objective = 0.0  # Objective value of the optimization problem
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))  # 4
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))  # 4 x 9
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # soft-min composite affine CBF coefficients:
        # h_soft_lin = a_x * x + a_y * y + a_c >= -slack
        self.cbf_soft_ax_k = cvxpy.Parameter(self.config.TK + 1)
        self.cbf_soft_ay_k = cvxpy.Parameter(self.config.TK + 1)
        self.cbf_soft_ac_k = cvxpy.Parameter(self.config.TK + 1)

        self.cbf_soft_ax_k.value = np.zeros(self.config.TK + 1)
        self.cbf_soft_ay_k.value = np.zeros(self.config.TK + 1)
        self.cbf_soft_ac_k.value = (
            np.ones(self.config.TK + 1) * self.config.SOFTMIN_INACTIVE_VALUE
        )

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))  # (2 * 8) x (2 * 8)

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))  # (2 * 7) x (2 * 7)

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)  # (4 * 8) x (4 * 8)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))  # (4 * 9) x (4 * 9), Qk + Qfk

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # --------------------------------------------------------
        # TODO: fill in the objectives here, you should be using cvxpy.quad_form() somehwhere
        
        # Objective part 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)  # # cvxpy.vec() - Flattens the matrix X into a vector in column-major order

        # Objective part 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective part 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)
        # --------------------------------------------------------

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))  # 4 x 9
        for t in range(self.config.TK):  # 8
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], 0.0  # reference steering angle is zero
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))  # 32 x 32
        B_block = block_diag(tuple(B_block))  # 32 x 16
        C_block = np.array(C_block)  # 32 x 1
        # creating the format of matrices

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape  # 32, 32
        self.Annz_k = cvxpy.Parameter(A_block.nnz)  # nnz: number of nonzero elements, nnz = 128
        data = np.ones(self.Annz_k.size)  # 128 x 1, size = 128, all elements are 1
        rows = A_block.row * n + A_block.col  # No. ? element in 32 x 32 matrix
        cols = np.arange(self.Annz_k.size)  # 128 elements that need to be care - diagonal & nonzero, 4 x 4 x 8
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))	 # (rows, cols)	data

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")
        # https://www.cvxpy.org/api_reference/cvxpy.atoms.affine.html#cvxpy.reshape

        # Same as A
        m, n = B_block.shape  # 32, 16 = 4 x 8, 2 x 8
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)  # nnz = 64
        data = np.ones(self.Bnnz_k.size)  # 64 = (4 x 2) x 8
        rows = B_block.row * n + B_block.col  # No. ? element in 32 x 16 matrix
        cols = np.arange(self.Bnnz_k.size)  # 0, 1, ... 63
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))  # (rows, cols)	data
        
        # sparse version instead of the old B_block
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        
        # real data
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # -------------------------------------------------------------
        # TODO: Constraint part 1:
        #       Add dynamics constraints to the optimization problem
        #       This constraint should be based on a few variables:
        #       self.xk, self.Ak_, self.Bk_, self.uk, and self.Ck_
        
        flatten_prev_xk = cvxpy.vec(self.xk[:, :-1])
        flatten_next_xk = cvxpy.vec(self.xk[:, 1:])
        # flatten_uk = cvxpy.diag(self.uk[:, :-1].flatten())
        # import pdb; pdb.set_trace()
        c1 = flatten_next_xk == self.Ak_ @ flatten_prev_xk + self.Bk_ @ cvxpy.vec(self.uk) + self.Ck_
        constraints.append(c1)
        
        # TODO: Constraint part 2:
        #       Add constraints on steering, change in steering angle
        #       cannot exceed steering angle speed limit. Should be based on:
        #       self.uk, self.config.MAX_DSTEER, self.config.DTK
        
        dsteering = cvxpy.diff(self.uk[1, :])
        c2_lower = -self.config.MAX_DSTEER * self.config.DTK <= dsteering
        c2_upper = dsteering <= self.config.MAX_DSTEER * self.config.DTK
        constraints.append(c2_lower)
        constraints.append(c2_upper)
        
        # TODO: Constraint part 3:
        #       Add constraints on upper and lower bounds of states and inputs
        #       and initial state constraint, should be based on:
        #       self.xk, self.x0k, self.config.MAX_SPEED, self.config.MIN_SPEED,
        #       self.uk, self.config.MAX_ACCEL, self.config.MAX_STEER
        
        # init state constraint
        c3 = self.xk[:, 0] == self.x0k
        constraints.append(c3)

        # state consraints
        speed = self.xk[2, :]
        c4_lower = self.config.MIN_SPEED <= speed
        c4_upper = speed <= self.config.MAX_SPEED
        constraints.append(c4_lower)
        constraints.append(c4_upper)

        # input constraints
        steering = self.uk[1, :]
        c5_lower = self.config.MIN_STEER <= steering
        c5_upper = steering <= self.config.MAX_STEER
        constraints.append(c5_lower)
        constraints.append(c5_upper)

        acc = self.uk[0, :]
        c6_lower = -self.config.MAX_ACCEL <= acc
        c6_upper = acc <= self.config.MAX_ACCEL
        constraints.append(c6_lower)
        constraints.append(c6_upper)

        # -------------------------------------------------------------
        # Linearized soft-min composite CBF obstacle avoidance constraint
        # h_soft_lin = a_x * x + a_y * y + a_c
        for t in range(1, self.config.TK + 1):
            h_soft_lin = (
                self.cbf_soft_ax_k[t] * self.xk[0, t]
                + self.cbf_soft_ay_k[t] * self.xk[1, t]
                + self.cbf_soft_ac_k[t]
            )
            constraints.append(h_soft_lin >= 0.0)

                # Create the optimization problem in CVXPY and setup the workspace
                # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        cyaw = cyaw.copy()

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((self.config.NXK, self.config.TK + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        _, _, _, ind = nearest_point(np.array([state.x, state.y]), np.array([cx, cy]).T)

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]

        # based on current velocity, distance traveled on the ref line between time steps
        preview_speed = max(abs(state.v), self.config.MIN_PREVIEW_SPEED)
        travel = preview_speed * self.config.DTK
        dind = max(travel / self.config.dlk, self.config.MIN_PREVIEW_DIND)
        #dind = 2
        ind_list = int(ind) + np.insert(
            np.cumsum(np.repeat(dind, self.config.TK)), 0, 0
        ).astype(int)
        ind_list[ind_list >= ncourse] -= ncourse
        ref_traj[0, :] = cx[ind_list]
        ref_traj[1, :] = cy[ind_list]
        ref_traj[2, :] = sp[ind_list]

        angle_thres = 4.5
        # https://edstem.org/us/courses/34340/discussion/2817574

        for i in range(len(cyaw)):
            if cyaw[i] - state.yaw > angle_thres:
                cyaw[i] -= 2*np.pi
            if state.yaw - cyaw[i] > angle_thres:
                cyaw[i] += 2*np.pi

        # cyaw[cyaw - state.yaw > angle_thres] = np.abs(
        #     cyaw[cyaw - state.yaw > angle_thres] - (2 * np.pi)
        # )
        # cyaw[cyaw - state.yaw < -angle_thres] = np.abs(
        #     cyaw[cyaw - state.yaw < -angle_thres] + (2 * np.pi)
        # )
        ref_traj[3, :] = cyaw[ind_list]

        print("ref_yaw ={}, cur_yaw ={}".format(cyaw[ind], state.yaw))
        print(" ")

        return ref_traj

    def predict_motion(self, x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.config.TK + 1)):
            state = self.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(self, state, a, delta):

        # input check
        if delta >= self.config.MAX_STEER:
            delta = self.config.MAX_STEER
        elif delta <= -self.config.MAX_STEER:
            delta = -self.config.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.config.DTK
        state.y = state.y + state.v * math.sin(state.yaw) * self.config.DTK
        state.yaw = (
            state.yaw + (state.v / self.config.WB) * math.tan(delta) * self.config.DTK
        )
        state.v = state.v + a * self.config.DTK

        if state.v > self.config.MAX_SPEED:
            state.v = self.config.MAX_SPEED
        elif state.v < self.config.MIN_SPEED:
            state.v = self.config.MIN_SPEED

        return state

    def get_model_matrix(self, v, phi, delta):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * math.cos(phi)
        A[0, 3] = -self.config.DTK * v * math.sin(phi)
        A[1, 2] = self.config.DTK * math.sin(phi)
        A[1, 3] = self.config.DTK * v * math.cos(phi)
        A[3, 2] = self.config.DTK * math.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * math.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * math.sin(phi) * phi
        C[1] = -self.config.DTK * v * math.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * math.cos(delta) ** 2)

        return A, B, C  # 4 x 4, 4 x 2, 4 x 1


    def get_valid_obstacles(self):
        """
        Return valid obstacles as (x, y, radius).

        The radius is the physical radius of the object. The CBF will inflate it by
        the car radius and safety margin.
        """
        obstacles = [
            (self.obs_1_x, self.obs_1_y, self.obs_1_r),
            (self.obs_2_x, self.obs_2_y, self.obs_2_r),
            (self.obs_3_x, self.obs_3_y, self.obs_3_r),
        ]

        valid_obstacles = []
        for ox, oy, radius in obstacles:
            if (
                abs(ox) < self.config.FAR_OBS_THRESHOLD
                and abs(oy) < self.config.FAR_OBS_THRESHOLD
                and radius > 0.0
            ):
                valid_obstacles.append((ox, oy, radius))

        return valid_obstacles

    def get_safe_radius(self, obs_radius):
        """CBF inflated radius for one circular obstacle."""
        return self.car_radius + obs_radius + self.config.SAFETY_MARGIN



    def project_cbf_linearization_point(self, x_bar, y_bar, obstacles):
        """
        If the nominal linearization point is inside the closest obstacle's
        safety region, move the linearization point to the outside boundary.

        obstacles must be a list of (ox, oy, obs_radius).
        """

        if len(obstacles) == 0:
            return x_bar, y_bar

        # Closest obstacle after sorting
        ox, oy, obs_radius = obstacles[0]

        dx = x_bar - ox
        dy = y_bar - oy
        dist = math.sqrt(dx * dx + dy * dy)

        # Project slightly outside the inflated safety boundary
        buffer = 0.1
        r_lin = self.get_safe_radius(obs_radius) + buffer

        # If already outside, do not modify the point
        if dist >= r_lin:
            return x_bar, y_bar

        # Avoid division by zero if exactly at obstacle center
        if dist < 1e-6:
            dx = 0.0
            dy = -1.0
            dist = 1.0

        x_safe = ox + r_lin * dx / dist
        y_safe = oy + r_lin * dy / dist

        return x_safe, y_safe



    def build_softmin_cbf_params(self, path_predict):
        """
        Build affine approximation of the soft-min composite barrier:
            h_soft(x) = -(1/kappa) * log(sum_i exp(-kappa * h_i(x)))
        where
            h_i(x,y) = (x - ox)^2 + (y - oy)^2 - r_safe^2
        Then linearize h_soft at a projected nominal point (x_bar, y_bar):
            h_soft_lin(x,y) = a_x * x + a_y * y + a_c
        The MPC enforces:
            h_soft_lin(x,y) >= 0
        """

        cbf_soft_ax = np.zeros(self.config.TK + 1)
        cbf_soft_ay = np.zeros(self.config.TK + 1)
        cbf_soft_ac = (
            np.ones(self.config.TK + 1)
            * self.config.SOFTMIN_INACTIVE_VALUE
        )

        obstacles_all = self.get_valid_obstacles()

        if len(obstacles_all) == 0:
            return cbf_soft_ax, cbf_soft_ay, cbf_soft_ac

        kappa = self.config.SOFTMIN_KAPPA
        for t in range(1, self.config.TK + 1):
            x_nom = path_predict[0, t]
            y_nom = path_predict[1, t]

            # Use only closest obstacles in the soft-min
            obstacles_sorted = sorted(
                obstacles_all,
                key=lambda obs: (x_nom - obs[0]) ** 2 + (y_nom - obs[1]) ** 2
            )

            obstacles = obstacles_sorted[
                :min(self.config.SOFTMIN_TOPK, len(obstacles_sorted))
            ]

            # Debug only. This should not control the computation.
            if t == 1 or t == self.config.TK // 2:
                closest_ox, closest_oy, closest_obs_r = obstacles[0]
                closest_safe_r = self.get_safe_radius(closest_obs_r)
                dist_nom = math.sqrt(
                    (x_nom - closest_ox) ** 2
                    + (y_nom - closest_oy) ** 2
                )
                h_nom = dist_nom ** 2 - closest_safe_r ** 2

                print(
                    f"[CBF DEBUG] t={t}, "
                    f"x_nom={x_nom:.3f}, y_nom={y_nom:.3f}, "
                    f"obs=({closest_ox:.3f},{closest_oy:.3f}), "
                    f"obs_r={closest_obs_r:.3f}, safe_r={closest_safe_r:.3f}, "
                    f"dist={dist_nom:.3f}, h_nom={h_nom:.3f}"
                )

            # IMPORTANT:
            # Do not linearize from a point deep inside the unsafe set.
            x_bar, y_bar = self.project_cbf_linearization_point(
                x_nom,
                y_nom,
                obstacles
            )

            h_list = []
            gx_list = []
            gy_list = []

            for ox, oy, obs_radius in obstacles:
                dx = x_bar - ox
                dy = y_bar - oy
                safe_r = self.get_safe_radius(obs_radius)

                h_i = dx * dx + dy * dy - safe_r ** 2
                gx_i = 2.0 * dx
                gy_i = 2.0 * dy

                h_list.append(h_i)
                gx_list.append(gx_i)
                gy_list.append(gy_i)

            h_arr = np.asarray(h_list)
            gx_arr = np.asarray(gx_list)
            gy_arr = np.asarray(gy_list)

            h_min = np.min(h_arr)
            exp_shift = np.exp(-kappa * (h_arr - h_min))
            weights = exp_shift / np.sum(exp_shift)

            h_soft = h_min - (1.0 / (self.config.BETA_1*kappa)) * np.log(np.sum(exp_shift))

            gx_soft = np.sum(weights * gx_arr)
            gy_soft = np.sum(weights * gy_arr)

            cbf_soft_ax[t] = gx_soft
            cbf_soft_ay[t] = gy_soft
            cbf_soft_ac[t] = h_soft - gx_soft * x_bar - gy_soft * y_bar

        return cbf_soft_ax, cbf_soft_ay, cbf_soft_ac
    


    def mpc_prob_solve(self, ref_traj, path_predict, x0, od):
        self.x0k.value = x0

        A_block = []
        B_block = []
        C_block = []
        for t in range(self.config.TK):
            A, B, C = self.get_model_matrix(
                path_predict[2, t], path_predict[3, t], od[t]
            )
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        cbf_soft_ax, cbf_soft_ay, cbf_soft_ac = self.build_softmin_cbf_params(path_predict)

        self.cbf_soft_ax_k.value = cbf_soft_ax
        self.cbf_soft_ay_k.value = cbf_soft_ay
        self.cbf_soft_ac_k.value = cbf_soft_ac

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        #self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        try:
            self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        except cvxpy.error.SolverError as e:
            print(f"OSQP solver error: {e}")
            self.mpc_fail_count += 1
            return None, None, None, None, None, None


        if (
            self.MPC_prob.status == cvxpy.OPTIMAL
            or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE
        ):
            ox = np.array(self.xk.value[0, :]).flatten()
            oy = np.array(self.xk.value[1, :]).flatten()
            ov = np.array(self.xk.value[2, :]).flatten()
            oyaw = np.array(self.xk.value[3, :]).flatten()
            oa = np.array(self.uk.value[0, :]).flatten()
            odelta = np.array(self.uk.value[1, :]).flatten()

            # save last successful solution
            self.last_ok_oa = oa.copy()
            self.last_ok_odelta = odelta.copy()
            self.last_mpc_ok = True
            self.mpc_fail_count = 0

        else:
            print(f"MPC solve failed. status={self.MPC_prob.status}")
            self.mpc_fail_count += 1
            return None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov




    def linear_mpc_control(self, ref_path, x0, oa, od):
        """
        Sequential convex MPC.

        We solve several convex QPs.
        Each QP is built by linearizing the dynamics and soft-min CBF
        around the latest predicted trajectory.
        """

        # -----------------------------
        # 1) Initial control guess
        # -----------------------------
        if oa is None or od is None:
            oa = np.ones(self.config.TK) * 0.2
            od = np.zeros(self.config.TK)
        else:
            oa = np.asarray(oa).copy()
            od = np.asarray(od).copy()

        # Number of SCP iterations.
        max_scp_iter = self.config.SCP_MAX_ITER

        # Damping prevents large jumps between SCP iterations.
        alpha = self.config.SCP_ALPHA

        best_a = oa.copy()
        best_delta = od.copy()
        best_x = None
        best_y = None
        best_yaw = None
        best_v = None
        final_path_predict = None

        for scp_it in range(max_scp_iter):

            # ---------------------------------------
            # 2) Nonlinear rollout with current guess
            # ---------------------------------------
            path_predict = self.predict_motion(x0, oa, od, ref_path)
            final_path_predict = path_predict

            # ---------------------------------------
            # 3) Solve one convexified MPC problem
            # Dynamics and CBF are linearized inside
            # mpc_prob_solve() using path_predict and od
            # ---------------------------------------
            new_a, new_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = self.mpc_prob_solve(
                ref_path, path_predict, x0, od)

            # If solver returned None, stop SCP
            if new_a is None or new_delta is None:
                break

            new_a = np.asarray(new_a).copy()
            new_delta = np.asarray(new_delta).copy()

            # ---------------------------------------
            # 4) Damped update
            # ---------------------------------------
            da = np.linalg.norm(new_a - oa)
            dd = np.linalg.norm(new_delta - od)

            oa = alpha * new_a + (1.0 - alpha) * oa
            od = alpha * new_delta + (1.0 - alpha) * od

            best_a = oa.copy()
            best_delta = od.copy()
            best_x = mpc_x
            best_y = mpc_y
            best_yaw = mpc_yaw
            best_v = mpc_v

            # ---------------------------------------
            # 5) Optional convergence check
            # ---------------------------------------
            if da < 1e-2 and dd < 1e-2:
                break

        if best_x is None:
            print("SCP hard soft-min MPC infeasible. Using safe stop.")

            best_a = np.full(self.config.TK, -self.config.MAX_ACCEL)
            best_delta = np.zeros(self.config.TK)

            final_path_predict = self.predict_motion(x0, best_a, best_delta, ref_path)

            best_x = final_path_predict[0, :]
            best_y = final_path_predict[1, :]
            best_v = final_path_predict[2, :]
            best_yaw = final_path_predict[3, :]

        # Visualize only once, not inside every SCP iteration
        if final_path_predict is not None:
            self.visualize_pred_path_in_rviz(final_path_predict)

        return best_a, best_delta, best_x, best_y, best_yaw, best_v, final_path_predict





    # visualization
    def make_cylinder_marker(self, marker_id, ns, x, y, radius, height, rgba, z=0.04):
        """Create a circular RViz marker using a thin cylinder."""
        marker = Marker()
        marker.header.frame_id = 'map'   # use 'map' for ROS 2/tf2 compatibility
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0

        # RViz cylinder scale.x and scale.y are diameters, not radius.
        marker.scale.x = 2.0 * float(radius)
        marker.scale.y = 2.0 * float(radius)
        marker.scale.z = float(height)

        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        return marker

    def visualize_obstacles_in_rviz(self):
        """
        Publish the obstacles in RViz.

        For each obstacle:
          - solid cylinder: physical obstacle radius
          - transparent cylinder: inflated CBF safety radius
        """
        marker_array = MarkerArray()
        obstacles = self.get_valid_obstacles()

        # Delete previous markers first. This prevents old obstacle markers from staying
        # in RViz if the number of valid obstacles changes.
        delete_marker = Marker()
        delete_marker.header.frame_id = 'map'
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, (ox, oy, obs_radius) in enumerate(obstacles):
            safe_radius = self.get_safe_radius(obs_radius)

            # Physical obstacle: red solid disk/cylinder.
            marker_array.markers.append(
                self.make_cylinder_marker(
                    marker_id=2 * i,
                    ns='obstacle_physical_radius',
                    x=ox,
                    y=oy,
                    radius=obs_radius,
                    height=self.config.OBSTACLE_MARKER_HEIGHT,
                    rgba=(1.0, 0.0, 0.0, 0.85),
                    z=0.04,
                )
            )

            # CBF inflated safety region: transparent orange disk/cylinder.
            marker_array.markers.append(
                self.make_cylinder_marker(
                    marker_id=2 * i + 1,
                    ns='obstacle_cbf_safe_radius',
                    x=ox,
                    y=oy,
                    radius=safe_radius,
                    height=0.02,
                    rgba=(1.0, 0.55, 0.0, 0.25),
                    z=0.02,
                )
            )

        self.vis_obstacles_pub.publish(marker_array)




    def clear_hybrid_astar_path_in_rviz(self):
        """Remove stale Hybrid A* path from RViz when the planner is inactive."""
        marker = Marker()
        marker.header.frame_id = '/map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'hybrid_astar_path'
        marker.id = 0
        marker.action = Marker.DELETE
        self.vis_hybrid_path_pub.publish(marker)

    def visualize_hybrid_astar_path_in_rviz(self, hybrid_path):
        """Publish the raw Hybrid A* path in RViz."""
        self.vis_hybrid_path_msg.points = []
        self.vis_hybrid_path_msg.header.frame_id = '/map'
        self.vis_hybrid_path_msg.header.stamp = self.get_clock().now().to_msg()
        self.vis_hybrid_path_msg.type = Marker.LINE_STRIP
        self.vis_hybrid_path_msg.action = Marker.ADD
        self.vis_hybrid_path_msg.color.r = 0.0
        self.vis_hybrid_path_msg.color.g = 1.0
        self.vis_hybrid_path_msg.color.b = 0.0
        self.vis_hybrid_path_msg.color.a = 1.0
        self.vis_hybrid_path_msg.scale.x = 0.05
        self.vis_hybrid_path_msg.id = 0
        self.vis_hybrid_path_msg.ns = 'hybrid_astar_path'

        for i in range(hybrid_path.shape[0]):
            point = Point(x=float(hybrid_path[i, 0]), y=float(hybrid_path[i, 1]), z=0.25)
            self.vis_hybrid_path_msg.points.append(point)

        self.vis_hybrid_path_pub.publish(self.vis_hybrid_path_msg)

    def visualize_waypoints_in_rviz(self):
        self.vis_waypoints_msg.points = []
        self.vis_waypoints_msg.header.frame_id = '/map'
        self.vis_waypoints_msg.type = Marker.POINTS
        self.vis_waypoints_msg.color.g = 0.75
        self.vis_waypoints_msg.color.a = 1.0
        self.vis_waypoints_msg.scale.x = 0.05
        self.vis_waypoints_msg.scale.y = 0.05
        self.vis_waypoints_msg.id = 0
        for i in range(self.waypoints.shape[0]):
            #point = Point(x = self.waypoints[i, 1], y = self.waypoints[i, 2], z = 0.1)
            point = Point(x = self.ref_pos_x[i], y = self.ref_pos_y[i], z = 0.1)
            #point = Point(x = self.waypoints[i, 0], y = self.waypoints[i, 1], z = 0.1) # f1tenth AutoDrive
            self.vis_waypoints_msg.points.append(point)
        
        # self.vis_waypoints_pub.publish(self.vis_waypoints_msg)

    def visualize_ref_traj_in_rviz(self, ref_traj):
        # visualize the path data in the world frame
        self.vis_ref_traj_msg.points = []
        self.vis_ref_traj_msg.header.frame_id = '/map'
        self.vis_ref_traj_msg.type = Marker.LINE_STRIP
        self.vis_ref_traj_msg.color.b = 0.75
        self.vis_ref_traj_msg.color.a = 1.0
        self.vis_ref_traj_msg.scale.x = 0.08
        self.vis_ref_traj_msg.scale.y = 0.08
        self.vis_ref_traj_msg.id = 0
        for i in range(ref_traj.shape[1]):
            point = Point(x = ref_traj[0, i], y = ref_traj[1, i], z = 0.2)
            self.vis_ref_traj_msg.points.append(point)
        
        self.vis_ref_traj_pub.publish(self.vis_ref_traj_msg)

    def visualize_pred_path_in_rviz(self, path_predict):
        # visualize the path data in the world frame
        self.vis_pred_path_msg.points = []
        self.vis_pred_path_msg.header.frame_id = '/map'
        self.vis_pred_path_msg.type = Marker.LINE_STRIP
        self.vis_pred_path_msg.color.r = 0.75
        self.vis_pred_path_msg.color.a = 1.0
        self.vis_pred_path_msg.scale.x = 0.08
        self.vis_pred_path_msg.scale.y = 0.08
        self.vis_pred_path_msg.id = 0
        for i in range(path_predict.shape[1]):
            point = Point(x = path_predict[0, i], y = path_predict[1, i], z = 0.2)
            self.vis_pred_path_msg.points.append(point)
        
        self.vis_pred_path_pub.publish(self.vis_pred_path_msg)


def main(args=None):
    rclpy.init(args=args)
    print("MPC Initialized")
    mpc_node = MPC()
    rclpy.spin(mpc_node)

    mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()