#!/usr/bin/env python3
"""
tube_robust_mpc_node_configured_v3_fast.py

ROS 2 Tube / Robust MPC node organized around your current F1TENTH setup:

Current defaults preserved from your uploaded node:
- node name:                 mpc_node
- map/trajectory name:        square_trajectory.csv
- real pose topic:            /optitrack/object_529/pose
- sim odom topic:             /ego_racecar/odom
- obstacle odom topics:       /optitrack/object_527/odom, /optitrack/object_528/odom
- drive topic:                /drive
- RViz marker topics:         /waypoints_marker, /ref_traj_marker, /pred_path_marker

Main difference from your soft-min CBF node:
- The MPC core is a tube/robust MPC tracker with invariant-set tightening.
- Obstacles are not inside a soft-min CBF in the main QP.
- The MPC QP is built once with CVXPY Parameters for fast repeated solves.
- The optional post-MPC safety filter is disabled by default for speed.

IMPORTANT SAFETY NOTE:
A mathematical guarantee is only valid under the assumptions used by tube MPC:
1) the additive model error is inside DISTURBANCE_BOUNDS,
2) localization delay/noise is inside the same bound or accounted for,
3) the tightened optimization problem remains feasible,
4) the low-level vehicle tracks speed/steering commands within actuator limits,
5) the square path is interpreted as a corridor, not an exact sharp-corner curve.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cvxpy as cp
import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.interpolate import interp1d

import rclpy
from rclpy.node import Node

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float64
from visualization_msgs.msg import Marker


# =============================================================================
# Utility functions
# =============================================================================


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def nearest_index(px: float, py: float, xs: np.ndarray, ys: np.ndarray) -> int:
    return int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))


def robust_block_rpi_box(Acl: np.ndarray, wbar: np.ndarray, max_terms: int = 800, tol: float = 1e-11) -> np.ndarray:
    """
    Conservative box outer approximation of the RPI set

        Z ~= W (+) Acl W (+) Acl^2 W (+) ...

    for box disturbance W = {w : |w_i| <= wbar_i}.

    If eta_{k+1} = Acl eta_k + w_k and eta_0 in Z,
    then eta_k remains approximately inside Z for bounded w_k.
    """
    n = Acl.shape[0]
    zbar = np.zeros(n)
    Ak = np.eye(n)

    for _ in range(max_terms):
        term = np.abs(Ak) @ wbar
        zbar += term
        if float(np.max(term)) < tol:
            break
        Ak = Acl @ Ak

    # Small engineering margin for numerical and truncation error.
    return 1.10 * zbar


def kinematic_step(x: np.ndarray, u: np.ndarray, dt: float, wb: float, max_steer: float) -> np.ndarray:
    """Discrete kinematic bicycle step. x=[x,y,v,yaw], u=[accel, steer_angle]."""
    px, py, v, yaw = x
    a, delta = u
    delta = float(np.clip(delta, -max_steer, max_steer))

    xn = np.zeros(4)
    xn[0] = px + dt * v * math.cos(yaw)
    xn[1] = py + dt * v * math.sin(yaw)
    xn[2] = v + dt * a
    xn[3] = yaw + dt * v * math.tan(delta) / wb
    return xn


def linearize_kinematic(v: float, yaw: float, delta: float, dt: float, wb: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Linearization of the discrete kinematic bicycle model around
    xbar=[xbar,ybar,v,yaw] and ubar=[abar,delta].

    Returns A, B, C such that:
        x_{k+1} ~= A x_k + B u_k + C

    The affine C does not depend on xbar position, only v/yaw/delta.
    """
    cdelta = max(1e-6, math.cos(delta))

    A = np.eye(4)
    A[0, 2] = dt * math.cos(yaw)
    A[0, 3] = -dt * v * math.sin(yaw)
    A[1, 2] = dt * math.sin(yaw)
    A[1, 3] = dt * v * math.cos(yaw)
    A[3, 2] = dt * math.tan(delta) / wb

    B = np.zeros((4, 2))
    B[2, 0] = dt
    B[3, 1] = dt * v / (wb * cdelta**2)

    C = np.zeros(4)
    C[0] = dt * v * math.sin(yaw) * yaw
    C[1] = -dt * v * math.cos(yaw) * yaw
    C[3] = -dt * v * delta / (wb * cdelta**2)

    return A, B, C


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discrete LQR gain for eta_{k+1}=A eta_k+B v_k.
    Returns K such that v_k = K eta_k.
    """
    P = solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MPCConfig:
    # State and input dimensions
    NXK: int = 4                       # x = [x, y, v, yaw]
    NU: int = 2                        # u = [acceleration, steering angle]
    TK: int = 15

    # Your current cost matrices
    Rk: np.ndarray = field(default_factory=lambda: np.diag([5.0, 35.0]))
    Rdk: np.ndarray = field(default_factory=lambda: np.diag([1.0, 35.0]))
    Qk: np.ndarray = field(default_factory=lambda: np.diag([60.0, 50.0, 10.0, 50.0]))
    Qfk: np.ndarray = field(default_factory=lambda: np.diag([60.0, 50.0, 10.0, 50.0]))

    # Your latest uploaded physical parameters
    DTK: float = 0.1
    dlk: float = 0.03
    LENGTH: float = 0.44
    WIDTH: float = 0.280
    WB: float = 0.330
    MIN_STEER: float = -0.353
    MAX_STEER: float = 0.353
    MAX_DSTEER: float = np.deg2rad(180.0)
    MAX_SPEED: float = 1.5
    MIN_SPEED: float = 0.0
    MAX_ACCEL: float = 5.0

    # Current obstacle approximation from your node
    OBS_RADIUS: float = 0.20
    SAFETY_MARGIN: float = 0.15
    FAR_OBS_THRESHOLD: float = 1.0e5

    # Reference speed scheduling
    BASE_SPEED: float = 0.70
    CORNER_SPEED: float = 0.12        # Important for hard 90-degree square corners
    CORNER_WINDOW_M: float = 0.45
    CORNER_YAW_JUMP_RAD: float = 0.35
    MIN_PREVIEW_SPEED: float = 0.08

    # Corridor from CSV widths. Your CSV uses 0.7 m left/right.
    DEFAULT_HALF_WIDTH: float = 0.70
    CORRIDOR_MARGIN: float = 0.10
    LONGITUDINAL_ERROR_MAX: float = 0.80
    YAW_ERROR_MAX: float = 0.65

    # Tube MPC design.
    # These are intentionally small because a global x-y-yaw tube is conservative.
    # Estimate and retune these from logs before claiming a hardware guarantee.
    TUBE_DESIGN_SPEED: float = 0.70
    DISTURBANCE_BOUNDS: np.ndarray = field(
        default_factory=lambda: np.array([0.002, 0.002, 0.010, 0.004], dtype=float)
    )

    # Terminal set around the final reference state.
    # The terminal constraint is disabled by default for your sharp square.
    # A hard terminal invariant constraint is theoretically nice, but on a
    # sharp 90-degree square it often makes the QP infeasible before the car
    # reaches the rounded/feasible part of the path. After the controller works,
    # you can enable it again with a larger rounded-square reference.
    TERMINAL_ERROR_BOX: np.ndarray = field(
        default_factory=lambda: np.array([1.20, 1.20, 0.80, 1.20], dtype=float)
    )
    USE_TERMINAL_CONSTRAINT: bool = False

    # Practical feasibility protection. These slacks are only for reference/corridor
    # tracking, not for actuator limits. If a slack becomes large, you are outside
    # the certified tube/corridor assumptions. The node publishes the largest slack.
    SOFTEN_CORRIDOR_AND_YAW: bool = True
    SOFTEN_TERMINAL: bool = True
    CORRIDOR_SLACK_WEIGHT: float = 5.0e4
    YAW_SLACK_WEIGHT: float = 1.0e4
    TERMINAL_SLACK_WEIGHT: float = 2.0e4

    # Optional local safety filter after Tube MPC
    ENABLE_OBSTACLE_FILTER: bool = False
    CBF_GAMMA: float = 0.25
    CBF_ACTIVATION_DISTANCE: float = 2.0

    # Fallback only if the optimization is infeasible or the pose is missing.
    BACKUP_BRAKE: float = -2.5

    # Solver-speed options. For hardware, 1e-3 is usually much faster than 1e-4.
    # Start with this, then tighten after the controller is stable.
    OSQP_EPS_ABS: float = 1.0e-3
    OSQP_EPS_REL: float = 1.0e-3
    OSQP_MAX_ITER: int = 4000
    OSQP_POLISH: bool = False


@dataclass
class State:
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    yaw: float = 0.0


@dataclass
class Trajectory:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    s: np.ndarray
    v: np.ndarray
    w_right: np.ndarray
    w_left: np.ndarray


# =============================================================================
# Main ROS2 node
# =============================================================================


class TubeRobustMPC(Node):
    def __init__(self):
        # Keep your current node name.
        super().__init__("mpc_node")

        self.config = MPCConfig()

        # ---------------------------------------------------------------------
        # Your current setup defaults
        # ---------------------------------------------------------------------
        self.is_real = True
        self.map_name = "square_trajectory"
        self.enable_drive = True

        self.pose_topic_real = "/optitrack/object_529/pose"
        self.pose_topic_sim = "/ego_racecar/odom"
        self.drive_topic = "/drive"
        self.obs_1_topic = "/optitrack/object_527/odom"
        self.obs_2_topic = "/optitrack/object_528/odom"
        self.obs_3_topic = "/optitrack/object_526/odom"

        self.vis_ref_traj_topic = "/ref_traj_marker"
        self.vis_waypoints_topic = "/waypoints_marker"
        self.vis_pred_path_topic = "/pred_path_marker"
        self.frame_id = "/map"  # kept from your code

        # Optional ROS parameters while preserving your defaults.
        self.declare_parameter("is_real", self.is_real)
        self.declare_parameter("map_name", self.map_name)
        self.declare_parameter("enable_drive", self.enable_drive)
        self.declare_parameter("pose_topic_real", self.pose_topic_real)
        self.declare_parameter("pose_topic_sim", self.pose_topic_sim)
        self.declare_parameter("drive_topic", self.drive_topic)
        self.declare_parameter("obs_1_topic", self.obs_1_topic)
        self.declare_parameter("obs_2_topic", self.obs_2_topic)
        self.declare_parameter("obs_3_topic", self.obs_3_topic)
        self.declare_parameter("base_speed", self.config.BASE_SPEED)
        self.declare_parameter("corner_speed", self.config.CORNER_SPEED)
        self.declare_parameter("enable_obstacle_filter", self.config.ENABLE_OBSTACLE_FILTER)

        self.is_real = bool(self.get_parameter("is_real").value)
        self.map_name = str(self.get_parameter("map_name").value)
        self.enable_drive = bool(self.get_parameter("enable_drive").value)
        self.pose_topic_real = str(self.get_parameter("pose_topic_real").value)
        self.pose_topic_sim = str(self.get_parameter("pose_topic_sim").value)
        self.drive_topic = str(self.get_parameter("drive_topic").value)
        self.obs_1_topic = str(self.get_parameter("obs_1_topic").value)
        self.obs_2_topic = str(self.get_parameter("obs_2_topic").value)
        self.obs_3_topic = str(self.get_parameter("obs_3_topic").value)
        self.config.BASE_SPEED = float(self.get_parameter("base_speed").value)
        self.config.CORNER_SPEED = float(self.get_parameter("corner_speed").value)
        self.config.ENABLE_OBSTACLE_FILTER = bool(self.get_parameter("enable_obstacle_filter").value)

        # ---------------------------------------------------------------------
        # Load and preprocess your square_trajectory.csv
        # ---------------------------------------------------------------------
        self.trajectory = self.load_trajectory_csv(self.map_name + ".csv")

        # ---------------------------------------------------------------------
        # Tube MPC invariant set design
        # ---------------------------------------------------------------------
        self.K_lqr, self.P_lqr, self.z_box, self.input_tightening = self.design_tube()

        # The tightened terminal box is the nominal terminal box.
        self.terminal_box_nom = np.maximum(self.config.TERMINAL_ERROR_BOX - self.z_box, 0.05)

        self.get_logger().info(f"Tube z_box = {self.z_box}")
        self.get_logger().info(f"Input tightening |K|Z = {self.input_tightening}")
        self.get_logger().info(f"Nominal terminal box = {self.terminal_box_nom}")

        # Build the CVXPY problem only once. The online callback only updates
        # Parameters and calls OSQP. This is the main speed improvement.
        self.build_mpc_problem()

        # ---------------------------------------------------------------------
        # ROS publishers/subscribers with your topic names
        # ---------------------------------------------------------------------
        pose_topic = self.pose_topic_real if self.is_real else self.pose_topic_sim
        pose_msg_type = PoseStamped if self.is_real else Odometry
        self.pose_sub = self.create_subscription(pose_msg_type, pose_topic, self.pose_callback, 1)

        self.obs_1 = self.create_subscription(Odometry, self.obs_1_topic, self.obs_1_callback, 10)
        self.obs_2 = self.create_subscription(Odometry, self.obs_2_topic, self.obs_2_callback, 10)
        # Keep obs_3 available but disabled by default in case you uncomment it later.
        # self.obs_3 = self.create_subscription(Odometry, self.obs_3_topic, self.obs_3_callback, 10)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 1)
        self.drive_msg = AckermannDriveStamped()

        self.vis_waypoints_pub = self.create_publisher(Marker, self.vis_waypoints_topic, 1)
        self.vis_ref_traj_pub = self.create_publisher(Marker, self.vis_ref_traj_topic, 1)
        self.vis_pred_path_pub = self.create_publisher(Marker, self.vis_pred_path_topic, 1)

        # Optional diagnostics. These do not change your current interface.
        self.feasible_pub = self.create_publisher(Bool, "~/mpc_feasible", 1)
        self.solve_time_pub = self.create_publisher(Float64, "~/solve_time_ms", 1)
        self.margin_pub = self.create_publisher(Float64, "~/min_tightened_margin", 1)
        self.slack_pub = self.create_publisher(Float64, "~/max_constraint_slack", 1)

        # ---------------------------------------------------------------------
        # State and memory
        # ---------------------------------------------------------------------
        self.latest_state: Optional[State] = None
        self.prev_pose_time: Optional[float] = None
        self.prev_pose_x: Optional[float] = None
        self.prev_pose_y: Optional[float] = None
        self.v_filt: Optional[float] = None

        self.obs_1_x = 1.0e6
        self.obs_1_y = 1.0e6
        self.obs_2_x = 1.0e6
        self.obs_2_y = 1.0e6
        self.obs_3_x = 1.0e6
        self.obs_3_y = 1.0e6

        self.last_nominal_u = np.zeros(2)  # [accel, steer]
        self.last_actual_steer = 0.0
        self.last_mpc_ok = False
        self.mpc_fail_count = 0

        self.visualize_waypoints_in_rviz()
        self.get_logger().info("Configured Tube/Robust MPC node started.")
        self.get_logger().info(f"Pose topic: {pose_topic}")
        self.get_logger().info(f"Drive topic: {self.drive_topic}")
        self.get_logger().info(f"Trajectory: {self.map_name}.csv")

    # -------------------------------------------------------------------------
    # Trajectory loading and preprocessing
    # -------------------------------------------------------------------------

    def load_trajectory_csv(self, csv_name: str) -> Trajectory:
        if not os.path.exists(csv_name):
            raise FileNotFoundError(
                f"Could not find {csv_name}. Put it in the directory where you run ros2 run, "
                f"or pass map_name as a ROS parameter."
            )

        raw = np.loadtxt(csv_name, delimiter=",", skiprows=1)
        x_raw = raw[:, 0]
        y_raw = raw[:, 1]

        if raw.shape[1] >= 4:
            w_right_raw = raw[:, 2]
            w_left_raw = raw[:, 3]
        else:
            w_right_raw = np.ones_like(x_raw) * self.config.DEFAULT_HALF_WIDTH
            w_left_raw = np.ones_like(x_raw) * self.config.DEFAULT_HALF_WIDTH

        # Close loop if needed.
        if np.hypot(x_raw[0] - x_raw[-1], y_raw[0] - y_raw[-1]) > 1e-6:
            x_raw = np.r_[x_raw, x_raw[0]]
            y_raw = np.r_[y_raw, y_raw[0]]
            w_right_raw = np.r_[w_right_raw, w_right_raw[0]]
            w_left_raw = np.r_[w_left_raw, w_left_raw[0]]

        ds_raw = np.hypot(np.diff(x_raw), np.diff(y_raw))
        s_raw = np.r_[0.0, np.cumsum(ds_raw)]
        total_length = float(s_raw[-1])

        s_new = np.arange(0.0, total_length, self.config.dlk)

        fx = interp1d(s_raw, x_raw, kind="linear")
        fy = interp1d(s_raw, y_raw, kind="linear")
        fw_right = interp1d(s_raw, w_right_raw, kind="linear")
        fw_left = interp1d(s_raw, w_left_raw, kind="linear")

        x = fx(s_new)
        y = fy(s_new)
        w_right = fw_right(s_new)
        w_left = fw_left(s_new)

        dx = np.gradient(x, self.config.dlk)
        dy = np.gradient(y, self.config.dlk)
        yaw = np.unwrap(np.arctan2(dy, dx))

        v_ref = np.ones_like(x) * self.config.BASE_SPEED

        # Slow down near the sharp 90-degree corners of the square.
        # This is not optional if you want reliable tracking of a hard square.
        dyaw = np.abs(np.diff(np.unwrap(yaw), prepend=yaw[0]))
        corner_indices = np.where(dyaw > self.config.CORNER_YAW_JUMP_RAD)[0]
        half_window = max(1, int(round(self.config.CORNER_WINDOW_M / self.config.dlk)))
        for idx in corner_indices:
            lo = max(0, idx - half_window)
            hi = min(len(v_ref), idx + half_window + 1)
            v_ref[lo:hi] = np.minimum(v_ref[lo:hi], self.config.CORNER_SPEED)

        self.get_logger().info(
            f"Loaded {csv_name}: {len(x)} resampled points, length={total_length:.2f} m, "
            f"base_speed={self.config.BASE_SPEED:.2f}, corner_speed={self.config.CORNER_SPEED:.2f}"
        )

        return Trajectory(x=x, y=y, yaw=yaw, s=s_new, v=v_ref, w_right=w_right, w_left=w_left)

    def sample_reference(self, state: State) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return xref(4,N+1), uref(2,N), w_right(N+1), w_left(N+1), indices.
        """
        traj = self.trajectory
        idx0 = nearest_index(state.x, state.y, traj.x, traj.y)
        total_len = float(traj.s[-1])

        idxs = [idx0]
        s_now = float(traj.s[idx0])
        for _ in range(self.config.TK):
            v_preview = max(float(traj.v[idxs[-1]]), self.config.MIN_PREVIEW_SPEED)
            s_now += v_preview * self.config.DTK
            if s_now >= total_len:
                s_now -= total_len
            idxs.append(int(np.searchsorted(traj.s, s_now, side="left") % len(traj.s)))

        idxs_np = np.array(idxs, dtype=int)
        xref = np.zeros((4, self.config.TK + 1))
        xref[0, :] = traj.x[idxs_np]
        xref[1, :] = traj.y[idxs_np]
        xref[2, :] = traj.v[idxs_np]

        # Unwrap reference yaw around the measured yaw to avoid jumps near +/- pi.
        yaw_seq = np.unwrap(traj.yaw[idxs_np].copy())
        yaw0 = yaw_seq[0] + wrap_to_pi(state.yaw - yaw_seq[0])
        yaw_seq = yaw0 + np.unwrap(yaw_seq - yaw_seq[0])
        xref[3, :] = yaw_seq

        # Reference steering from curvature, used only as a soft input reference.
        uref = np.zeros((2, self.config.TK))
        for k in range(self.config.TK):
            ds = max(1e-6, traj.s[idxs_np[k + 1]] - traj.s[idxs_np[k]])
            if ds < 0.0:
                ds += total_len
            dyaw = wrap_to_pi(traj.yaw[idxs_np[k + 1]] - traj.yaw[idxs_np[k]])
            kappa = dyaw / max(ds, self.config.dlk)
            uref[1, k] = float(np.clip(math.atan(self.config.WB * kappa), self.config.MIN_STEER, self.config.MAX_STEER))
            uref[0, k] = (xref[2, k + 1] - xref[2, k]) / self.config.DTK

        return xref, uref, traj.w_right[idxs_np], traj.w_left[idxs_np], idxs_np

    # -------------------------------------------------------------------------
    # Tube design
    # -------------------------------------------------------------------------

    def design_tube(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A, B, _ = linearize_kinematic(
            v=self.config.TUBE_DESIGN_SPEED,
            yaw=0.0,
            delta=0.0,
            dt=self.config.DTK,
            wb=self.config.WB,
        )
        K, P = dlqr(A, B, self.config.Qk, self.config.Rk)
        Acl = A + B @ K
        eigs = np.linalg.eigvals(Acl)
        max_abs_eig = float(np.max(np.abs(eigs)))

        if max_abs_eig >= 1.0:
            self.get_logger().warn(
                f"Ancillary LQR A+BK is not Schur. max |eig|={max_abs_eig:.3f}. "
                "Tube guarantee is not valid."
            )

        z_box = robust_block_rpi_box(Acl, self.config.DISTURBANCE_BOUNDS)
        input_tightening = np.abs(K) @ z_box
        return K, P, z_box, input_tightening

    # -------------------------------------------------------------------------
    # Fast persistent CVXPY/OSQP problem
    # -------------------------------------------------------------------------

    def build_mpc_problem(self) -> None:
        """
        Build the Tube MPC QP once.

        The first v2 implementation rebuilt cp.Problem inside every callback.
        That is the usual reason for 300-700 ms solve times in Python/CVXPY.
        This version keeps the problem fixed and updates only Parameters.
        """
        N = self.config.TK

        self.x_var = cp.Variable((4, N + 1))
        self.u_var = cp.Variable((2, N))

        self.s_lat_var = cp.Variable(N + 1, nonneg=True)
        self.s_lon_var = cp.Variable(N + 1, nonneg=True)
        self.s_yaw_var = cp.Variable(N + 1, nonneg=True)
        self.s_terminal_var = cp.Variable(4, nonneg=True)

        self.x0_par = cp.Parameter(4)
        self.xref_par = cp.Parameter((4, N + 1))
        self.uref_par = cp.Parameter((2, N))
        self.last_u_par = cp.Parameter(2)

        self.A_par = [cp.Parameter((4, 4)) for _ in range(N)]
        self.B_par = [cp.Parameter((4, 2)) for _ in range(N)]
        self.C_par = [cp.Parameter(4) for _ in range(N)]

        self.cy_par = cp.Parameter(N + 1)
        self.sy_par = cp.Parameter(N + 1)
        self.left_tight_par = cp.Parameter(N + 1)
        self.right_tight_par = cp.Parameter(N + 1)
        self.long_tight_par = cp.Parameter(N + 1)
        self.yaw_tight_par = cp.Parameter(N + 1)

        constraints = []
        objective = 0.0

        x = self.x_var
        u = self.u_var

        constraints.append(x[:, 0] == self.x0_par)

        accel_tight = self.config.MAX_ACCEL - self.input_tightening[0]
        steer_upper_tight = self.config.MAX_STEER - self.input_tightening[1]
        steer_lower_tight = self.config.MIN_STEER + self.input_tightening[1]
        steer_correction = float(self.input_tightening[1])
        dsteer_tight = self.config.MAX_DSTEER * self.config.DTK - 2.0 * steer_correction

        self.accel_tight = float(accel_tight)
        self.steer_upper_tight = float(steer_upper_tight)
        self.steer_lower_tight = float(steer_lower_tight)
        self.dsteer_tight = float(dsteer_tight)

        if self.accel_tight <= 0.0 or self.steer_upper_tight <= self.steer_lower_tight:
            self.get_logger().error("Tube is too large. Input tightened set is empty.")
        if self.dsteer_tight <= 0.0:
            self.get_logger().error("Tube is too large. Steering-rate tightened set is empty.")

        for k in range(N):
            constraints.append(x[:, k + 1] == self.A_par[k] @ x[:, k] + self.B_par[k] @ u[:, k] + self.C_par[k])

            e = x[:, k] - self.xref_par[:, k]
            objective += cp.quad_form(e, self.config.Qk)
            objective += cp.quad_form(u[:, k] - self.uref_par[:, k], self.config.Rk)

            if k == 0:
                objective += cp.quad_form(u[:, k] - self.last_u_par, self.config.Rdk)
                constraints.append(cp.abs(u[1, k] - self.last_u_par[1]) <= self.dsteer_tight)
            else:
                objective += cp.quad_form(u[:, k] - u[:, k - 1], self.config.Rdk)
                constraints.append(cp.abs(u[1, k] - u[1, k - 1]) <= self.dsteer_tight)

            constraints += [
                u[0, k] <= self.accel_tight,
                u[0, k] >= -self.accel_tight,
                u[1, k] <= self.steer_upper_tight,
                u[1, k] >= self.steer_lower_tight,
                x[2, k] <= self.config.MAX_SPEED - self.z_box[2],
                x[2, k] >= self.config.MIN_SPEED,
            ]

            ex = x[0, k] - self.xref_par[0, k]
            ey = x[1, k] - self.xref_par[1, k]
            e_lat = -self.sy_par[k] * ex + self.cy_par[k] * ey
            e_lon = self.cy_par[k] * ex + self.sy_par[k] * ey
            e_yaw = x[3, k] - self.xref_par[3, k]

            if self.config.SOFTEN_CORRIDOR_AND_YAW:
                constraints += [
                    e_lat <= self.left_tight_par[k] + self.s_lat_var[k],
                    -e_lat <= self.right_tight_par[k] + self.s_lat_var[k],
                    e_lon <= self.long_tight_par[k] + self.s_lon_var[k],
                    -e_lon <= self.long_tight_par[k] + self.s_lon_var[k],
                    e_yaw <= self.yaw_tight_par[k] + self.s_yaw_var[k],
                    -e_yaw <= self.yaw_tight_par[k] + self.s_yaw_var[k],
                ]
                objective += self.config.CORRIDOR_SLACK_WEIGHT * cp.sum_squares(self.s_lat_var[k])
                objective += 0.25 * self.config.CORRIDOR_SLACK_WEIGHT * cp.sum_squares(self.s_lon_var[k])
                objective += self.config.YAW_SLACK_WEIGHT * cp.sum_squares(self.s_yaw_var[k])
            else:
                constraints += [
                    e_lat <= self.left_tight_par[k],
                    -e_lat <= self.right_tight_par[k],
                    e_lon <= self.long_tight_par[k],
                    -e_lon <= self.long_tight_par[k],
                    e_yaw <= self.yaw_tight_par[k],
                    -e_yaw <= self.yaw_tight_par[k],
                ]

        eN = x[:, N] - self.xref_par[:, N]
        objective += cp.quad_form(eN, self.config.Qfk)
        constraints += [
            x[2, N] <= self.config.MAX_SPEED - self.z_box[2],
            x[2, N] >= self.config.MIN_SPEED,
        ]

        if self.config.USE_TERMINAL_CONSTRAINT:
            if self.config.SOFTEN_TERMINAL:
                constraints.append(cp.abs(eN) <= self.terminal_box_nom + self.s_terminal_var)
                objective += self.config.TERMINAL_SLACK_WEIGHT * cp.sum_squares(self.s_terminal_var)
            else:
                constraints.append(cp.abs(eN) <= self.terminal_box_nom)

        self.mpc_problem = cp.Problem(cp.Minimize(objective), constraints)
        self.get_logger().info(
            f"Fast persistent MPC problem built: variables={self.mpc_problem.size_metrics.num_scalar_variables}, "
            f"constraints={self.mpc_problem.size_metrics.num_scalar_eq_constr + self.mpc_problem.size_metrics.num_scalar_leq_constr}"
        )

    # -------------------------------------------------------------------------
    # ROS callbacks
    # -------------------------------------------------------------------------

    def obs_1_callback(self, msg: Odometry):
        self.obs_1_x = msg.pose.pose.position.x
        self.obs_1_y = msg.pose.pose.position.y

    def obs_2_callback(self, msg: Odometry):
        self.obs_2_x = msg.pose.pose.position.x
        self.obs_2_y = msg.pose.pose.position.y

    def obs_3_callback(self, msg: Odometry):
        self.obs_3_x = msg.pose.pose.position.x
        self.obs_3_y = msg.pose.pose.position.y

    def pose_callback(self, pose_msg):
        state = self.extract_vehicle_state(pose_msg)
        self.latest_state = state
        self.control_step(state)

    def extract_vehicle_state(self, pose_msg) -> State:
        if self.is_real:
            px = pose_msg.pose.position.x
            py = pose_msg.pose.position.y
            q = pose_msg.pose.orientation
            stamp = pose_msg.header.stamp
            twist_v = None
        else:
            px = pose_msg.pose.pose.position.x
            py = pose_msg.pose.pose.position.y
            q = pose_msg.pose.pose.orientation
            stamp = pose_msg.header.stamp
            twist_v = pose_msg.twist.twist.linear.x

        yaw = yaw_from_quat(q)

        if (not self.is_real) and twist_v is not None:
            v = float(np.clip(twist_v, self.config.MIN_SPEED, self.config.MAX_SPEED))
        else:
            curr_time = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
            if self.prev_pose_time is None:
                v = float(self.drive_msg.drive.speed)
            else:
                dt = curr_time - self.prev_pose_time
                if dt <= 1e-4:
                    v = float(self.drive_msg.drive.speed)
                else:
                    dx = px - self.prev_pose_x
                    dy = py - self.prev_pose_y
                    raw_v = math.sqrt(dx * dx + dy * dy) / dt
                    raw_v = float(np.clip(raw_v, self.config.MIN_SPEED, self.config.MAX_SPEED))
                    alpha = 0.30
                    if self.v_filt is None:
                        self.v_filt = raw_v
                    else:
                        self.v_filt = alpha * raw_v + (1.0 - alpha) * self.v_filt
                    v = self.v_filt

            self.prev_pose_time = curr_time
            self.prev_pose_x = px
            self.prev_pose_y = py

        return State(x=px, y=py, v=v, yaw=yaw)

    # -------------------------------------------------------------------------
    # MPC solve
    # -------------------------------------------------------------------------

    def control_step(self, state: State):
        xref, uref, w_right, w_left, _ = self.sample_reference(state)
        self.visualize_ref_traj_in_rviz(xref)

        x_meas = np.array([state.x, state.y, state.v, state.yaw], dtype=float)
        # Put measured yaw on the same branch as the reference yaw.
        x_meas[3] = xref[3, 0] + wrap_to_pi(x_meas[3] - xref[3, 0])

        result = self.solve_tube_mpc(x_meas, xref, uref, w_right, w_left)

        feasible_msg = Bool()
        solve_msg = Float64()
        margin_msg = Float64()

        if result is None:
            self.mpc_fail_count += 1
            self.last_mpc_ok = False
            feasible_msg.data = False
            solve_msg.data = -1.0
            margin_msg.data = -1.0
            slack_msg = Float64()
            slack_msg.data = -1.0
            self.feasible_pub.publish(feasible_msg)
            self.solve_time_pub.publish(solve_msg)
            self.margin_pub.publish(margin_msg)
            self.slack_pub.publish(slack_msg)

            # Emergency fallback. The robust guarantee is not active when this happens.
            speed_cmd = float(np.clip(state.v + self.config.BACKUP_BRAKE * self.config.DTK,
                                      self.config.MIN_SPEED, self.config.MAX_SPEED))
            steer_cmd = self.rate_limit_steer(0.0)
            self.publish_drive(speed_cmd, steer_cmd)
            self.get_logger().warn("Tube MPC infeasible. Applying fallback brake.")
            return

        x_nom = result["x"]
        u_nom = result["u"]
        solve_ms = result["solve_ms"]
        min_margin = result["min_margin"]
        max_slack = result.get("max_slack", 0.0)

        z0 = x_nom[:, 0]
        eta0 = x_meas - z0
        eta0[3] = wrap_to_pi(eta0[3])
        eta0 = np.clip(eta0, -self.z_box, self.z_box)

        # Tube MPC law: actual input = nominal input + ancillary correction.
        u_actual = u_nom[:, 0] + self.K_lqr @ eta0

        # Optional local obstacle safety filter.
        if self.config.ENABLE_OBSTACLE_FILTER:
            u_actual = self.apply_obstacle_filter(x_meas, u_actual)

        a_cmd = float(np.clip(u_actual[0], -self.config.MAX_ACCEL, self.config.MAX_ACCEL))
        steer_cmd = float(np.clip(u_actual[1], self.config.MIN_STEER, self.config.MAX_STEER))
        steer_cmd = self.rate_limit_steer(steer_cmd)

        speed_cmd = float(np.clip(state.v + a_cmd * self.config.DTK,
                                  self.config.MIN_SPEED, self.config.MAX_SPEED))

        if self.enable_drive:
            self.publish_drive(speed_cmd, steer_cmd)

        self.last_actual_steer = steer_cmd
        self.last_nominal_u = u_nom[:, 0].copy()
        self.last_mpc_ok = True
        self.mpc_fail_count = 0

        self.visualize_pred_path_in_rviz(x_nom)

        feasible_msg.data = True
        solve_msg.data = float(solve_ms)
        margin_msg.data = float(min_margin)
        slack_msg = Float64()
        slack_msg.data = float(max_slack)
        self.feasible_pub.publish(feasible_msg)
        self.solve_time_pub.publish(solve_msg)
        self.margin_pub.publish(margin_msg)
        self.slack_pub.publish(slack_msg)

        self.get_logger().info(
            f"steering={steer_cmd:.3f}, speed={speed_cmd:.3f}, solve={solve_ms:.1f} ms, "
            f"margin={min_margin:.3f}, slack={max_slack:.3f}",
            throttle_duration_sec=0.5,
        )

    def solve_tube_mpc(
        self,
        x_meas: np.ndarray,
        xref: np.ndarray,
        uref: np.ndarray,
        w_right: np.ndarray,
        w_left: np.ndarray,
    ) -> Optional[dict]:
        """
        Fast online solve. No new CVXPY variables/problem are created here.
        Only Parameter values are updated.
        """
        N = self.config.TK

        if self.accel_tight <= 0.0 or self.steer_upper_tight <= self.steer_lower_tight or self.dsteer_tight <= 0.0:
            return None

        self.x0_par.value = x_meas
        self.xref_par.value = xref
        self.uref_par.value = uref
        self.last_u_par.value = self.last_nominal_u

        cy = np.cos(xref[3, :])
        sy = np.sin(xref[3, :])
        left_tight = np.zeros(N + 1)
        right_tight = np.zeros(N + 1)
        long_tight = np.zeros(N + 1)
        yaw_tight = np.zeros(N + 1)

        min_margin = 1.0e9

        for k in range(N):
            delta_ref = float(np.clip(uref[1, k], self.config.MIN_STEER, self.config.MAX_STEER))
            A, B, C = linearize_kinematic(
                v=float(xref[2, k]),
                yaw=float(xref[3, k]),
                delta=delta_ref,
                dt=self.config.DTK,
                wb=self.config.WB,
            )
            self.A_par[k].value = A
            self.B_par[k].value = B
            self.C_par[k].value = C

        for k in range(N + 1):
            lat_tube = abs(sy[k]) * self.z_box[0] + abs(cy[k]) * self.z_box[1]
            lon_tube = abs(cy[k]) * self.z_box[0] + abs(sy[k]) * self.z_box[1]

            left_tight[k] = float(w_left[k]) - self.config.CORRIDOR_MARGIN - lat_tube
            right_tight[k] = float(w_right[k]) - self.config.CORRIDOR_MARGIN - lat_tube
            long_tight[k] = self.config.LONGITUDINAL_ERROR_MAX - lon_tube
            yaw_tight[k] = self.config.YAW_ERROR_MAX - self.z_box[3]

            min_margin = min(min_margin, left_tight[k], right_tight[k], long_tight[k], yaw_tight[k])

        # With slacks enabled, the QP can still solve even if the car starts
        # outside the tightened corridor. Clamp small/negative margins to avoid
        # numerical pathologies, but publish min_margin so you know the guarantee
        # is not active when it is negative.
        if self.config.SOFTEN_CORRIDOR_AND_YAW:
            left_tight = np.maximum(left_tight, 0.02)
            right_tight = np.maximum(right_tight, 0.02)
            long_tight = np.maximum(long_tight, 0.05)
            yaw_tight = np.maximum(yaw_tight, 0.05)
        elif min_margin <= 0.0:
            self.get_logger().warn("Tightened corridor is empty. Reduce tube/disturbance or margin.")
            return None

        self.cy_par.value = cy
        self.sy_par.value = sy
        self.left_tight_par.value = left_tight
        self.right_tight_par.value = right_tight
        self.long_tight_par.value = long_tight
        self.yaw_tight_par.value = yaw_tight

        t0 = time.perf_counter()
        try:
            self.mpc_problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=self.config.OSQP_MAX_ITER,
                eps_abs=self.config.OSQP_EPS_ABS,
                eps_rel=self.config.OSQP_EPS_REL,
                polish=self.config.OSQP_POLISH,
                adaptive_rho=True,
            )
        except cp.error.SolverError as exc:
            self.get_logger().warn(f"OSQP solver error: {exc}")
            return None

        solve_ms = 1.0e3 * (time.perf_counter() - t0)
        if self.mpc_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.get_logger().warn(f"MPC failed. status={self.mpc_problem.status}")
            return None

        if self.x_var.value is None or self.u_var.value is None:
            return None

        slack_values = []
        for sv in (
            self.s_lat_var.value,
            self.s_lon_var.value,
            self.s_yaw_var.value,
            self.s_terminal_var.value,
        ):
            if sv is not None:
                slack_values.append(float(np.max(np.asarray(sv))))
        max_slack = max(slack_values) if slack_values else 0.0

        return {
            "x": np.asarray(self.x_var.value, dtype=float),
            "u": np.asarray(self.u_var.value, dtype=float),
            "solve_ms": solve_ms,
            "min_margin": float(min_margin),
            "max_slack": float(max_slack),
        }

    # -------------------------------------------------------------------------
    # Optional local obstacle safety filter
    # -------------------------------------------------------------------------

    def get_valid_obstacles(self) -> List[Tuple[float, float]]:
        obstacles = [
            (self.obs_1_x, self.obs_1_y),
            (self.obs_2_x, self.obs_2_y),
            (self.obs_3_x, self.obs_3_y),
        ]
        valid = []
        for ox, oy in obstacles:
            if abs(ox) < self.config.FAR_OBS_THRESHOLD and abs(oy) < self.config.FAR_OBS_THRESHOLD:
                valid.append((ox, oy))
        return valid

    def apply_obstacle_filter(self, x_now: np.ndarray, u_des: np.ndarray) -> np.ndarray:
        """
        Small post-MPC QP safety filter.

        This is not the main robust-MPC proof. It is only a local guard for nearby
        circular obstacles. Because Euler position dynamics do not depend directly
        on steering in one step, this filter uses a two-step lookahead barrier.
        """
        obstacles = self.get_valid_obstacles()
        if len(obstacles) == 0:
            return u_des

        car_radius = 0.5 * math.hypot(self.config.LENGTH, self.config.WIDTH)
        safe_radius = car_radius + self.config.OBS_RADIUS + self.config.SAFETY_MARGIN

        u = cp.Variable(2)
        constraints = [
            u[0] <= self.config.MAX_ACCEL,
            u[0] >= -self.config.MAX_ACCEL,
            u[1] <= self.config.MAX_STEER,
            u[1] >= self.config.MIN_STEER,
            u[1] <= self.last_actual_steer + self.config.MAX_DSTEER * self.config.DTK,
            u[1] >= self.last_actual_steer - self.config.MAX_DSTEER * self.config.DTK,
        ]
        objective = cp.sum_squares(u - u_des)

        def h_value(xs: np.ndarray, ox: float, oy: float) -> float:
            return (xs[0] - ox) ** 2 + (xs[1] - oy) ** 2 - safe_radius**2

        def two_step_h(u_eval: np.ndarray, ox: float, oy: float) -> float:
            x1 = kinematic_step(x_now, u_eval, self.config.DTK, self.config.WB, self.config.MAX_STEER)
            x1[2] = float(np.clip(x1[2], self.config.MIN_SPEED, self.config.MAX_SPEED))
            x2 = kinematic_step(x1, u_eval, self.config.DTK, self.config.WB, self.config.MAX_STEER)
            return h_value(x2, ox, oy)

        active = False
        for ox, oy in obstacles:
            dist = math.hypot(x_now[0] - ox, x_now[1] - oy)
            if dist > self.config.CBF_ACTIVATION_DISTANCE:
                continue

            h0 = h_value(x_now, ox, oy)
            h2_des = two_step_h(u_des, ox, oy)

            eps = 1e-4
            grad = np.zeros(2)
            for j in range(2):
                du = np.zeros(2)
                du[j] = eps
                grad[j] = (two_step_h(u_des + du, ox, oy) - two_step_h(u_des - du, ox, oy)) / (2.0 * eps)

            rhs = (1.0 - self.config.CBF_GAMMA) ** 2 * h0
            constraints.append(h2_des + grad[0] * (u[0] - u_des[0]) + grad[1] * (u[1] - u_des[1]) >= rhs)
            active = True

        if not active:
            return u_des

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False, max_iter=1000)
        except cp.error.SolverError:
            return u_des

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
            return u_des

        return np.asarray(u.value, dtype=float).flatten()

    # -------------------------------------------------------------------------
    # Command and visualization
    # -------------------------------------------------------------------------

    def rate_limit_steer(self, steer: float) -> float:
        max_step = self.config.MAX_DSTEER * self.config.DTK
        return float(np.clip(steer, self.last_actual_steer - max_step, self.last_actual_steer + max_step))

    def publish_drive(self, speed: float, steer: float):
        self.drive_msg.header.stamp = self.get_clock().now().to_msg()
        self.drive_msg.drive.speed = float(np.clip(speed, self.config.MIN_SPEED, self.config.MAX_SPEED))
        self.drive_msg.drive.steering_angle = float(np.clip(steer, self.config.MIN_STEER, self.config.MAX_STEER))
        self.drive_pub.publish(self.drive_msg)

    def make_marker(self, marker_type: int, marker_id: int, scale: float, rgba: Tuple[float, float, float, float]) -> Marker:
        msg = Marker()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.type = marker_type
        msg.action = Marker.ADD
        msg.id = marker_id
        msg.scale.x = scale
        msg.scale.y = scale
        msg.scale.z = scale
        msg.color.r = rgba[0]
        msg.color.g = rgba[1]
        msg.color.b = rgba[2]
        msg.color.a = rgba[3]
        return msg

    def visualize_waypoints_in_rviz(self):
        msg = self.make_marker(Marker.POINTS, 0, 0.05, (0.0, 0.75, 0.0, 1.0))
        for xi, yi in zip(self.trajectory.x, self.trajectory.y):
            msg.points.append(Point(x=float(xi), y=float(yi), z=0.1))
        self.vis_waypoints_pub.publish(msg)

    def visualize_ref_traj_in_rviz(self, xref: np.ndarray):
        msg = self.make_marker(Marker.LINE_STRIP, 0, 0.08, (0.0, 0.0, 0.75, 1.0))
        for k in range(xref.shape[1]):
            msg.points.append(Point(x=float(xref[0, k]), y=float(xref[1, k]), z=0.2))
        self.vis_ref_traj_pub.publish(msg)

    def visualize_pred_path_in_rviz(self, xpred: np.ndarray):
        msg = self.make_marker(Marker.LINE_STRIP, 0, 0.08, (0.75, 0.0, 0.0, 1.0))
        for k in range(xpred.shape[1]):
            msg.points.append(Point(x=float(xpred[0, k]), y=float(xpred[1, k]), z=0.2))
        self.vis_pred_path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TubeRobustMPC()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()