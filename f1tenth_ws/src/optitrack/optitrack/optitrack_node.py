#!/usr/bin/env python3
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped
from util import quaternion_to_euler
from NatNetClient import NatNetClient


@dataclass
class Sample:
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    rpy: Tuple[float, float, float]
    t: float


# ---------------- Quaternion helpers ----------------
def quat_norm(q):
    x, y, z, w = q
    return math.sqrt(x*x + y*y + z*z + w*w)

def quat_normalize(q):
    x, y, z, w = q
    n = quat_norm(q)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x/n, y/n, z/n, w/n)

def quat_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    )

def rotate_vec_by_quat(v, q):
    vx, vy, vz = v
    qv = (vx, vy, vz, 0.0)
    qn = quat_normalize(q)
    out = quat_mul(quat_mul(qn, qv), quat_conj(qn))
    return (out[0], out[1], out[2])

def angular_velocity_from_quats(q_prev, q_curr, dt):
    if dt <= 1e-9:
        return (0.0, 0.0, 0.0)

    q0 = quat_normalize(q_prev)
    q1 = quat_normalize(q_curr)

    dq = quat_mul(quat_conj(q0), q1)
    dq = quat_normalize(dq)

    x, y, z, w = dq
    if w < 0.0:
        x, y, z, w = (-x, -y, -z, -w)

    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w*w))

    if s < 1e-8 or angle < 1e-8:
        return (0.0, 0.0, 0.0)

    ax, ay, az = (x / s, y / s, z / s)
    return (ax * angle / dt, ay * angle / dt, az * angle / dt)


class OptiTrackMultiNode(Node):
    def __init__(self):
        super().__init__("optitrack_multi_node")

        # Parameters
        self.declare_parameter("client_ip", "192.168.0.37")
        self.declare_parameter("server_ip", "192.168.0.4")
        self.declare_parameter("robot_ids", [529, 527, 528])
        self.declare_parameter("use_multicast", True)

        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_hz", 100.0)
        self.declare_parameter("twist_in_body_frame", False)
        self.declare_parameter("vel_smoothing_alpha", 0.0)

        self.client_ip = self.get_parameter("client_ip").value
        self.server_ip = self.get_parameter("server_ip").value
        self.robot_ids = list(self.get_parameter("robot_ids").value)
        self.use_multicast = bool(self.get_parameter("use_multicast").value)

        self.frame_id = self.get_parameter("frame_id").value
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.twist_in_body_frame = bool(self.get_parameter("twist_in_body_frame").value)
        self.alpha = float(self.get_parameter("vel_smoothing_alpha").value)
        self.alpha = max(0.0, min(0.99, self.alpha))

        if len(self.robot_ids) < 1:
            raise RuntimeError("robot_ids must contain at least one object id")

        # First object in the list -> PoseStamped
        self.pose_object_id = self.robot_ids[0]

        # Per-object storage
        self._lock = threading.Lock()
        self._latest: Dict[int, Optional[Sample]] = {rid: None for rid in self.robot_ids}
        self._prev_used: Dict[int, Optional[Sample]] = {rid: None for rid in self.robot_ids}
        self._latest_used_t: Dict[int, Optional[float]] = {rid: None for rid in self.robot_ids}
        self._v_filt: Dict[int, Tuple[float, float, float]] = {rid: (0.0, 0.0, 0.0) for rid in self.robot_ids}
        self._w_filt: Dict[int, Tuple[float, float, float]] = {rid: (0.0, 0.0, 0.0) for rid in self.robot_ids}

        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            f"/optitrack/object_{self.pose_object_id}/pose",
            10
        )

        self.odom_pubs = {}
        self.rpy_pubs = {}
        for rid in self.robot_ids[1:]:
            self.odom_pubs[rid] = self.create_publisher(
                Odometry, f"/optitrack/object_{rid}/odom", 10
            )
            self.rpy_pubs[rid] = self.create_publisher(
                Vector3, f"/optitrack/object_{rid}/rpy", 10
            )

        # NatNet
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(self.client_ip)
        self.streaming_client.set_server_address(self.server_ip)
        self.streaming_client.set_use_multicast(self.use_multicast)
        self.streaming_client.rigid_body_listener = self._on_rigid_body

        ok = self.streaming_client.run()
        if not ok:
            raise RuntimeError("NatNetClient failed to start.")

        self.get_logger().info(
            f"Publishing PoseStamped for object {self.pose_object_id} "
            f"and Odometry for objects {self.robot_ids[1:]}"
        )

        period = 1.0 / max(1.0, self.publish_hz)
        self.timer = self.create_timer(period, self._timer_cb)

    def _on_rigid_body(self, rigid_id, position, rotation_quaternion):
        if rigid_id not in self.robot_ids:
            return

        pos = (float(position[0]), float(position[1]), float(position[2]))
        q = (
            float(rotation_quaternion[0]),
            float(rotation_quaternion[1]),
            float(rotation_quaternion[2]),
            float(rotation_quaternion[3]),
        )
        q = quat_normalize(q)

        roll, pitch, yaw = quaternion_to_euler(rotation_quaternion)
        rpy = (float(roll), float(pitch), float(yaw))

        s = Sample(pos=pos, quat=q, rpy=rpy, t=time.monotonic())

        with self._lock:
            self._latest[rigid_id] = s

    def _publish_pose_object(self, rid: int, s: Sample):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = s.pos[0]
        msg.pose.position.y = s.pos[1]
        msg.pose.position.z = s.pos[2]

        msg.pose.orientation.x = s.quat[0]
        msg.pose.orientation.y = s.quat[1]
        msg.pose.orientation.z = s.quat[2]
        msg.pose.orientation.w = s.quat[3]

        self.pose_pub.publish(msg)

    def _publish_odom_object(self, rid: int, s: Sample):
        prev = self._prev_used[rid]

        if prev is None:
            v = (0.0, 0.0, 0.0)
            w = (0.0, 0.0, 0.0)
        else:
            dt = s.t - prev.t
            if dt <= 1e-6:
                return

            v_world = (
                (s.pos[0] - prev.pos[0]) / dt,
                (s.pos[1] - prev.pos[1]) / dt,
                (s.pos[2] - prev.pos[2]) / dt,
            )
            w_world = angular_velocity_from_quats(prev.quat, s.quat, dt)

            if self.twist_in_body_frame:
                q_inv = quat_conj(s.quat)
                v = rotate_vec_by_quat(v_world, q_inv)
                w = rotate_vec_by_quat(w_world, q_inv)
            else:
                v = v_world
                w = w_world

        if self.alpha > 0.0:
            self._v_filt[rid] = tuple(
                self.alpha * self._v_filt[rid][i] + (1.0 - self.alpha) * v[i]
                for i in range(3)
            )
            self._w_filt[rid] = tuple(
                self.alpha * self._w_filt[rid][i] + (1.0 - self.alpha) * w[i]
                for i in range(3)
            )
            v_out, w_out = self._v_filt[rid], self._w_filt[rid]
        else:
            v_out, w_out = v, w

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = f"object_{rid}"

        msg.pose.pose.position.x = s.pos[0]
        msg.pose.pose.position.y = s.pos[1]
        msg.pose.pose.position.z = s.pos[2]

        msg.pose.pose.orientation = Quaternion(
            x=s.quat[0], y=s.quat[1], z=s.quat[2], w=s.quat[3]
        )

        msg.twist.twist.linear.x = v_out[0]
        msg.twist.twist.linear.y = v_out[1]
        msg.twist.twist.linear.z = v_out[2]

        msg.twist.twist.angular.x = w_out[0]
        msg.twist.twist.angular.y = w_out[1]
        msg.twist.twist.angular.z = w_out[2]

        rpy_msg = Vector3()
        rpy_msg.x = s.rpy[0]
        rpy_msg.y = s.rpy[1]
        rpy_msg.z = s.rpy[2]

        self.odom_pubs[rid].publish(msg)
        self.rpy_pubs[rid].publish(rpy_msg)

    def _timer_cb(self):
        with self._lock:
            latest_copy = dict(self._latest)

        for rid, s in latest_copy.items():
            if s is None:
                continue

            if self._latest_used_t[rid] is not None and abs(s.t - self._latest_used_t[rid]) < 1e-12:
                continue

            self._latest_used_t[rid] = s.t
            self._prev_used[rid] = s if self._prev_used[rid] is None else self._prev_used[rid]

            if rid == self.pose_object_id:
                self._publish_pose_object(rid, s)
                self._prev_used[rid] = s
            else:
                self._publish_odom_object(rid, s)
                self._prev_used[rid] = s

    def destroy_node(self):
        try:
            if hasattr(self.streaming_client, "shutdown"):
                self.streaming_client.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OptiTrackMultiNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
