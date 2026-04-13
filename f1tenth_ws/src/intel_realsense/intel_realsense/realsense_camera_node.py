#!/usr/bin/env python3

import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RealSenseCameraNode(Node):
    def __init__(self):
        super().__init__('realsense_camera_node')

        # Parameters
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30)
        self.declare_parameter('serial_no', '')
        self.declare_parameter('align_depth_to_color', True)

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')

        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.serial_no = self.get_parameter('serial_no').value
        self.align_depth_to_color = self.get_parameter('align_depth_to_color').value

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value

        self.bridge = CvBridge()

        self.color_pub = self.create_publisher(Image, color_topic, 10)
        self.depth_pub = self.create_publisher(Image, depth_topic, 10)

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial_no:
            self.config.enable_device(self.serial_no)

        # D435 streams
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        # Align depth to color if requested
        self.align = rs.align(rs.stream.color) if self.align_depth_to_color else None

        self.timer = self.create_timer(1.0 / float(self.fps), self.timer_callback)

        self.get_logger().info(
            f'RealSense camera node started | width={self.width}, height={self.height}, fps={self.fps}'
        )

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000)

            if self.align is not None:
                frames = self.align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                self.get_logger().warning('Missing color or depth frame')
                return

            color_image = np.asanyarray(color_frame.get_data())   # uint8, HxWx3
            depth_image = np.asanyarray(depth_frame.get_data())   # uint16, HxW

            stamp = self.get_clock().now().to_msg()

            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
            color_msg.header.stamp = stamp
            color_msg.header.frame_id = 'camera_color_optical_frame'

            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = 'camera_depth_optical_frame'

            self.color_pub.publish(color_msg)
            self.depth_pub.publish(depth_msg)

        except Exception as e:
            self.get_logger().error(f'Error reading/publishing RealSense frames: {e}')

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseCameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()