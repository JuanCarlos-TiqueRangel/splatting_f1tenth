#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SmallImageViewerNode(Node):
    def __init__(self):
        super().__init__('small_image_viewer_node')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('display_width', 320)
        self.declare_parameter('display_height', 240)
        self.declare_parameter('max_depth_m', 4.0)

        color_topic = self.get_parameter('color_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.display_width = self.get_parameter('display_width').value
        self.display_height = self.get_parameter('display_height').value
        self.max_depth_m = float(self.get_parameter('max_depth_m').value)

        self.bridge = CvBridge()
        self.color_image = None
        self.depth_image = None

        self.create_subscription(Image, color_topic, self.color_callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)

        self.timer = self.create_timer(0.03, self.show_images)

        self.window_name = 'D435 Small Viewer'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width * 2, self.display_height)

        self.get_logger().info('Small image viewer node started')

    def color_callback(self, msg: Image):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Color conversion error: {e}')

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion error: {e}')

    def show_images(self):
        if self.color_image is None or self.depth_image is None:
            return

        color_small = cv2.resize(
            self.color_image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA
        )

        depth = self.depth_image.copy()

        # Depth is expected in uint16 millimeters
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)

        max_depth_m = max(self.max_depth_m, 0.1)
        depth_m = depth.astype(np.float32) / 1000.0

        depth_norm = np.clip(depth_m / max_depth_m, 0.0, 1.0)
        depth_vis = (depth_norm * 255.0).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        depth_small = cv2.resize(
            depth_vis,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA
        )

        combined = np.hstack((color_small, depth_small))

        cv2.putText(
            combined,
            'RGB',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            combined,
            'DEPTH',
            (self.display_width + 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(self.window_name, combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info('Closing viewer...')
            cv2.destroyAllWindows()
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SmallImageViewerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()