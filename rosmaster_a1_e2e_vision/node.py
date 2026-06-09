from __future__ import annotations

from typing import Optional

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from .control_adapter import create_control_adapter
from .inference import InferenceBackend, create_inference_backend
from .model import PreprocessConfig
from .utils import SteeringFilter, clamp


class E2EVisualDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("e2e_visual_driver")

        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("start_stop_topic", "/e2e/start_stop")
        self.declare_parameter("safety_stop_topic", "/e2e/safety_stop")
        self.declare_parameter("control_topic", "/ackermann_cmd")
        self.declare_parameter("control_adapter", "ackermann_drive_stamped")

        self.declare_parameter("model_backend", "torch")
        self.declare_parameter("model_path", "")
        self.declare_parameter("model_device", "cpu")

        self.declare_parameter("image_width", 200)
        self.declare_parameter("image_height", 66)
        self.declare_parameter("crop_top_pixels", 0)

        self.declare_parameter("fixed_speed_mps", 0.12)
        self.declare_parameter("visual_stop_threshold", 0.7)
        self.declare_parameter("max_steering_abs", 0.45)
        self.declare_parameter("steering_low_pass_alpha", 0.35)
        self.declare_parameter("steering_rate_limit_per_sec", 0.75)

        self._bridge = CvBridge()
        self._running_enabled = False
        self._safety_stop = False
        self._backend: Optional[InferenceBackend] = None
        self._last_time_sec: Optional[float] = None

        self._fixed_speed = float(self.get_parameter("fixed_speed_mps").value)
        self._visual_stop_threshold = float(self.get_parameter("visual_stop_threshold").value)

        max_steering_abs = float(self.get_parameter("max_steering_abs").value)
        low_pass_alpha = float(self.get_parameter("steering_low_pass_alpha").value)
        rate_limit = float(self.get_parameter("steering_rate_limit_per_sec").value)
        self._steering_filter = SteeringFilter(max_abs=max_steering_abs, low_pass_alpha=low_pass_alpha, max_rate_per_sec=rate_limit)

        control_topic = str(self.get_parameter("control_topic").value)
        control_adapter_name = str(self.get_parameter("control_adapter").value)
        self._control_adapter = create_control_adapter(self, control_adapter_name, control_topic)

        preprocess = PreprocessConfig(
            image_width=int(self.get_parameter("image_width").value),
            image_height=int(self.get_parameter("image_height").value),
            crop_top_pixels=int(self.get_parameter("crop_top_pixels").value),
        )

        model_path = str(self.get_parameter("model_path").value)
        if model_path:
            backend_name = str(self.get_parameter("model_backend").value)
            model_device = str(self.get_parameter("model_device").value)
            self._backend = create_inference_backend(backend_name, model_path, preprocess, model_device)
            self.get_logger().info(f"Loaded model backend={backend_name} path={model_path}")
        else:
            self.get_logger().warning("No model_path configured. Node will publish safe stop only.")

        rgb_topic = str(self.get_parameter("rgb_topic").value)
        start_stop_topic = str(self.get_parameter("start_stop_topic").value)
        safety_stop_topic = str(self.get_parameter("safety_stop_topic").value)

        self.create_subscription(Image, rgb_topic, self._on_image, 10)
        self.create_subscription(Bool, start_stop_topic, self._on_start_stop, 10)
        self.create_subscription(Bool, safety_stop_topic, self._on_safety_stop, 10)

        self.get_logger().info("e2e_visual_driver ready")

    def _on_start_stop(self, msg: Bool) -> None:
        self._running_enabled = bool(msg.data)
        if not self._running_enabled:
            self._steering_filter.reset()
            self._control_adapter.publish(0.0, 0.0)

    def _on_safety_stop(self, msg: Bool) -> None:
        self._safety_stop = bool(msg.data)
        if self._safety_stop:
            self._control_adapter.publish(0.0, 0.0)

    def _on_image(self, msg: Image) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        dt = 0.0 if self._last_time_sec is None else max(0.0, now_sec - self._last_time_sec)
        self._last_time_sec = now_sec

        if not self._running_enabled or self._safety_stop or self._backend is None:
            self._control_adapter.publish(0.0, 0.0)
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        rgb = frame[:, :, ::-1]

        steering_raw, stop_prob = self._backend.predict(rgb)
        max_abs = self._steering_filter.max_abs
        steering_raw = clamp(steering_raw, -max_abs, max_abs)
        steering = self._steering_filter.apply(steering_raw, dt)

        speed = self._fixed_speed
        if stop_prob >= self._visual_stop_threshold:
            speed = 0.0

        self._control_adapter.publish(speed, steering)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = E2EVisualDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
