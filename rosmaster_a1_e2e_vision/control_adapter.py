from __future__ import annotations

from abc import ABC, abstractmethod

from geometry_msgs.msg import Twist


class ControlPublisherAdapter(ABC):
    @abstractmethod
    def publish(self, speed_mps: float, steering: float) -> None:
        raise NotImplementedError


class TwistControlAdapter(ControlPublisherAdapter):
    def __init__(self, node, topic: str) -> None:
        self._publisher = node.create_publisher(Twist, topic, 10)

    def publish(self, speed_mps: float, steering: float) -> None:
        msg = Twist()
        msg.linear.x = float(speed_mps)
        msg.angular.z = float(steering)
        self._publisher.publish(msg)


class AckermannDriveStampedAdapter(ControlPublisherAdapter):
    def __init__(self, node, topic: str) -> None:
        from ackermann_msgs.msg import AckermannDriveStamped

        self._msg_type = AckermannDriveStamped
        self._publisher = node.create_publisher(self._msg_type, topic, 10)
        self._node = node

    def publish(self, speed_mps: float, steering: float) -> None:
        msg = self._msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.drive.speed = float(speed_mps)
        msg.drive.steering_angle = float(steering)
        self._publisher.publish(msg)


def create_control_adapter(node, adapter_type: str, topic: str) -> ControlPublisherAdapter:
    adapter_type = adapter_type.lower().strip()
    if adapter_type == "ackermann_drive_stamped":
        return AckermannDriveStampedAdapter(node, topic)
    if adapter_type == "twist":
        return TwistControlAdapter(node, topic)
    raise ValueError(f"Unsupported control_adapter '{adapter_type}'")
