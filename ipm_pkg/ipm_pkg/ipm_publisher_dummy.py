import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class IpmPublisherDummy(Node):
    """
    This IPM fast publisher publishes odom information from the IPM. You can also use this node to publish the IPM image.
    It only works on the front facing camera.
    """
    def __init__(self):
        super().__init__('ipm_publisher_dummy')
        timer_period = .05  # seconds
        self.odometry_publisher = self.create_publisher(Odometry, 'dynamic_obs_odom', 10)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        opp_car_odom = Odometry()
        opp_car_odom.pose.pose.position.x = 2.5
        opp_car_odom.pose.pose.position.y = 0.0
        opp_car_odom.twist.twist.linear.x = 2.0
        opp_car_odom.twist.twist.linear.y = 0.0
        self.odometry_publisher.publish(opp_car_odom)
        print(f"Published x: {opp_car_odom.pose.pose.position.x}, y: {opp_car_odom.pose.pose.position.y}, v:{opp_car_odom.twist.twist.linear.x}")


def main(args=None):
    rclpy.init(args=args)
    ipm_publisher_node = IpmPublisherDummy()
    print("IPM Dummy Publisher Initiated")
    rclpy.spin(ipm_publisher_node)
    ipm_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
