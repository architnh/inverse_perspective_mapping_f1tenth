import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class IpmPublisherSim(Node):
    """
    This IPM fast publisher publishes odom information from the IPM. You can also use this node to publish the IPM image.
    It only works on the front facing camera.
    """
    def __init__(self):
        super().__init__('ipm_publisher_sim')
        timer_period = .1  # seconds
        self.odometry_publisher = self.create_publisher(Odometry, 'dynamic_obs_odom', 10)
        self.ego_pose_subscriber = self.create_subscription(Odometry, 'ego_racecar/odom', self.ego_pose_callback, 1)
        self.oppo_pose_subscriber = self.create_subscription(Odometry, 'opp_racecar/odom', self.oppo_pose_callback, 1)

        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.time_last_vel = time.time()

        self.ego_position = None
        self.ego_rotation = None
        self.opp_position = None
        self.opp_position_old = None
        self.opp_rotation = None
        self.opp_vx = 0.0
        self.opp_vy = 0.0
        self.opp_called = False
        self.ego_called = False
        self.local_old = None
        self.dist_old = 0
        self.heading_old = 0


    def ego_pose_callback(self, msg):
        self.ego_position = msg.pose.pose.position
        self.ego_rotation = msg.pose.pose.orientation
        self.ego_called = True

    def oppo_pose_callback(self, msg):
        if not self.opp_called:
            self.opp_position_old = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        else:
            self.opp_position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
            self.opp_rotation = msg.pose.pose.orientation
        self.opp_called = True

    def global_2_local(self, current_quat, current_position, goal_point_global):
        # Construct transformation matrix from rotation matrix and position
        H_global2car = np.zeros([4, 4]) #rigid body transformation from  the global frame of referce to the car
        H_global2car[3, 3] = 1
        current_rotation_matrix = R.from_quat(np.array([current_quat.x,current_quat.y,current_quat.z,current_quat.w])).as_matrix()
        H_global2car[0:3, 0:3] = np.array(current_rotation_matrix)
        H_global2car[0:3, 3] = np.array([current_position.x, current_position.y, current_position.z])

        # Calculate point
        goal_point_global = np.append(goal_point_global, 1).reshape(4, 1)
        goal_point_car = np.linalg.inv(H_global2car) @ goal_point_global

        return goal_point_car

    def timer_callback(self):
        current_time = time.time()
        if self.ego_called and self.opp_called:
            local_new = self.global_2_local(self.ego_rotation, self.ego_position, self.opp_position)
            dist_new = np.linalg.norm(local_new)
            heading_new = np.arctan2(local_new[1], local_new[0])

            self.opp_vx, self.opp_vy = estimate_velocity(self.dist_old, dist_new, self.heading_old, heading_new,
                                                         current_time - self.time_last_vel)

            self.dist_old = dist_new
            self.heading_old = heading_new
            self.time_last_vel = current_time
            self.opp_position_old = self.opp_position

            opp_car_odom = Odometry()
            opp_car_local = self.global_2_local(self.ego_rotation, self.ego_position, self.opp_position)
            opp_car_odom.pose.pose.position.x = float(opp_car_local[0])
            opp_car_odom.pose.pose.position.y = float(opp_car_local[1])
            opp_car_odom.twist.twist.linear.x = float(self.opp_vx)
            opp_car_odom.twist.twist.linear.y = float(self.opp_vy)
            self.odometry_publisher.publish(opp_car_odom)
            print(f"Published x: {opp_car_odom.pose.pose.position.x}, y: {opp_car_odom.pose.pose.position.y}, v:{opp_car_odom.twist.twist.linear.x}")

def estimate_velocity(distance1, distance2, heading1, heading2, delta_t):
    """
    Args:
    :param distance1: initial distance estimate of opponent car
    :param distance2: latest distance estimate of opponent car
    :param heading: direction estimate of where the opponent car is headed
    :param delta_t: Time between two readings
    :return: velocity estimate
    :return: positive indicates opponent is faster, moving away
    :return: negative indicates, opponent is slower, ego_car is gaining on opponent
    """
    velocity_x = (distance2*np.cos(heading2) - distance1*np.cos(heading1))/delta_t
    velocity_y = (distance2*np.sin(heading2) - distance1*np.sin(heading1))/delta_t

    return velocity_x, velocity_y

def main(args=None):
    rclpy.init(args=args)
    ipm_publisher_node = IpmPublisherSim()
    print("IPM Dummy Publisher Initiated")
    rclpy.spin(ipm_publisher_node)
    ipm_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
