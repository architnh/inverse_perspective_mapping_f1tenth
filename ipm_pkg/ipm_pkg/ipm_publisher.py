import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from utils import *
from constants import *
from viz_utils import *
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from ipm_video import ipm_odom_data
import numpy as np
import time


class IpmPublisher(Node):
    def __init__(self):
        super().__init__('ipm_publisher')

        # settings
        timer_period = 1.0  # seconds
        self.publish = True

        cam_jsons = CAM_JSONS
        cam_nums = CAM_NUMS

        cam_jsons_paths = []
        for cam_json in cam_jsons:
            cam_jsons_paths.append(os.path.join("calibration", "complete_calibrations", cam_json))

        self.car = Car(cam_jsons_paths, cam_nums, debug=True)


        # Create image publisher
        self.frame_num = 0
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, 'inv_perspect_img', 10)
        self.odometry_publisher = self.create_publisher(Odometry, 'dynamic_obs_odom',10)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        ## Construct IPM ##
        # Define IPM plane
        self.plane = Plane(IPM_PLANE_X_SIZE, IPM_PLANE_Y_SIZE, METER_PER_PIX)   # returned from the ipm/constants file
        self.xyz_flat = self.plane.xyz_coord(flat=True)
        self.image_center = np.array((self.plane.W / 2, self.plane.H / 2), dtype=np.int32)


        # Project world coordinates into each camera
        self.cam_pix_coords = self.car.project_points(self.xyz_flat, self.plane.H, self.plane.W)

        #Declare time_between callbacks
        self.t_current = 0
        self.t_past = 0
        self.initial_distance = 0
        self.initial_head = 0
        self.seen_tag = False
        self.plot_point = False

    def timer_callback(self):
        # Get images from the cameras
        images = self.car.get_images()

        # Place pixels into plane image
        ipm_images = self.car.interpolate_images(images, self.cam_pix_coords)

        tag_uv, tag_cam_num = self.car.find_apriltags(images, self.plane)

        if tag_uv is not None:
            # List the point distance and heading
            print("TAG FOUND")
            self.t_current = time.time()
            if self.seen_tag:
                dist, head = pixel_dist_and_heading(self.image_center, tag_uv, METER_PER_PIX)
                delta_t = self.t_current - self.t_past
                vx, vy = self.car.estimate_velocity(self.initial_distance, dist, self.initial_head, head, delta_t)
                self.initial_distance = dist
                self.initial_head = head
                self.plot_point = True
            else:
                self.seen_tag = True

            self.t_past = self.t_current
        else:
            self.seen_tag = False
            self.plot_point = False

        # Create IPM image
        # Show the images
        ipm_img = ipm_images[0] + ipm_images[1] + ipm_images[2] + ipm_images[3]
        ipm_img = self.car.draw_car_and_cameras(ipm_img)
        if self.plot_point:
            ipm_img = draw_point_uv(ipm_img, tag_uv, RED, size=5)
            cv2.putText(ipm_img, f'Distance : {round(dist, 2)} m', (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_img, f'Heading : {np.round(head, 2)} rad', (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_img, f'Velocity X : {np.round(vx, 2)} m/s', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)
            cv2.putText(ipm_img, f'Velocity Y : {np.round(vy, 2)} m/s', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, RED, 2, cv2.LINE_AA)

        if self.publish:
            msg = self.bridge.cv2_to_imgmsg(ipm_img, encoding="bgr8")
            msg.header.frame_id = str(self.frame_num)  # msg is a sensor_msgs.msg.Image type
            self.frame_num += 1
            self.image_publisher.publish(msg)
            if self.plot_point:
                opp_car_odom = Odometry()
                opp_car_odom.pose.pose.position.x = float(dist * np.cos(head))
                opp_car_odom.pose.pose.position.y = float(dist * np.sin(head))
                opp_car_odom.twist.twist.linear.x = float(vx)
                opp_car_odom.twist.twist.linear.y = float(vy)
                self.odometry_publisher.publish(opp_car_odom)


def main(args=None):
    rclpy.init(args=args)
    ipm_publisher_node = IpmPublisher()
    print("IPM Image Publisher Initiated")
    rclpy.spin(ipm_publisher_node)
    ipm_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
