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
import numpy as np
import time


class IpmPublisherFast(Node):
    """
    This IPM fast publisher publishes odom information from the IPM. You can also use this node to publish the IPM image.
    It only works on the front facing camera.
    """
    def __init__(self):
        super().__init__('ipm_publisher_fast')

        # settings
        timer_period = .01  # seconds
        self.publish_odom = True
        self.publish_image = False

        self.t_fps_past = 0.0

        cam_json = "camera_f_lowres_complete.json"
        cam_num = 2
        cam_json_path = os.path.join("calibration", "complete_calibrations", cam_json)
        self.cam = Camera(cam_json_path)
        self.cap = cv2.VideoCapture(cam_num)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cnt = 0

        # Create publishers
        self.frame_num = 0
        self.bridge = CvBridge()
        self.odometry_publisher = self.create_publisher(Odometry, 'dynamic_obs_odom', 10)
        self.image_publisher = self.create_publisher(Image, 'inv_perspect_img', 10)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        ## Construct IPM ##
        # Define IPM plane
        self.plane = Plane(IPM_PLANE_X_SIZE, IPM_PLANE_Y_SIZE, METER_PER_PIX)   # returned from the ipm/constants file
        self.xyz_flat = self.plane.xyz_coord(flat=True, front_only=False)
        self.image_center = np.array((self.plane.W / 2, self.plane.H / 2), dtype=np.int32)

        # Project world coordinates into each camera
        self.cam_pix_coords = self.cam.project_points(self.xyz_flat, self.plane.H, self.plane.W)

        # Declare time_between callbacks
        self.t_past = 0
        self.distance_past = 0
        self.head_past = 0
        self.seen_tag = False  # If a tag has been seen in the previous frame
        self.seen_tag_twice = False  # If a tag has been seen in two continous frames
        self.vx_history = []
        self.vy_history = []

    def timer_callback(self):
       
        ret, image = self.cap.read()
        
        if ret:
            if True:
                fname = str(self.cnt) + "img.png"
                cv2.imwrite(fname, image)
                self.cnt +=1
                print("Wrote image")
             # Get images from the cameras
            t_fps_current = time.time()
            delta = t_fps_current - self.t_fps_past
            print(f"Hz: {1/delta}")
            self.t_fps_past = t_fps_current
            tag_found, tag_corners = return_apriltag_location(image, return_center=False)

            if tag_found:  # If a tag has been identified
                tag_uv, valid_dist = return_bounding_midpoint(tag_corners, self.cam, self.plane)
                if valid_dist:
                    dist_current, head_current = pixel_dist_and_heading(self.image_center, tag_uv, METER_PER_PIX)
                    t_current = time.time()
                    print(f"Tag found at {dist_current} m, {head_current} rad")
                    if self.seen_tag:  # If a tag has been seen in the previous frame
                        delta_t = t_current - self.t_past
                        vx, vy = estimate_velocity(self.distance_past, dist_current, self.head_past, head_current, delta_t)
                        self.vx_history.append(vx)
                        self.vy_history.append(vy)

                        # Process the velocity
                        if self.vx_history.__len__() > 3:
                            self.vx_history.pop(0)
                        if self.vy_history.__len__() > 3:
                            self.vy_history.pop(0)
                        vx_avg = np.mean(self.vx_history)
                        vy_avg = np.mean(self.vy_history)
                        if abs(vx_avg) < 0.2:
                            vx_avg = 0
                        if abs(vy_avg) < 0.2:
                            vy_avg = 0
                        self.seen_tag_twice = True  # Tell the publisher that a tag has been seen in two frames,
                        # so publish velocity

                    else:  # First time seeing a tag
                        self.seen_tag = True

                    self.distance_past = dist_current
                    self.head_past = head_current
                    self.t_past = t_current

            if time.time() - self.t_past > .8:  # If we have not seen a while
                self.seen_tag = False
                self.seen_tag_twice = False
                self.vx_history = []
                self.vy_history = []

            if self.publish_odom:
                if self.seen_tag_twice and tag_found:
                    opp_car_odom = Odometry()
                    opp_car_odom.pose.pose.position.x = float(dist_current * np.cos(head_current))
                    opp_car_odom.pose.pose.position.y = float(dist_current * np.sin(head_current))
                    opp_car_odom.twist.twist.linear.x = float(vx_avg)
                    opp_car_odom.twist.twist.linear.y = float(vy_avg)
                    self.odometry_publisher.publish(opp_car_odom)




def main(args=None):
    rclpy.init(args=args)
    ipm_publisher_node = IpmPublisherFast()
    print("IPM Publisher Fast Initiated")
    rclpy.spin(ipm_publisher_node)
    ipm_publisher_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
