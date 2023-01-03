#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA
import math
import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetection
from sensor_msgs.msg import Image
import pyrealsense2 as rs

from time import time
import cv2
import time
# import torch
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from std_msgs.msg import Float32
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import PoseStamped
# from geometry_msgs.msg import PointStamped
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from geometry_msgs.msg import Point
# from nav_msgs.msg import Odometry
# from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
# from nav_msgs.msg import OccupancyGrid
# from visualization_msgs.msg import Marker
# from visualization_msgs.msg import MarkerArray

class PerceptionNode(Node):
    def __init__(self):
        depth_old = 0
        apriltag_topic = 'detections'
        # pose_topic = "ego_racecar/odom"
        # scan_topic = "/scan"
        # # Create ROS subscribers and publishers.
        # self.subscription = self.create_subscription(
        #     LaserScan,
        #     scan_topic,
        #     self.scan_callback,
        #     10)

        # self.subscription = self.create_subscription(
        #     Odometry,
        #     pose_topic,
        #     self.odom_callback,
        #     10)
        # self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 10)
        self.publisher_ = self.create_publisher(Float32, 'depth', 10)
        self.subscription = self.create_subscription(
            AprilTagDetection,
            apriltag_topic,
            self.tag_callback,
            10)
        self.subscription = self.create_subscription(
            Image,
            '/depth/image_rect_raw',
            self.depth_callback,
            10)


    def depth_callback(self, depth_msg):
        self.dpt = depth_msg.data
        print(type(self.dpt))
        print(np.shape(self.dpt))


    def scan_callback(self, scan_msg):
        # self.speed = odom_msg.twist.twist.linear.x
        pass

    def tag_callback(self, scan_msg):
        pipeline = rs.pipeline()
        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 960, 540, rs.format.z16, 60)

        # Start streaming
        pipeline.start(config)
        pixel_x = scan_msg.pose.pose.pose.point.x
        pixel_y = scan_msg.pose.pose.pose.point.y
        pixel_z = scan_msg.pose.pose.pose.point.z #i guess we dont need it?
        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/python-tutorial-1-depth.py
        dpt_frame = pipeline.wait_for_frames().get_depth_frame().as_depth_frame() 
        #pixel_distance_in_meters = dpt_frame.get_distance(pixel_x,pixel_y)
        count = 0
        # average out the value of depth pixel by creating a convolution of width 5 around the pixel
        for i in range(pixel_x-5, pixel_x+5):
            for j in range(pixel_y-5, pixel_y+5):
                depth_avg += dpt_frame.get_distance(i,j)
                count +=1 
        depth_avg = depth_avg/count
        msg = Float32()
        msg.data = float(depth_avg)
        self.publisher_.publish(msg)
        vel = calculate_velocity(depth_old, depth_avg)
        print(vel)
        depth_old=depth_avg


        
    def calculate_velocity(depth1, depth2):
        #For calculations, visit -
        #https://dev.intelrealsense.com/docs/high-speed-capture-mode-of-intel-realsense-depth-camera-d435
        focal_length=1
        frame_rate=60 #Hz
        velocity = (depth1 - depth2)*frame_rate*depth2/focal_length
        return velocity


    def getimage(number):
        # cap = cv2.VideoCapture("v4l2src device=/dev/video2 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink")
        # time_old = time.time()
        # if cap.isOpened():
        #     cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
        #     while True:
        #         time_now = time.time()
        #         ret_val, img = cap.read()
        #         print(1/(time_now - time_old), 'Hz')
        #         time_old = time_now
        #         cv2.imshow('demo', img)
        #         cv2.waitKey(1)
        # else:
        #     print("Camera open failed")
        # cap = cv2.VideoCapture("v4l2src device=/dev/video0 extra-controls=\"c,exposure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw,format=BGR ! appsink")
        # cap = cv2.VideoCapture(1) #works IR
        cap = cv2.VideoCapture(number) #works visual range number=2
        time_old = time.time()
        if cap.isOpened():
            cv2.namedWindow("demo", cv2.WINDOW_AUTOSIZE)
            time_new = time.time()
            ret_val, img = cap.read()
            print("Camera open")
        else:
            print("Camera open failed")
        cv2.destroyAllWindows()
        image = img 
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    percept_node = PerceptionNode()
    rclpy.spin(percept_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    percept_node.destroy_node()
    rclpy.shutdown()
