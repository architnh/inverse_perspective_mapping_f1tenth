#define BOOST_BIND_NO_PLACEHOLDERS
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include "nav_msgs/msg/odometry.hpp"
#include <sstream>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;

nav_msgs::msg::Odometry current_car_pose;
Eigen::Quaterniond q;
float pixel_size = 0.005;

Eigen::AngleAxisd yaw(0.0, Eigen::Vector3d::UnitZ());
Eigen::AngleAxisd pitch(0.0, Eigen::Vector3d::UnitY());
Eigen::AngleAxisd roll(3.14159, Eigen::Vector3d::UnitX());
Eigen::Quaternion<double> q1 =    roll * pitch * yaw ;
Eigen::Matrix3d z_offset = q1.matrix();

cv::Mat plane_image = cv::imread("src/Final_Race_F1_tenth/point_cloud_creator/src/plane.jpg");

class converter: public rclcpp::Node
{
  public:
  converter()
  : Node("converter")
  {
    bool topic_name = false;  // Set flag true for simulation, false for real

    points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/image_point_cloud_rviz",1);
    //timer_ = this->create_wall_timer(0ms, std::bind(&converter::point_cloud_pub, this));
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "/inv_perspect_img", 10, std::bind(&converter::image_callback, this, _1));
    
    if(topic_name == true){
        std::string pose_topic = "ego_racecar/odom";
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(pose_topic, 1, 
        std::bind(&converter::pose_callback, this, std::placeholders::_1));
    }
    else{
        std::string pose_topic = "pf/pose/odom";
        pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(pose_topic, 1, 
        std::bind(&converter::pose_callback, this, std::placeholders::_1));
    }
  
  }
private:

  void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    
    current_car_pose = *pose_msg;
    
    q.x()= current_car_pose.pose.pose.orientation.x;
    q.y()= current_car_pose.pose.pose.orientation.y;
    q.z()= current_car_pose.pose.pose.orientation.z;
    q.w()= current_car_pose.pose.pose.orientation.w;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr frame)const
  {

      auto rotation_mat = q.toRotationMatrix();

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
      cv_bridge::CvImagePtr cv_ptr;
      try
      {
        cv_ptr = cv_bridge::toCvCopy(frame, frame->encoding);//local_frame, "bgr8");// sensor_msgs::image_encodings::BGR8);
      }
      catch (cv_bridge::Exception& e)
      {
        std::cout<<"cv_bridge exception"<<std::endl;
        return;
      }

      cv::Mat image = cv::Mat (cv_ptr->image).clone();

      int image_width= image.size().width;
      int image_height = image.size().height;

      float half_width = image_width / 2;
      float half_height = image_height / 2;

      pcl::PointXYZRGB point;
      point.z = 0.0;

      Eigen::Vector3d global_coords;
      Eigen::Vector3d shift_coords(0,0,0);
      cv::Vec3b color;
      float car_position[2]={current_car_pose.pose.pose.position.x, current_car_pose.pose.pose.position.y};
      for (int y=0;y<image_height;y++){
          for (int x=0;x<image_width;x++){

              color = image.at<cv::Vec3b>(cv::Point(x,y));
              int32_t rgb = (color[2] << 16) | (color[1] << 8) | color[0];

              if( (int)color[0] > 0 && (int)color[1] > 0 && (int)color[2] > 0){
                //Eigen::Vector3d shift_coords((x - half_width) * pixel_size, (y - half_height)* pixel_size, 0);
                shift_coords[0] = (x - half_width) * pixel_size;
                shift_coords[1] = (y - half_height)* pixel_size;
                global_coords = rotation_mat * z_offset * shift_coords;

                point.x = car_position[0] + global_coords[0];
                point.y = car_position[1] + global_coords[1];

                
                point.rgb = *reinterpret_cast<float*>(&rgb);
                point_cloud->points.push_back(point);
              }
          }
      }
      
    auto output = sensor_msgs::msg::PointCloud2();
    pcl::toROSMsg(*point_cloud, output);
    output.header.frame_id="map";
    points_pub_->publish(output);
  }

  void point_cloud_pub(){

      auto rotation_mat = q.toRotationMatrix();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

      int image_width= plane_image.size().width;
      int image_height = plane_image.size().height;

      float half_width = image_width / 2;
      float half_height = image_height / 2;

      pcl::PointXYZRGB point;
      point.z = 0.0;

      Eigen::Vector3d global_coords;
      Eigen::Vector3d shift_coords(0,0,0);
      cv::Vec3b color;
      float car_position[2]={current_car_pose.pose.pose.position.x, current_car_pose.pose.pose.position.y};
      for (int y=0;y<image_height;y++){
          for (int x=0;x<image_width;x++){

              color = plane_image.at<cv::Vec3b>(cv::Point(x,y));
              //std::cout<<"color="<<(int)color[0]<<","<<(int)color[1]<<","<<(int)color[2]<<std::endl;
              
              if( (int)color[0] > 0 && (int)color[1] > 0 && (int)color[2] > 0){

                shift_coords[0] = (x - half_width) * pixel_size;
                shift_coords[1] = (y - half_height)* pixel_size;
                global_coords = rotation_mat * z_offset * shift_coords;

                point.x = car_position[0] + global_coords[0];
                point.y = car_position[1] + global_coords[1];

                int32_t rgb = (color[2] << 16) | (color[1] << 8) | color[0];
                point.rgb = *reinterpret_cast<float*>(&rgb);
                point_cloud->points.push_back(point);
              }
          }
      }
    auto output = sensor_msgs::msg::PointCloud2();
    pcl::toROSMsg(*point_cloud, output);
    output.header.frame_id="map";
    points_pub_->publish(output);
  }
  //rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr points_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<converter>());
  rclcpp::shutdown();
  return 0;
}
