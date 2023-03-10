// RRT assignment

// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <vector>
#include <random>
#include <Eigen/Geometry>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/bool.hpp"
//#include "std_msgs/msg/float32multiarray.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
/// CHECK: include needed ROS msg type headers and libraries

using namespace std;

// Struct defining the RRT_Node object in the RRT tree.
// More fields could be added to thiis struct if more info needed.
// You can choose to use this or not
typedef struct RRT_Node {
    double x, y;
    double cost; // only used for RRT*
    int parent; // index of parent node in the tree vector
    bool is_root = false;
} RRT_Node;

class RRT : public rclcpp::Node {
public:
    RRT();
    virtual ~RRT();

    Eigen::Quaterniond q;
    Eigen::Matrix3d rotation_mat;

    float update_rate = 0.0001;//0.04;//Time between updating rrt graph. Time to execute RRT* is around 0.0001 min to 0.002 sec. Recommended to keep above 0.03

    const static int occu_grid_x_size=100;//135;//always make this an even number
    const static int occu_grid_y_size=100;//125;//always make this an even

    //DANGER BE CAREFUL CHANGING THIS NUMBER SEE NOTE BELOW
    const float  resolution=0.04;//0.04;//
    //WARNING IF YOU CHANGE RESOLUTION, ALSO CHANGE THE DIVIDE BY NUMBER IN THE TWO VARIABLES BELOW
    const static int x_size=  occu_grid_x_size/0.04;//0.04;
    const static int y_size= occu_grid_y_size/0.04;//0.04;

    const static int center_y = occu_grid_y_size/2;
    const static int center_x = occu_grid_x_size * 0.2; //occu_grid_x_size/2;

    nav_msgs::msg::Odometry current_car_pose;
    const int x_size_nonstatic= occu_grid_x_size/ resolution;

    int occu_grid[x_size][y_size]= {0};
    //int occu_grid_flat[x_size * y_size]= {0};

    //RRT Stuff
    float max_expansion_dist = 0.5; //meters
    int max_iter = 300;//500;
    float goal_threshold = 0.1; //meters
    float l_value=3;
    nav_msgs::msg::Odometry global_goal;

    rclcpp::Time previous_time = rclcpp::Clock().now();

    bool found_path=false;
    bool message_sent=false;
    std::vector<RRT_Node> final_path_output;
    nav_msgs::msg::Odometry current_goal;
    bool rrt_use_it=false;

    std::vector<float> opponent_position={0,0,0};
    std::vector<float> opponent_velocity={0,0,0};
    float opponent_heading = 0.0;
    bool seen_opponent=false;
    rclcpp::Time time_last_opponent_msg = rclcpp::Clock().now();
    float opponent_topic_timeout = .5;

    std::vector<std::vector<float>> spline_points;
private:
    // TODO: add the publishers and subscribers you need
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr rrt_rviz;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr rrt_path_rviz;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr goal_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr node_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_pub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr global_goal_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr pure_pursuit_standoff_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr dynamic_obs_sub;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_path_pub;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr use_rrt_pub;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr spline_points_pub;
    // random generator, use this
    std::mt19937 gen;
    std::uniform_real_distribution<> x_dist;
    std::uniform_real_distribution<> y_dist;

    // callbacks
    // where rrt actually happens
    void pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg);
    // updates occupancy grid
    void scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg);

    void global_goal_callback(const nav_msgs::msg::Odometry::ConstSharedPtr goal_msg);

    void standoff_callback(const std_msgs::msg::Float32::ConstSharedPtr l_dist);

    void opponent_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom);

    std::vector<signed char> dynamic_obstacle(std::vector<float> obstacle_position, std::vector<float> obstacle_velocity,std::vector<signed char> flat_grid);
    bool check_if_gaining_on_opponent(std::vector<float> obstacle_position, std::vector<float> obstacle_velocity);

    std::vector<RRT_Node> perform_rrt();
    // RRT methods
    std::vector<double> sample();
    int nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point);
    RRT_Node steer(RRT_Node &nearest_node, std::vector<double> &sampled_point);
    bool check_collision(RRT_Node &nearest_node, RRT_Node &new_node);
    bool chck_coll(std::vector<double> nearest_node, std::vector<double> new_node);
    bool is_goal(RRT_Node &latest_added_node, double goal_x, double goal_y);

    //Publisher functions
    void update_rrt_path_lines(std::vector<RRT_Node> &tree, std::vector<RRT_Node> &path);
    void update_rrt_rviz(std::vector<RRT_Node> &tree);
    void update_goal_point(float goal_point_x, float goal_point_y);
    void update_nodes(std::vector<RRT_Node> &tree);
    void update_path(std::vector<RRT_Node> &path);

    std::vector<RRT_Node> find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node);
    // RRT* methods
    double cost(std::vector<RRT_Node> &tree, RRT_Node &node);
    double line_cost(RRT_Node &n1, RRT_Node &n2);
    std::vector<int> near(std::vector<RRT_Node> &tree, RRT_Node &node);
    std::vector<std::vector<int>> bresenhams_line_algorithm(int goal_point[2], int origin_point[2]);
    void check_to_activate_rrt(std::vector<signed char> &obstacle_data);
    void check_to_activate_rrt_alt(std::vector<signed char> &obstacle_data);
    int find_spline_index(float x, float y);


};


