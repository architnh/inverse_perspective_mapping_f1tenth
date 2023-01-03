cmake_minimum_required(VERSION 3.5)
project(point_cloud_creator)

# Default to C99
#if(NOT CMAKE_C_STANDARD)
#  set(CMAKE_C_STANDARD 99)
#endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
#find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


foreach(PACKAGE
  ament_cmake
  ament_cmake_python
  rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  ackermann_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  Eigen3
  OpenCV
  cv_bridge
  tf2_ros)
  find_package(${PACKAGE} REQUIRED)

endforeach()

include_directories(include)
#include_directories(
#  ${EIGEN3_INCLUDE_DIRS}
#)


# Create Cpp executable
add_executable(image2point src/image2point.cpp)
ament_target_dependencies(image2point
  rclcpp OpenCV geometry_msgs cv_bridge ackermann_msgs nav_msgs sensor_msgs std_msgs tf2_ros
)


# Install Cpp executables
install(TARGETS
  image2point
  DESTINATION lib/${PROJECT_NAME})



if(BUILD_TESTING)
 find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
