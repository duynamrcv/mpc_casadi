cmake_minimum_required(VERSION 3.0.2)
project(mpc_casadi)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  src/trajectory.py
  src/nmpc_node.py
  src/nmpc_nav_node.py

  test/test_nmpc.py

  plotting/plot_tracking.py
  plotting/plot_navigation.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)