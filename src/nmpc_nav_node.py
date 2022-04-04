#!/usr/bin/env python3
import numpy as np
import casadi as ca
import math
import time

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import *

import sys
sys.path.append("/home/duynam/vast_ws/src/mpc_casadi/src/")
try:
    from nmpc_controller import NMPCController
except:
    raise

# Odometry callback
odom = Odometry()
def odom_callback(data):
    global odom
    odom = data

# Reference path callback
path = Path()
def path_callback(data):
    global path
    path = data

def quaternion2Yaw(orientation):
    q0 = orientation.x
    q1 = orientation.y
    q2 = orientation.z
    q3 = orientation.w

    yaw = math.atan2(2.0*(q2*q3 + q0*q1), 1.0 - 2.0*(q1*q1 + q2*q2))
    return yaw

# def desired_trajectory(pos, t, T, N_):
#     # initial state / last state
#     x_ = np.zeros((N_+1, 3))
#     x_[0] = pos
#     u_ = np.zeros((N_, 3))

#     for i in range(N_):
#         t_predict = t + T*i
#         x_ref_ = 4*math.cos(2*math.pi/30*t_predict)
#         y_ref_ = 4*math.sin(2*math.pi/30*t_predict)
#         theta_ref_ = 2*math.pi/30*t_predict + math.pi/2
        
#         dotx_ref_ = -2*math.pi/30*y_ref_
#         doty_ref_ =  2*math.pi/30*x_ref_
#         dotq_ref_ =  2*math.pi/30

#         vx_ref_ = dotx_ref_*math.cos(dotq_ref_) + doty_ref_*math.sin(dotq_ref_)
#         vy_ref_ = -dotx_ref_*math.sin(dotq_ref_) + doty_ref_*math.cos(dotq_ref_)
#         omega_ref_ = dotq_ref_

#         x_[i+1] = np.array([x_ref_, y_ref_, theta_ref_])
#         u_[i] = np.array([vx_ref_, vy_ref_, omega_ref_])

#     return x_, u_

def desired_trajectory(pos, N:int, path:Path):
    # initial state
    x_ = np.zeros((N+1, 3))


    # Process the path
    traj = []
    for i in range(len(path.poses)):
        x = path.poses[i].pose.position.x
        y = path.poses[i].pose.position.y
        q = quaternion2Yaw(path.poses[i].pose.orientation)
        traj.append([x,y,q])
    traj = np.array(traj)
    x_ref = traj[:,0]
    y_ref = traj[:,1]
    q_ref = traj[:,2]

    x_ref_ = x_ref[:N]
    y_ref_ = y_ref[:N]
    q_ref_ = q_ref[:N]
    length = len(x_ref_)

    if length < N:
        x_ex = np.ones(N - length)*x_ref_[-1]
        x_ref_ = np.concatenate((x_ref_, x_ex), axis=None)

        y_ex = np.ones(N - length)*y_ref_[-1]
        y_ref_ = np.concatenate((y_ref_, y_ex), axis=None)

        q_ex = np.ones(N - length)*q_ref_[-1]
        q_ref_ = np.concatenate((q_ref_, q_ex), axis=None)

    dx_ref_ = np.diff(x_ref_)
    dx_ref_ = np.concatenate((dx_ref_[0], dx_ref_), axis=None)
    dy_ref_ = np.diff(y_ref_)
    dy_ref_ = np.concatenate((dy_ref_[0], dy_ref_), axis=None)
    dq_ref_ = np.diff(q_ref_)
    dq_ref_ = np.concatenate((dq_ref_[0], dq_ref_), axis=None)

    vx_ref_ = dx_ref_*np.cos(dq_ref_) + dy_ref_*np.sin(dq_ref_)
    vy_ref_ = -dx_ref_*np.sin(dq_ref_) + dy_ref_*np.cos(dq_ref_)
    omega_ref_ = dq_ref_

    x_ = np.array([x_ref_, y_ref_, q_ref_]).T
    x_ = np.concatenate((np.array([pos]), x_), axis=0)
    u_ = np.array([vx_ref_, vy_ref_, omega_ref_]).T
    return x_, u_


def correct_state(states, tracjectories):
    error = tracjectories - states
    error[:,2] = error[:,2] - np.floor((error[:,2] + np.pi)/(2*np.pi))*2*np.pi
    tracjectories = states + error
    return tracjectories

def nmpc_node():
    rospy.init_node("nmpc_node", anonymous=True)
    rate = 10

    # Subscriber
    rospy.Subscriber("/odom", Odometry, odom_callback)

    path_topic = rospy.get_param('~path_topic')     # private parameter
    rospy.Subscriber(path_topic, Path, path_callback)
    
    # Publisher
    pub_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=rate)
    pub_pre_path = rospy.Publisher('/predict_path', Path, queue_size=rate)
    
    r = rospy.Rate(rate)

    print("[INFO] Init Node...")
    while(odom.header.frame_id == "" or path.header.frame_id == ""):
        r.sleep()
        continue
    print("[INFO] NMPC Node is ready!!!")

    T = 1/rate
    N = 20

    min_vx = rospy.get_param('/RobotConstraints/min_vx')
    max_vx = rospy.get_param('/RobotConstraints/max_vx')
    min_vy = rospy.get_param('/RobotConstraints/min_vy')
    max_vy = rospy.get_param('/RobotConstraints/max_vy')
    min_omega = rospy.get_param('/RobotConstraints/min_omega')
    max_omega = rospy.get_param('/RobotConstraints/max_omega')

    # Create the current robot position
    px = odom.pose.pose.position.x
    py = odom.pose.pose.position.y
    pq = quaternion2Yaw(odom.pose.pose.orientation)
    pos = np.array([px, py, pq])

    nmpc = NMPCController(pos, min_vx, max_vx, min_vy, max_vy, min_omega, max_omega, T, N) 
    t0 = 0
    while not rospy.is_shutdown():
        # Current position
        px = odom.pose.pose.position.x
        py = odom.pose.pose.position.y
        pq = quaternion2Yaw(odom.pose.pose.orientation)
        pos = np.array([px, py, pq])

        next_traj, next_cons = desired_trajectory(pos, N, path)
        next_traj = correct_state(nmpc.next_states, next_traj)
        vel = nmpc.solve(next_traj, next_cons)

        # Publish cmd_vel
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = vel[1]
        vel_msg.angular.z = vel[2]
        pub_vel.publish(vel_msg)


        # Publish the predict path
        predict = nmpc.next_states
        predict_msg = Path()
        predict_msg.header.frame_id = "odom"
        predict_msg.header.stamp = rospy.Time.now()
        for pos in predict:
            pose = PoseStamped()
            pose.header = predict_msg.header
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            quat = quaternion_from_euler(0, 0, pos[2])
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            predict_msg.poses.append(pose)
        pub_pre_path.publish(predict_msg)

        r.sleep()
        t0 += T

if __name__ == '__main__':    
    try:
        nmpc_node()
    except rospy.ROSInterruptException:
        pass