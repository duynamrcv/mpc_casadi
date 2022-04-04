import math
import numpy as np
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist, PoseStamped

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

def process_reference(path:Path):
    traj = []
    for i in range(len(path.poses)):
        x = path.poses[i].pose.position.x
        y = path.poses[i].pose.position.y
        q = quaternion2Yaw(path.poses[i].pose.orientation)
        traj.append([x,y,q])
    traj = np.array(traj)
    return traj

def process_odometry(odom:Odometry):
    x = odom.pose.pose.position.x
    y = odom.pose.pose.position.y
    q = quaternion2Yaw(odom.pose.pose.orientation)

    vx = odom.twist.twist.linear.x
    vy = odom.twist.twist.linear.y
    w = odom.twist.twist.angular.z

    t = odom.header.stamp.to_sec()
    return [x, y, q], [vx, vy, w, t]

poses = []  # store the robot position
traj = []   # store the desired trajectory
vels = []   # store the robot velocities

def plotting():
    print("[INFO] Plot data!!!")
    global traj
    global poses
    global vels
    traj = np.array(traj)
    poses = np.array(poses)
    vels = np.array(vels)

    # Plot the tracked data
    plt.figure()
    plt.plot(traj[:,0], traj[:,1], "-b", linewidth=2, label="Reference")
    plt.plot(poses[:,0], poses[:,1], "--r", linewidth=2, label="NMPC")
    plt.legend()
    plt.grid(True)
    plt.title("Tracked trajectory")
    plt.axis('equal')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # import scipy.io
    # scipy.io.savemat('/home/duynam/vast_ws/src/mpc_casadi/data/track_n40.mat', dict(ans=poses))
    # scipy.io.savemat('/home/duynam/vast_ws/src/mpc_casadi/data/track_ref.mat', dict(ans=traj))

    # # Plot the velocity
    # plt.figure()
    # plt.suptitle("Velocity")
    # plt.subplot(311)
    # plt.plot(vels[:,0], vels[:,1], "-b", linewidth=2)
    # plt.xlabel("ROS time")
    # plt.ylabel("value [m/s]")
    # plt.title("$v_x$")

    # plt.subplot(312)
    # plt.plot(vels[:,0], vels[:,2], "-b", linewidth=2)
    # plt.xlabel("ROS time")
    # plt.ylabel("value [m/s]")
    # plt.title("$v_y$")

    # plt.subplot(313)
    # plt.plot(vels[:,0], vels[:,3], "-b", linewidth=2)
    # plt.xlabel("ROS time")
    # plt.ylabel("value [rad/s]")
    # plt.title("$\omega$")

    plt.show()  

def plot_tracking():
    rospy.init_node("nmpc_node", anonymous=True)
    rate = 10

    # Subscriber
    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/path", Path, path_callback)

    r = rospy.Rate(rate)

    print("[INFO] Init Node Plotting...")
    while(odom.header.frame_id == "" or path.header.frame_id == ""):
        r.sleep()
        continue
    print("[INFO] Ready to plotting!!!")

    global traj
    traj = process_reference(path)

    while not rospy.is_shutdown():
        pos, vel = process_odometry(odom)
        poses.append(pos)
        vels.append(vel)

        r.sleep()
        rospy.on_shutdown(plotting)

if __name__ == '__main__':    
    try:
        plot_tracking()
    except rospy.ROSInterruptException:
        pass