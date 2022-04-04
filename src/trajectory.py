import math
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf.transformations import *

# Make path
def path_generator():
    T = 0.1
    path = Path()
    path.header.frame_id = "/odom"
    path.header.stamp = rospy.Time.now()
    sim_time = 30

    iter = 0
    while sim_time - iter*T >= 0:
        # Reference
        t_predict = T*iter
        # x_ref = 4*math.cos(2*math.pi/30*t_predict)
        # y_ref = 4*math.sin(2*math.pi/30*t_predict)
        # q_ref = 2*math.pi/30*t_predict + math.pi/2

        x_ref = 1/3*t_predict
        y_ref = -4*math.cos(2*math.pi/30*t_predict)
        q_ref = math.atan2(2*math.pi/30*4*math.sin(2*math.pi/30*t_predict), 1/3)

        # Convert to ROS message
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = x_ref
        pose.pose.position.y = y_ref
        quat = quaternion_from_euler(0, 0, q_ref)
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]

        path.poses.append(pose)

        iter += 1
    return path

def trajectory_publish():
    rospy.init_node("trajectory_node", anonymous=True)
    rate = 1

    # Publisher
    pub_path = rospy.Publisher('/path', Path, queue_size=1)

    path = path_generator()
    print(len(path.poses))

    r = rospy.Rate(rate)
    while not rospy.is_shutdown():
        pub_path.publish(path)
        r.sleep()

if __name__ == '__main__':    
    try:
        trajectory_publish()
    except rospy.ROSInterruptException:
        pass