import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformRegistration

mynode_ = None
pub_ = None
regions_ = {
    "right": 0,
    "fright": 0,
    "bright": 0,
    "front1": 0,
    "front2": 0,
    "fleft": 0,
    "left": 0,
}
errorAccum_ = 0  # Variable used to calculate integral part of PID controller
errorPrev_ = 0  # Variable used to calculate differential part of PID controller

twstmsg_ = None


# main function attached to timer callback
def timer_callback():
    global pub_, twstmsg_
    if twstmsg_ != None:
        pub_.publish(twstmsg_)


def clbk_laser(msg):
    global regions_, twstmsg_

    regions_ = {
        # LIDAR readings are anti-clockwise
        "front1": find_nearest(msg.ranges[0:5]),
        "front2": find_nearest(msg.ranges[355:360]),
        "right": find_nearest(msg.ranges[265:275]),
        "bright": find_nearest(msg.ranges[230:240]),
        "fright": find_nearest(msg.ranges[310:320]),
        "fleft": find_nearest(msg.ranges[40:50]),
        "left": find_nearest(msg.ranges[85:95]),
    }
    twstmsg_ = movement()


# Find nearest point
def find_nearest(list):
    f_list = filter(lambda item: item > 0.0, list)  # exclude zeros
    return min(min(f_list, default=10), 10)


# PID controller function, arguments: error, gain, integral constant, differential constant
def controller(error, kp, ki, kd) -> float:
    antiWindup = 0.2  # Anti windup variable constraints integral part
    global errorAccum_, errorPrev_

    if abs(errorAccum_ + error) < antiWindup:  # Constraint to integral error
        errorAccum_ = error + errorAccum_
    errorDerivative = error - errorPrev_
    errorPrev_ = error
    outputVal = kp * error + ki * errorAccum_ + kd * (errorDerivative)
    return outputVal


# Basic movement method
def movement():
    # print("here")
    global regions_, mynode_
    regions = regions_

    # print("Min distance in front region: ", regions_['front1'],regions_['front2'])

    # create an object of twist class, used to express the linear and angular velocity of the turtlebot
    msg = Twist()

    # Controller
    desiredDistance = 0.3  # desired distance from the wall

    if regions_["fright"] >= (
        regions_["bright"] * 8 / 10
    ):  # Modyfication that helps avoiding obstacles and improves performance
        error = desiredDistance - (2 ** (1 / 2)) * regions_["fright"] / 2
    else:
        error = desiredDistance + (2 ** (1 / 2)) * regions_["bright"] / 2

    print(error)

    # set of PID variables
    kp = 0.3
    ki = 0.05
    kd = 0.01
    angular = controller(error, kp, ki, kd)

    # If an obstacle is found to be within desiredDistance/2 stops robot and turn to find right wall
    if (regions_["front1"]) < desiredDistance / 2:
        msg.linear.x = 0.0
        msg.angular.z = 0.2
        return msg
    elif (regions_["front2"]) < desiredDistance / 2:
        msg.linear.x = 0.0
        msg.angular.z = 0.2
        return msg
    elif (regions_["fright"]) < desiredDistance / 2:
        msg.linear.x = 0.0
        msg.angular.z = 0.2
        return msg
    else:
        msg.linear.x = 0.1
        msg.angular.z = angular
        return msg


# used to stop the rosbot
def stop():
    global pub_
    msg = Twist()
    msg.angular.z = 0.0
    msg.linear.x = 0.0
    pub_.publish(msg)


def main():
    global pub_, mynode_

    rclpy.init()
    mynode_ = rclpy.create_node("reading_laser")

    # define qos profile (the subscriber default 'reliability' is not compatible with robot publisher 'best effort')
    qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )

    # publisher for twist velocity messages (default qos depth 10)
    pub_ = mynode_.create_publisher(Twist, "/cmd_vel", 10)

    # subscribe to laser topic (with our qos)
    sub = mynode_.create_subscription(LaserScan, "/scan", clbk_laser, qos)

    # Configure timer
    timer_period = 0.2  # seconds
    timer = mynode_.create_timer(timer_period, timer_callback)

    # Run and handle keyboard interrupt (ctrl-c)
    try:
        rclpy.spin(mynode_)
    except KeyboardInterrupt:
        stop()  # stop the robot
    except:
        stop()  # stop the robot
    finally:
        # Clean up
        mynode_.destroy_timer(timer)
        mynode_.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
