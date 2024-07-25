

import rclpy
import math

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformRegistration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from fuzzyLogicControllerWall import FuzzyLogicControllerWall
from fuzzyLogicControllerObstacle import FuzzyLogicControllerObstacle
from combining import Combining

mynode_ = None
pub_ = None
regions_ = {
    'right': 0,
    'fright': 0,
    'bright' : 0,
    'front1': 0,
    'front2': 0,
    'fleft': 0,
    'left': 0,
}

twstmsg_ = None

#Creating objects for fuzzy logic controlelrs and combination of codes
fuzzyContr = FuzzyLogicControllerWall()
fuzzyContrObstacle = FuzzyLogicControllerObstacle()
combining = Combining()

# main function attached to timer callback
def timer_callback():
    global pub_, twstmsg_
    if ( twstmsg_ != None ):
        pub_.publish(twstmsg_)


def clbk_laser(msg):
    global regions_, twstmsg_
    
    regions_ = {
        #LIDAR readings are anti-clockwise
        'front1':  find_nearest (msg.ranges[0:5]),
        'front2':  find_nearest (msg.ranges[355:360]),
        'right':  find_nearest(msg.ranges[265:275]),
        'bright':  find_nearest(msg.ranges[230:240]),
        'fright': find_nearest (msg.ranges[335:354]),
        'fleft':  find_nearest (msg.ranges[6:25]),
        'left':   find_nearest (msg.ranges[85:95])
    }     
    twstmsg_= movement()

    
# Find nearest point
def find_nearest(list):
    f_list = filter(lambda item: item > 0.0, list)  # exclude zeros
    return min(min(f_list, default=1), 1)




#Basic movement method
def movement():
    #print("here")
    global regions_, mynode_, fuzzyContr
    regions = regions_
    
    #print("Min distance in front region: ", regions_['front1'],regions_['front2'])
    
    #create an object of twist class, used to express the linear and angular velocity of the turtlebot 
    msg = Twist()
    
    print(regions_['fright'], regions_['fleft'], min(regions_['front1'], regions_['front2']))
    velXOutRWF, velZOutRWF = fuzzyContr.rullBase(regions_['fright'], regions_['bright'])
    velXOutOA, velZOutOA = fuzzyContrObstacle.rullBase(regions_['fright'], regions_['fleft'], min(regions_['front1'], regions_['front2']))

    velXOut, velZOut = combining.combine(min(regions_['front1'], regions_['front2']), regions_['fright'], regions_['fleft'], regions_['bright'], velXOutRWF, velXOutOA, velZOutRWF, velZOutOA)

    msg.linear.x = 1 * velXOut * 1
    msg.angular.z = 1 * velZOut * 1
    return msg

#used to stop the rosbot
def stop():
    global pub_
    msg = Twist()
    msg.angular.z = 0.0
    msg.linear.x = 0.0
    pub_.publish(msg)


def main():
    global pub_, mynode_

    rclpy.init()
    mynode_ = rclpy.create_node('reading_laser')

    # define qos profile (the subscriber default 'reliability' is not compatible with robot publisher 'best effort')
    qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )

    # publisher for twist velocity messages (default qos depth 10)
    pub_ = mynode_.create_publisher(Twist, '/cmd_vel', 10)

    # subscribe to laser topic (with our qos)
    sub = mynode_.create_subscription(LaserScan, '/scan', clbk_laser, qos)

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
        stop()
        mynode_.destroy_timer(timer)
        mynode_.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()