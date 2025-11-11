#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
from tf_transformations import euler_from_quaternion
import time

class EbotNav(Node):
    def __init__(self):
        super().__init__('ebot_nav')
        # publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)

        # robot state
        self.x = None
        self.y = None
        self.yaw = None

        # lidar state
        self.min_front_dist = float('inf')
        self.front_clear_distance = 0.6   # meters threshold to consider "clear"
        self.obstacle_avoidance_turn_speed = 0.4

        # waypoints: list of (x,y,yaw)
        self.waypoints = [
            (-1.53, -1.95,  1.57),
            ( 0.13,  1.24,  0.00),
            ( 0.38, -3.32, -1.57),
        ]
        self.current_wp_index = 0

        # tolerances
        self.pos_tol = 0.3
        self.yaw_tol = math.radians(10.0)

        # control gains & limits
        self.kp_lin = 0.6
        self.max_lin = 0.6
        self.kp_ang = 1.2
        self.max_ang = 1.2

        # loop timer
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('ebot_nav node started')

    def odom_cb(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = yaw

    def scan_cb(self, msg: LaserScan):
        # compute the minimum distance in a front sector
        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        n = len(ranges)
        # define front window +/- 20 degrees
        window = math.radians(20)
        i_min = max(0, int(( -window - angle_min) / angle_inc))
        i_max = min(n - 1, int((  window - angle_min) / angle_inc))
        # filter out "inf" and 0.0
        min_f = float('inf')
        for i in range(i_min, i_max + 1):
            r = ranges[i]
            if r == 0.0 or math.isinf(r) or math.isnan(r):
                continue
            if r < min_f:
                min_f = r
        self.min_front_dist = min_f

    def angle_diff(self, a, b):
        # return smallest signed difference a - b in [-pi, pi]
        d = a - b
        while d > math.pi:
            d -= 2.0 * math.pi
        while d < -math.pi:
            d += 2.0 * math.pi
        return d

    def control_loop(self):
        if self.x is None or self.yaw is None:
            return  # waiting for odom

        if self.current_wp_index >= len(self.waypoints):
            # finished all waypoints: stop
            self.stop_robot()
            self.get_logger().info('All waypoints reached. Stopping.')
            # keep node alive but no more commands
            return

        goal = self.waypoints[self.current_wp_index]
        gx, gy, gyaw = goal

        # compute errors
        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        yaw_error = self.angle_diff(angle_to_goal, self.yaw)

        # obstacle check
        obstacle_in_front = (self.min_front_dist < self.front_clear_distance)

        twist = Twist()

        # If obstacle detected in front: reactive rotate until clear
        if obstacle_in_front and dist > 0.2:
            # rotate away (pick direction based on sign of yaw_error)
            # simple: rotate in place with fixed speed until front is clear
            twist.linear.x = 0.0
            twist.angular.z = self.obstacle_avoidance_turn_speed
            self.cmd_pub.publish(twist)
            self.get_logger().info(f'Obstacle detected at {self.min_front_dist:.2f} m — rotating to avoid')
            return

        # If far from waypoint: first align or go forward with heading correction
        if dist > self.pos_tol:
            # if facing wrong direction (large angular error), rotate in place first
            if abs(yaw_error) > math.radians(20):
                # rotate to face goal
                twist.linear.x = 0.0
                ang = self.kp_ang * yaw_error
                ang = max(-self.max_ang, min(self.max_ang, ang))
                twist.angular.z = ang
            else:
                # move forward with steering
                lin = self.kp_lin * dist
                lin = max(-self.max_lin, min(self.max_lin, lin))
                ang = self.kp_ang * yaw_error
                ang = max(-self.max_ang, min(self.max_ang, ang))
                twist.linear.x = lin
                twist.angular.z = ang
            self.cmd_pub.publish(twist)
            return
        else:
            # close enough in position — now rotate to desired final yaw
            yaw_final_error = self.angle_diff(gyaw, self.yaw)
            if abs(yaw_final_error) > self.yaw_tol:
                # rotate to goal orientation
                twist.linear.x = 0.0
                ang = self.kp_ang * yaw_final_error
                ang = max(-self.max_ang, min(self.max_ang, ang))
                twist.angular.z = ang
                self.cmd_pub.publish(twist)
                return
            else:
                # waypoint reached: stop, advance
                self.get_logger().info(f'Waypoint {self.current_wp_index+1} reached: pos_err={dist:.3f}, yaw_err={yaw_final_error:.3f}')
                self.stop_robot()
                # small pause to stabilize
                time.sleep(0.5)
                self.current_wp_index += 1
                return

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        # publish multiple times to ensure stop
        for _ in range(3):
            self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = EbotNav()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
