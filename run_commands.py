#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
import time
import os

class CommandExecutor:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('command_executor', anonymous=True)
        
        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Rate object for controlling loop speed
        self.rate = rospy.Rate(10)  # 10Hz
        
        # Velocity message
        self.cmd_vel = Twist()
        
        # Conversion factors (adjust these based on your robot's characteristics)
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s
        
    def execute_commands(self, filename):
        """Read commands from file and execute them sequentially"""
        if not os.path.exists(filename):
            rospy.logerr(f"Command file {filename} not found!")
            return
            
        with open(filename, 'r') as f:
            commands = f.readlines()
            
        for command in commands:
            if rospy.is_shutdown():
                break
                
            command = command.strip()
            if not command or command.startswith('#'):
                continue  # skip empty lines and comments
                
            rospy.loginfo(f"Executing: {command}")
            self._execute_single_command(command)
            
    def _execute_single_command(self, command):
        """Parse and execute a single command"""
        parts = command.split()
        if len(parts) < 2:
            rospy.logwarn(f"Invalid command format: {command}")
            return
            
        action = parts[0].lower()
        try:
            value = float(parts[1])
        except ValueError:
            rospy.logwarn(f"Invalid value in command: {command}")
            return
            
        # Stop any previous motion
        self._stop()
        
        if action == "forward":
            self._move_forward(value)
        elif action == "backward":
            self._move_backward(value)
        elif action == "clockwise":
            self._rotate_clockwise(value)
        elif action == "counterclockwise":
            self._rotate_counterclockwise(value)
        else:
            rospy.logwarn(f"Unknown command: {action}")
            
    def _move_forward(self, distance_cm):
        """Move forward specified distance in cm"""
        distance_m = distance_cm / 100.0
        duration = distance_m / self.linear_speed
        
        self.cmd_vel.linear.x = self.linear_speed
        self.cmd_vel.angular.z = 0
        self._publish_for_duration(duration)
        
    def _move_backward(self, distance_cm):
        """Move backward specified distance in cm"""
        distance_m = distance_cm / 100.0
        duration = distance_m / self.linear_speed
        
        self.cmd_vel.linear.x = -self.linear_speed
        self.cmd_vel.angular.z = 0
        self._publish_for_duration(duration)
        
    def _rotate_clockwise(self, angle_rad):
        """Rotate clockwise specified angle in radians"""
        duration = angle_rad / self.angular_speed
        
        self.cmd_vel.linear.x = 0
        self.cmd_vel.angular.z = -self.angular_speed
        self._publish_for_duration(duration)
        
    def _rotate_counterclockwise(self, angle_rad):
        """Rotate counter-clockwise specified angle in radians"""
        duration = angle_rad / self.angular_speed
        
        self.cmd_vel.linear.x = 0
        self.cmd_vel.angular.z = self.angular_speed
        self._publish_for_duration(duration)
        
    def _publish_for_duration(self, duration):
        """Publish velocity command for specified duration"""
        start_time = rospy.Time.now().to_sec()
        
        while (rospy.Time.now().to_sec() - start_time) < duration:
            if rospy.is_shutdown():
                break
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.rate.sleep()
            
        self._stop()
        
    def _stop(self):
        """Stop the robot"""
        self.cmd_vel.linear.x = 0
        self.cmd_vel.angular.z = 0
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.rate.sleep()

if __name__ == '__main__':
    try:
        executor = CommandExecutor()
        
        # Get command file path from parameter server or use default
        command_file = rospy.get_param('~command_file', 'commands.txt')
        
        rospy.loginfo(f"Starting command execution from file: {command_file}")
        executor.execute_commands(command_file)
        rospy.loginfo("Command execution completed")
        
    except rospy.ROSInterruptException:
        pass