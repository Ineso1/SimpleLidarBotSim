import numpy as np
from .lidar_bot import LidarBot

class MyBot:
    def __init__(self, map_data, initial_pose=(2.0, 3.0, 0.0)):
        self.robot = LidarBot(map_data, pose=initial_pose)
        self.exploration_strategy = None
        self.trajectory_generator = None
        self.dt = 0.1

    def set_goal(self, position, angle=np.pi / 2, velocity=(0.0, 0.0, 0.0)):
        self.robot.set_target(position, velocity, angle)

    def step(self):
        """One simulation step: scan, control, update"""
        self.robot.perform_scan()
        lidar_msg = self.get_lidar_msg()
        self.robot.apply_control(self.dt)
        self.robot.update(self.dt)
        return lidar_msg

    def run(self, steps=200):
        """Runs the robot for a number of time steps"""
        for _ in range(steps):
            self.step()

    def get_pose(self):
        return self.robot.pose

    def get_lidar_msg(self):
        """Returns the current LiDAR scan as a dictionary (ROS-like message structure)"""
        scan = self.robot.scan
        return {
            "angle_min": scan.angle_min,
            "angle_max": scan.angle_max,
            "angle_increment": scan.angle_increment,
            "range_min": scan.range_min,
            "range_max": scan.range_max,
            "ranges": scan.ranges.copy(),
            "intensities": scan.intensities.copy(),
            "frame_id": scan.header.get("frame_id", "laser_frame"),
            "stamp": scan.header.get("stamp", 0),
        }

    def animate(self):
        self.robot.data.animate_bot_on_map()
