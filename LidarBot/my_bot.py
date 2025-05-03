import numpy as np
from .lidar_bot import LidarBot
from Exploration.occupancy_map import OccupancyMap 


class MyBot:
    def __init__(self, initial_pose=(2.0, 3.0, 0.0), map_size=5.0, resolution=0.1):
        self.robot = LidarBot(pose=initial_pose)
        self.exploration_strategy = None
        self.trajectory_generator = None
        self.dt = 0.1
        self.map_size = map_size
        self.resolution = resolution
        self.pointAchived = False
        self.occupancy_map = OccupancyMap(initial_size=100, resolution=resolution)


    def set_goal(self, position, angle=np.pi / 2, velocity=(0.0, 0.0, 0.0)):
        self.robot.set_target(position, velocity, angle)
        self.robot.pid_x.done = False
        self.robot.pid_y.done = False
        self.robot.pid_theta.done = False


    def step(self):
        self.robot.perform_scan()
        lidar_msg = self.get_lidar_msg()
        self.update_map_from_lidar(lidar_msg)
        self.robot.apply_control(self.dt)
        self.robot.update(self.dt)        
        return lidar_msg

    def run(self, steps=200):
        for _ in range(steps):
            self.step()

    def loop(self):
        while not (self.robot.pid_x.done and self.robot.pid_y.done and self.robot.pid_theta.done):
            self.step()

    def get_pose(self):
        return self.robot.pose

    def get_lidar_msg(self):
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
    
    def update_map_from_lidar(self, scan_msg):
        x, y, theta = self.robot.pose
        angle = scan_msg["angle_min"]
        for r in scan_msg["ranges"]:
            if scan_msg["range_min"] <= r <= scan_msg["range_max"]:
                angle_world = theta + angle
                hit_x = x + r * np.cos(angle_world)
                hit_y = y + r * np.sin(angle_world)

                self.occupancy_map.set_occupancy(hit_x, hit_y, 0.0)

                num_steps = int(r / self.occupancy_map.resolution)
                for i in range(num_steps):
                    intermediate_r = i * self.occupancy_map.resolution
                    fx = x + intermediate_r * np.cos(angle_world)
                    fy = y + intermediate_r * np.sin(angle_world)
                    self.occupancy_map.set_occupancy(fx, fy, 1.0)
            angle += scan_msg["angle_increment"]
        self.occupancy_map.history.append(self.occupancy_map.occupancy_grid.copy())
        print(len(self.occupancy_map.history))

    def animate(self):
        self.robot.data.animate_bot_on_map()
        self.occupancy_map.plot()

    def animate2(self):
        self.robot.data.animate_bot_with_mapping(self.occupancy_map)
