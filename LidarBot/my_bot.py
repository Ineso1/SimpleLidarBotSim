import numpy as np
from .lidar_bot import LidarBot

class MyBot:
    def __init__(self, initial_pose=(2.0, 3.0, 0.0), map_size=5.0, resolution=0.1):
        self.robot = LidarBot(pose=initial_pose)
        self.exploration_strategy = None
        self.trajectory_generator = None
        self.dt = 0.1
        self.map_size = map_size
        self.resolution = resolution
        
        # Calculate the size of the occupancy grid
        self.grid_size = int(self.map_size / self.resolution)
        # Create an empty occupancy map (0 = free, 1 = occupied, -1 = unknown)
        self.occupancy_map = -np.ones((self.grid_size, self.grid_size))

    def set_goal(self, position, angle=np.pi / 2, velocity=(0.0, 0.0, 0.0)):
        self.robot.set_target(position, velocity, angle)

    def step(self):
        """One simulation step: scan, control, update"""
        self.robot.perform_scan()
        lidar_msg = self.get_lidar_msg()
        self.robot.apply_control(self.dt)
        self.robot.update(self.dt)
        
        # Update the occupancy map based on the LiDAR scan
        self.update_occupancy_map()
        
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

    def update_occupancy_map(self):
        """Update the occupancy map using the current LiDAR scan."""
        scan = self.robot.scan
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment
        ranges = scan.ranges
        
        # Get the robot's current position
        x, y, theta = self.get_pose()

        # Log the current pose for debugging
        print(f"Current pose: x={x}, y={y}, theta={theta}")

        # Mark the robot's current position as free
        grid_x, grid_y = self.world_to_grid(x, y)
        print(f"Robot grid position: grid_x={grid_x}, grid_y={grid_y}")
        self.occupancy_map[grid_x, grid_y] = 0  # Free space

        # Update the map based on LiDAR readings
        for i, distance in enumerate(ranges):
            if np.isinf(distance) or np.isnan(distance):
                continue

            angle = angle_min + i * angle_increment
            lx = x + distance * np.cos(angle)
            ly = y + distance * np.sin(angle)

            grid_lx, grid_ly = self.world_to_grid(lx, ly)
            if self.is_within_map(grid_lx, grid_ly):
                self.occupancy_map[grid_lx, grid_ly] = 1  # Occupied


    def world_to_grid(self, world_x, world_y):
        """Converts world coordinates (x, y) to grid coordinates, ensuring they stay within bounds."""
        grid_x = int((world_x / self.resolution) + self.grid_size / 2)
        grid_y = int((world_y / self.resolution) + self.grid_size / 2)

        # Ensure the grid coordinates are within bounds
        grid_x = np.clip(grid_x, 0, self.grid_size - 1)
        grid_y = np.clip(grid_y, 0, self.grid_size - 1)

        return grid_x, grid_y


    def is_within_map(self, grid_x, grid_y):
        """Check if grid coordinates are within the map bounds."""
        return 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size

    def animate(self):
        self.robot.data.animate_bot_on_map()
