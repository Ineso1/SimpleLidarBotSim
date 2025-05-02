import numpy as np
from bresenham import bresenham

class OccupancyMap:
    def __init__(self, size_meters=5.0, resolution=0.05):
        self.size_meters = size_meters
        self.resolution = resolution  # meters per cell
        self.grid_size = int(size_meters / resolution)
        self.map = -1 * np.ones((self.grid_size, self.grid_size), dtype=np.int8)  # -1 = unknown

    def world_to_grid(self, x, y):
        i = int(y / self.resolution)
        j = int(x / self.resolution)
        return i, j

    def update_from_scan(self, bot):
        scan = bot.get_lidar_msg()
        x, y, theta = bot.get_pose()
        angle_min = scan["angle_min"]
        angle_increment = scan["angle_increment"]
        ranges = scan["ranges"]

        robot_i, robot_j = self.world_to_grid(x, y)
        self.mark_cell(robot_i, robot_j, 0)  # Robot's own cell is free

        for idx, dist in enumerate(ranges):
            if np.isinf(dist) or np.isnan(dist):
                continue
            if dist > 5.0:  # Ignore very far hits
                continue

            angle = theta + angle_min + idx * angle_increment
            hit_x = x + dist * np.cos(angle)
            hit_y = y + dist * np.sin(angle)

            hit_i, hit_j = self.world_to_grid(hit_x, hit_y)
            if 0 <= hit_i < self.grid_size and 0 <= hit_j < self.grid_size:
                self.mark_cell(hit_i, hit_j, 1)  # Hit is an obstacle
                self._mark_ray(robot_i, robot_j, hit_i, hit_j)

    def mark_cell(self, i, j, value):
        if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
            self.map[i, j] = value

    def _mark_ray(self, i0, j0, i1, j1):
        for i, j in bresenham(j0, i0, j1, i1):  # bresenham uses (x, y) order
            self.mark_cell(j, i, 0)

    def save_map(self, filename='map.npy'):
        np.save(filename, self.map)

    def load_map(self, filename='map.npy'):
        self.map = np.load(filename)
