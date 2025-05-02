import numpy as np
from PIL import Image

class MyMap:
    def __init__(self, filename="map1.png", grid_resolution=0.01, desired_size=5, show_grid=False):
        self.grid_resolution = grid_resolution
        self.img = Image.open(filename).convert('L')
        self.occupancy_grid = np.array(self.img) < 128
        self.height, self.width = self.occupancy_grid.shape
        self.resize_map(desired_size)
        self.show_grid = show_grid

    def resize_map(self, desired_size):
        pixels = int(desired_size / self.grid_resolution)
        self.img = self.img.resize((pixels, pixels))
        self.occupancy_grid = np.array(self.img) < 128
        self.height, self.width = self.occupancy_grid.shape

    def world_to_map(self, x, y):
        mx = int(x / self.grid_resolution)
        my = int(y / self.grid_resolution)
        return mx, self.height - my - 1

    def is_obstacle(self, x, y):
        mx, my = self.world_to_map(x, y)
        if 0 <= mx < self.width and 0 <= my < self.height:
            return self.occupancy_grid[my, mx]
        return True

    def draw_grid(self, ax):
        if self.show_grid:
            for y in np.arange(0, self.height * self.grid_resolution, self.grid_resolution):
                ax.axhline(y=y, color='black', linewidth=0.5, linestyle='--')
            for x in np.arange(0, self.width * self.grid_resolution, self.grid_resolution):
                ax.axvline(x=x, color='black', linewidth=0.5, linestyle='--')

    def save_occupancy_grid_to_csv(self, filename="occupancy_grid.csv"):
        np.savetxt(filename, self.occupancy_grid.astype(int), delimiter=',', fmt='%d')
