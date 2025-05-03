import numpy as np
import matplotlib.pyplot as plt

class OccupancyMap:
    def __init__(self, initial_size=100, resolution=0.1):
        """
        initial_size: number of grid cells in each dimension (map is square initially)
        resolution: size of each grid cell in meters
        """
        self.resolution = resolution
        self.width = initial_size
        self.height = initial_size

        # Grid initialized to 0.5 (unknown); shape: (height, width)
        self.occupancy_grid = 0.5 * np.ones((self.height, self.width), dtype=np.float32)

        # Origin (world coordinate corresponding to grid index (0, 0))
        self.origin_x = - (self.width // 2) * resolution
        self.origin_y = - (self.height // 2) * resolution

        self.history = []

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices (i, j)."""
        j = int((x - self.origin_x) / self.resolution)
        i = int((y - self.origin_y) / self.resolution)
        return i, j

    def grid_to_world(self, i, j):
        """Convert grid indices to world coordinates (x, y)."""
        x = self.origin_x + j * self.resolution
        y = self.origin_y + i * self.resolution
        return x, y

    def is_in_bounds(self, i, j):
        return 0 <= i < self.height and 0 <= j < self.width

    def expand_to_include(self, x, y):
        """Expand the map if the point (x, y) lies outside the current grid."""
        i, j = self.world_to_grid(x, y)
        needs_expand = not self.is_in_bounds(i, j)

        if not needs_expand:
            return

        # Compute new dimensions and offsets
        min_x = min(self.origin_x, x - 5 * self.resolution)
        min_y = min(self.origin_y, y - 5 * self.resolution)
        max_x = max(self.origin_x + self.width * self.resolution, x + 5 * self.resolution)
        max_y = max(self.origin_y + self.height * self.resolution, y + 5 * self.resolution)

        new_width = int(np.ceil((max_x - min_x) / self.resolution))
        new_height = int(np.ceil((max_y - min_y) / self.resolution))

        # Create new grid
        new_grid = 0.5 * np.ones((new_height, new_width), dtype=np.float32)

        # Compute offset
        offset_i = int((self.origin_y - min_y) / self.resolution)
        offset_j = int((self.origin_x - min_x) / self.resolution)

        # Copy old grid into new grid
        new_grid[offset_i:offset_i + self.height, offset_j:offset_j + self.width] = self.occupancy_grid

        # Update map info
        self.occupancy_grid = new_grid
        self.height = new_height
        self.width = new_width
        self.origin_x = min_x
        self.origin_y = min_y

    def set_occupancy(self, x, y, value):
        """Set occupancy value at a specific world coordinate."""
        self.expand_to_include(x, y)
        i, j = self.world_to_grid(x, y)
        self.occupancy_grid[i, j] = value

    def get_occupancy(self, x, y):
        """Get occupancy value at a specific world coordinate."""
        if not self.is_in_bounds(*self.world_to_grid(x, y)):
            return 0.5  # Unknown
        i, j = self.world_to_grid(x, y)
        return self.occupancy_grid[i, j]

    def draw_grid(self, ax=None):
        """Optional: draw the grid lines for visualization."""
        if ax is not None:
            for x in np.arange(self.origin_x, self.origin_x + self.width * self.resolution, self.resolution):
                ax.axvline(x, color='gray', lw=0.1)
            for y in np.arange(self.origin_y, self.origin_y + self.height * self.resolution, self.resolution):
                ax.axhline(y, color='gray', lw=0.1)

    def plot(self, show_grid=True):
        """Visualize the occupancy map."""
        fig, ax = plt.subplots(figsize=(6, 6))
        extent = [
            self.origin_x,
            self.origin_x + self.width * self.resolution,
            self.origin_y,
            self.origin_y + self.height * self.resolution,
        ]
        ax.imshow(
            1 - self.occupancy_grid,  # flip so 0=black (occupied), 1=white (free)
            cmap='gray',
            origin='lower',
            extent=extent
        )
        ax.set_title("Occupancy Map")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        if show_grid:
            self.draw_grid(ax)
        plt.show()