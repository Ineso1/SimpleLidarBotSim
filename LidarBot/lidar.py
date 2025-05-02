import numpy as np

class Lidar:
    def __init__(self, max_distance=1.0, noise_stddev=0.02):
        self.max_distance = max_distance
        self.noise_stddev = noise_stddev

    def measure(self, pose, map_data, angle):
        x0, y0, _ = pose
        for d in np.linspace(0, self.max_distance, 100):
            x = x0 + d * np.cos(angle)
            y = y0 + d * np.sin(angle)
            if map_data.is_obstacle(x, y):
                noisy_d = np.clip(d + np.random.normal(0, self.noise_stddev), 0, self.max_distance)
                return x0 + noisy_d * np.cos(angle), y0 + noisy_d * np.sin(angle)
        noisy_d = np.clip(self.max_distance + np.random.normal(0, self.noise_stddev), 0, self.max_distance)
        return x0 + noisy_d * np.cos(angle), y0 + noisy_d * np.sin(angle)
