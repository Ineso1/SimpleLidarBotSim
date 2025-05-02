import numpy as np
from .lidar_bot_data import LidarBotData
from .pid import PIDController
from .scan import SimulatedLaserScan
from .lidar import Lidar
from .map_handler import MyMap
import os

class LidarBot(Lidar):
    def __init__(self, pose=(2.0, 3.0, 0.0), max_distance=1.5, noise_stddev=0.01, robot_radius=0.2):
        super().__init__(max_distance=max_distance, noise_stddev=noise_stddev)
        map_path = os.path.join(os.path.dirname(__file__), "..", "map1.png")
        map_data = MyMap(map_path, grid_resolution=0.1, desired_size=5, show_grid=True)
        self.pose = pose
        self.map = map_data
        self.robot_radius = robot_radius
        self.scan = SimulatedLaserScan(
            angle_min=-np.pi,
            angle_max=np.pi,
            angle_increment=np.pi / 15,  # 30 beams
            range_min=0.05,
            range_max=max_distance
        )

        self.pid_x = PIDController(1.0, 0.0, 0.0, 0.02)  
        self.pid_y = PIDController(1.0, 0.0, 0.0, 0.02)  
        self.pid_theta = PIDController(1.0, 0.0, 0.0, 0.1) 
        
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0

        self.target_pos  = (0.0, 0.0)
        self.target_vel  = (0.0, 0.0, 0.0)
        self.target_angle  = 0.0

        self.data = LidarBotData(map_data)

    def set_velocity(self, vx, vy, omega):
        self.vx = vx
        self.vy = vy
        self.omega = omega

    def set_target(self, target_pos, target_vel, target_angle):
        self.target_pos  = target_pos
        self.target_vel  = target_vel
        self.target_angle  = target_angle

    def update(self, dt):
        x, y, theta = self.pose
        x += self.vx * dt
        y += self.vy * dt
        theta += self.omega * dt
        self.pose = (x, y, theta)
        self.data.record(self.pose, (self.vx, self.vy, self.omega), self.target_pos, self.target_angle)

    def perform_scan(self):
        self.scan.ranges.clear()
        self.scan.intensities.clear()
        angles = np.arange(self.scan.angle_min, self.scan.angle_max, self.scan.angle_increment)
        x0, y0, theta = self.pose  # <- include theta
        for angle in angles:
            world_angle = angle + theta  # Rotate by robot orientation
            hit_x, hit_y = self.measure(self.pose, self.map, world_angle)
            distance = np.hypot(hit_x - x0, hit_y - y0)
            self.scan.ranges.append(distance)
            self.scan.intensities.append(1.0)
        
        if not hasattr(self.data, 'scan_history'):
            self.data.scan_history = []
        
        self.data.scan_history.append({
            'angle_min': self.scan.angle_min,
            'angle_inc': self.scan.angle_increment,
            'ranges': list(self.scan.ranges),
        })


    def apply_control(self, dt):
        x, y, theta = self.pose
        theta = self.normalize_angle(theta)
        x_d, y_d = self.target_pos
        vx_d, vy_d, omega_d = self.target_vel
        theta_d = self.normalize_angle(self.target_angle)

        self.pid_x.desire_position(x_d, vx_d)
        self.pid_y.desire_position(y_d, vy_d)
        self.pid_theta.desire_position(theta_d, omega_d)  

        control_x = self.pid_x.compute(x, self.vx, dt)
        control_y = self.pid_y.compute(y, self.vy, dt)
        control_theta = self.pid_theta.compute(theta, self.omega, dt)
        
        self.set_velocity(control_x, control_y, control_theta)

    def normalize_angle(self, angle):
            angle = (angle + np.pi) % (2 * np.pi) - np.pi
            return angle
