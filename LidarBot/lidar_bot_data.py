import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

class LidarBotData:
    def __init__(self, map):
        self.map = map                     # map
        self.pose_history = []             # (x, y, theta)
        self.velocity_history = []         # (vx, vy, omega)
        self.error_x_history = []          # x-position error
        self.error_y_history = []          # y-position error
        self.error_theta_history = []      # angular error

    def record(self, pose, velocity, target_pose, target_angle):
        self.pose_history.append(pose)
        self.velocity_history.append(velocity)

        x, y, theta = pose
        x_d, y_d = target_pose

        error_x = x_d - x
        error_y = y_d - y
        error_theta = self.normalize_angle(target_angle - theta)

        self.error_x_history.append(error_x)
        self.error_y_history.append(error_y)
        self.error_theta_history.append(error_theta)

    def plot_errors(self):
        t = range(len(self.error_x_history))
        plt.figure(figsize=(10, 6))
        plt.plot(t, self.error_x_history, label='X Error')
        plt.plot(t, self.error_y_history, label='Y Error')
        plt.plot(t, self.error_theta_history, label='Theta Error')
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.title("PID Tracking Errors")
        plt.legend()
        plt.grid(True)
        plt.show()

    def animate_trajectory(self):
        poses = np.array(self.pose_history)
        plt.plot(poses[:, 0], poses[:, 1], label="Bot Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Trajectory")
        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def plotMap(self):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(~self.map.occupancy_grid, cmap='gray', origin='upper',
                extent=[0, self.map.width * self.map.grid_resolution, 0, self.map.height * self.map.grid_resolution])
        ax.set_aspect('equal')
        self.map.draw_grid(ax)
        plt.tight_layout()
        plt.show()

    def animate_bot_on_map(self):
        poses = np.array(self.pose_history)
        xs, ys, thetas = poses[:, 0], poses[:, 1], poses[:, 2]

        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw the map
        ax.imshow(~self.map.occupancy_grid, cmap='gray', origin='upper',
                extent=[0, self.map.width * self.map.grid_resolution, 0, self.map.height * self.map.grid_resolution])
        ax.set_aspect('equal')
        self.map.draw_grid(ax)

        # Trajectory line
        path_line, = ax.plot([], [], 'b-', label="Path")
        robot_marker, = ax.plot([], [], 'ro', label="Bot")
        direction_line, = ax.plot([], [], 'r-')

        max_beams = 1000
        lidar_rays = [ax.plot([], [], 'g-', alpha=0.3)[0] for _ in range(max_beams)]

        def init():
            path_line.set_data([], [])
            robot_marker.set_data([], [])
            direction_line.set_data([], [])
            for ray in lidar_rays:
                ray.set_data([], [])
            return [path_line, robot_marker, direction_line] + lidar_rays

        def update(frame):
            path_line.set_data(xs[:frame], ys[:frame])
            robot_marker.set_data([xs[frame]], [ys[frame]])

            dx = 0.2 * np.cos(thetas[frame])
            dy = 0.2 * np.sin(thetas[frame])
            direction_line.set_data(
                [xs[frame], xs[frame] + dx],
                [ys[frame], ys[frame] + dy]
            )

            # Lidar rays
            if hasattr(self, 'scan_history'):
                scan = self.scan_history[frame]
                angle_min = scan['angle_min']
                angle_inc = scan['angle_inc']
                ranges = scan['ranges']
                angle_base = thetas[frame]
                x0, y0 = xs[frame], ys[frame]
                for i, r in enumerate(ranges):
                    if i >= max_beams:
                        break
                    angle = angle_base + angle_min + i * angle_inc
                    x1 = x0 + r * np.cos(angle)
                    y1 = y0 + r * np.sin(angle)
                    lidar_rays[i].set_data([x0, x1], [y0, y1])
            return [path_line, robot_marker, direction_line] + lidar_rays

        ani = animation.FuncAnimation(fig, update, frames=len(xs), init_func=init, blit=True, interval=50, repeat=False)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def draw_lidar_scan(self, bot):
        pose = bot.pose
        x0, y0, theta = pose
        angles = np.arange(bot.scan.angle_min, bot.scan.angle_max, bot.scan.angle_increment)
        ranges = bot.scan.ranges

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw map
        ax.imshow(~self.map.occupancy_grid, cmap='gray', origin='upper',
                extent=[0, self.map.width * self.map.grid_resolution,
                        0, self.map.height * self.map.grid_resolution])
        
        # Draw robot
        ax.plot(x0, y0, 'ro', label="Robot")

        # Draw LIDAR rays
        for angle, dist in zip(angles, ranges):
            end_x = x0 + dist * np.cos(angle + theta)
            end_y = y0 + dist * np.sin(angle + theta)
            ax.plot([x0, end_x], [y0, end_y], 'b-', alpha=0.3)

        ax.set_aspect('equal')
        ax.set_title("LIDAR Scan")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def animate_bot_with_mapping(self, occupancy_map):
        poses = np.array(self.pose_history)
        xs, ys, thetas = poses[:, 0], poses[:, 1], poses[:, 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Robot environment view
        ax1.set_title("Robot on Environment Map")
        env_img = ax1.imshow(~self.map.occupancy_grid, cmap='gray', origin='upper',
                            extent=[0, self.map.width * self.map.grid_resolution,
                                    0, self.map.height * self.map.grid_resolution])
        ax1.set_aspect('equal')
        self.map.draw_grid(ax1)
        path_line, = ax1.plot([], [], 'b-', label="Path")
        robot_marker, = ax1.plot([], [], 'ro', label="Bot")
        direction_line, = ax1.plot([], [], 'r-')
        max_beams = 1000
        lidar_rays = [ax1.plot([], [], 'g-', alpha=0.3)[0] for _ in range(max_beams)]
        ax1.legend()

        ax2.set_title("Occupancy Map Being Built")
        map_img = ax2.imshow(1 - occupancy_map.occupancy_grid, cmap='gray', origin='lower',
                     extent=[-5.5, 5.5, -5.5, 5.5], vmin=0, vmax=1)
        ax2.set_xlim(-0.5, 5.5)
        ax2.set_ylim(-0.5, 5.5)
        ax2.set_aspect('equal')

        def init():
            path_line.set_data([], [])
            robot_marker.set_data([], [])
            direction_line.set_data([], [])
            for ray in lidar_rays:
                ray.set_data([], [])
            return [env_img, map_img, path_line, robot_marker, direction_line] + lidar_rays

        def update(frame):
            # Update robot path
            path_line.set_data(xs[:frame], ys[:frame])
            robot_marker.set_data([xs[frame]], [ys[frame]])
            dx = 0.2 * np.cos(thetas[frame])
            dy = 0.2 * np.sin(thetas[frame])
            direction_line.set_data([xs[frame], xs[frame] + dx],
                                    [ys[frame], ys[frame] + dy])

            # Lidar rays
            if hasattr(self, 'scan_history'):
                scan = self.scan_history[frame]
                angle_min = scan['angle_min']
                angle_inc = scan['angle_inc']
                ranges = scan['ranges']
                angle_base = thetas[frame]
                x0, y0 = xs[frame], ys[frame]
                for i, r in enumerate(ranges):
                    if i >= max_beams:
                        break
                    angle = angle_base + angle_min + i * angle_inc
                    x1 = x0 + r * np.cos(angle)
                    y1 = y0 + r * np.sin(angle)
                    lidar_rays[i].set_data([x0, x1], [y0, y1])

            # Update occupancy map from history (if available)
            if frame < len(occupancy_map.history):
                map_img.set_data(1 - occupancy_map.history[frame])

            return [map_img, path_line, robot_marker, direction_line] + lidar_rays

        ani = animation.FuncAnimation(fig, update, frames=len(xs), init_func=init,
                                    blit=True, interval=1, repeat=False)
        plt.tight_layout()
        plt.show()


