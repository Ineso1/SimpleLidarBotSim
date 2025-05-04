import numpy as np
import matplotlib.pyplot as plt
from .lidar_bot import LidarBot
from Exploration.occupancy_map import OccupancyMap

class MyBot:
    def __init__(self, initial_pose=(2.0, 3.0, 0.0), map_size=5.0, resolution=0.01):
        self.robot = LidarBot(pose=initial_pose)
        self.exploration_strategy = None
        self.trajectory_generator = None
        self.dt = 0.1
        self.map_size = map_size
        self.resolution = resolution
        self.pointAchived = False
        self.occupancy_map = OccupancyMap(initial_size=100, resolution=resolution)
        
        # RRT* Parameters
        self.MAX_ITER = 500
        self.STEP_SIZE = 0.2
        self.GOAL_TOLERANCE = 0.15
        self.current_path = []
        self.current_waypoint = 0

    class Node:
        def __init__(self, pos, parent=None):
            self.pos = pos  # (x, y)
            self.parent = parent

    def get_line_cells(self, p1, p2):
        """Bresenham's line algorithm for occupancy map checks"""
        i1, j1 = self.occupancy_map.world_to_grid(p1[0], p1[1])
        i2, j2 = self.occupancy_map.world_to_grid(p2[0], p2[1])
        
        cells = []
        dx = abs(j2 - j1)
        dy = abs(i2 - i1)
        sx = 1 if j1 < j2 else -1
        sy = 1 if i1 < i2 else -1
        err = dx - dy
        
        current_j, current_i = j1, i1
        cells.append((current_i, current_j))
        
        while not (current_i == i2 and current_j == j2):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_j += sx
            if e2 < dx:
                err += dx
                current_i += sy
            cells.append((current_i, current_j))
        
        return cells

    def collision_check(self, p1, p2):
        """Check path between two points using occupancy map"""
        cells = self.get_line_cells(p1, p2)
        for (i, j) in cells:
            if not self.occupancy_map.is_in_bounds(i, j):
                return True
            if self.occupancy_map.occupancy_grid[i, j] <= 0.5:
                return True
        return False

    def rrt_plan(self, start, goal):
        """RRT* path planning algorithm"""
        tree = [self.Node(start)]
        
        for _ in range(self.MAX_ITER):
            # Random sample with bias towards goal
            if np.random.random() < 0.1:
                rand = goal
            else:
                # Generate random point in known free space
                while True:
                    min_x = self.occupancy_map.origin_x
                    max_x = self.occupancy_map.origin_x + self.occupancy_map.width * self.resolution
                    min_y = self.occupancy_map.origin_y
                    max_y = self.occupancy_map.origin_y + self.occupancy_map.height * self.resolution
                    
                    rand_x = np.random.uniform(min_x, max_x)
                    rand_y = np.random.uniform(min_y, max_y)
                    i, j = self.occupancy_map.world_to_grid(rand_x, rand_y)
                    
                    if self.occupancy_map.is_in_bounds(i, j) and self.occupancy_map.occupancy_grid[i, j] >= 0.8:
                        rand = (rand_x, rand_y)
                        break
            
            # Find nearest node
            nearest = min(tree, key=lambda node: 
                         (node.pos[0]-rand[0])**2 + (node.pos[1]-rand[1])**2)
            
            # Steer towards sample
            theta = np.arctan2(rand[1]-nearest.pos[1], rand[0]-nearest.pos[0])
            new_pos = (
                nearest.pos[0] + self.STEP_SIZE*np.cos(theta),
                nearest.pos[1] + self.STEP_SIZE*np.sin(theta)
            )
            
            # Check collision
            if not self.collision_check(nearest.pos, new_pos):
                new_node = self.Node(new_pos, nearest)
                tree.append(new_node)
                
                # Check goal proximity
                if np.hypot(new_pos[0]-goal[0], new_pos[1]-goal[1]) <= self.GOAL_TOLERANCE:
                    print("RRT* path found!")
                    return self.extract_path(new_node)
        
        print("RRT* failed to find path")
        return []

    def extract_path(self, goal_node):
        """Extract path from RRT* tree"""
        path = []
        current = goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        path.reverse()
        return path

    def set_goal(self, position, angle=np.pi/2, velocity=(0.0, 0.0, 0.0)):
        """Set final goal and plan path"""
        self.final_goal = position
        start = self.get_pose()[:2]
        self.current_path = self.rrt_plan(start, position)
        self.current_waypoint = 0
        
        if self.current_path:
            print(f"Planned path with {len(self.current_path)} waypoints")
            self.update_waypoint()

    def update_waypoint(self):
        """Update to next waypoint in path"""
        if self.current_waypoint < len(self.current_path):
            waypoint = self.current_path[self.current_waypoint]
            self.robot.set_target(waypoint, (0.0, 0.0, 0.0), np.pi/2)
            self.current_waypoint += 1

    def step(self):
        """Modified step function with path following"""
        # Lidar mapping
        self.robot.perform_scan()
        lidar_msg = self.get_lidar_msg()
        self.update_map_from_lidar(lidar_msg)
        
        # Path following logic
        if self.current_path:
            current_pos = self.get_pose()[:2]
            distance_to_waypoint = np.hypot(
                current_pos[0] - self.robot.target_position[0],
                current_pos[1] - self.robot.target_position[1]
            )
            
            if distance_to_waypoint < 0.1:
                self.update_waypoint()
        
        # Robot control
        self.robot.apply_control(self.dt)
        self.robot.update(self.dt)        
        return lidar_msg

    # Existing methods remain unchanged below
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

    def animate(self):
        self.robot.data.animate_bot_on_map()
        self.occupancy_map.plot()

    def animate2(self):
        self.robot.data.animate_bot_with_mapping(self.occupancy_map)
