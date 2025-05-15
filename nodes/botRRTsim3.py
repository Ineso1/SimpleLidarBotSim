import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray

class RRTWaypoints(Node):
    def __init__(self):
        super().__init__('rrt_planner')
        
        # Subscribers and publishers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.waypoint_pub = self.create_publisher(
            Float32MultiArray, 
            '/waypoints', 
            10
        )
        
        # Initialize map parameters
        self.grid = None
        self.map_info = None
        
        # RRT parameters (aligned with simulation)
        self.START = (1.0, 1.0)    # World coordinates
        self.GOAL = (8.0, 8.0)     # World coordinates
        self.MAX_ITER = 5000
        self.STEP_SIZE = 0.5
        self.GOAL_TOLERANCE = 0.15

    def map_callback(self, msg):
        self.get_logger().info("Received map")
        self.map_info = msg.info
        
        # Convert occupancy grid (0-100 to binary 0/1)
        self.grid = np.array(msg.data, dtype=np.uint8).reshape(
            msg.info.height, 
            msg.info.width
        )
        self.grid = np.where(self.grid >= 50, 1, 0)  # Threshold at 50%
        
        # Validate start and goal positions
        start_valid, goal_valid = self.validate_positions()
        if not (start_valid and goal_valid):
            return
        
        # Run RRT planner
        planner = RRTPlanner(
            self.START,
            self.GOAL,
            self.grid,
            self.map_info,
            self.MAX_ITER,
            self.STEP_SIZE,
            self.GOAL_TOLERANCE
        )
        
        path = planner.generate_path()
        if path:
            self.publish_path(path)
        else:
            self.get_logger().warn("No path found")

    def validate_positions(self):
        # Convert to grid coordinates
        start_i, start_j = self.world_to_grid(*self.START)
        goal_i, goal_j = self.world_to_grid(*self.GOAL)
        
        # Check bounds
        valid = True
        if not (0 <= start_i < self.grid.shape[0] and 0 <= start_j < self.grid.shape[1]):
            self.get_logger().error("Start position out of bounds")
            valid = False
        if not (0 <= goal_i < self.grid.shape[0] and 0 <= goal_j < self.grid.shape[1]):
            self.get_logger().error("Goal position out of bounds")
            valid = False
            
        # Check occupancy
        if valid and self.grid[start_i, start_j] != 0:
            self.get_logger().error("Start position is occupied")
            valid = False
        if valid and self.grid[goal_i, goal_j] != 0:
            self.get_logger().error("Goal position is occupied")
            valid = False
            
        return valid, valid

    def world_to_grid(self, x, y):
        j = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        i = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return i, j

    def publish_path(self, path):
        print("//////////")
        print(path)
        flat_path = [coord for point in path for coord in point]
        msg = Float32MultiArray()
        msg.data = flat_path
        self.waypoint_pub.publish(msg)
        self.get_logger().info(f"Published path with {len(path)} waypoints")

class RRTPlanner:
    def __init__(self, start, goal, grid, map_info, max_iter, step_size, goal_tol):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.map_info = map_info
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_tol = goal_tol
        
        # Calculate world bounds
        self.min_x = map_info.origin.position.x
        self.min_y = map_info.origin.position.y
        self.max_x = self.min_x + map_info.resolution * map_info.width
        self.max_y = self.min_y + map_info.resolution * map_info.height

    def generate_path(self):
        tree, goal_node = self.build_rrt()
        if goal_node is None:
            return None
        return self.extract_path(goal_node)

    def build_rrt(self):
        tree = [RRTNode(self.start)]
        
        for _ in range(self.max_iter):
            # Sample with 10% goal bias
            if np.random.random() < 0.1:
                rand = self.goal
            else:
                rand = self.sample_free()
                if rand is None:
                    continue
                
            # Find nearest node
            nearest = min(tree, key=lambda n: self.distance(n.pos, rand))
            
            # Steer towards random point
            new_pos = self.steer(nearest.pos, rand)
            
            # Collision check
            if not self.check_collision(nearest.pos, new_pos):
                new_node = RRTNode(new_pos, nearest)
                tree.append(new_node)
                
                # Check goal proximity
                if self.distance(new_pos, self.goal) <= self.goal_tol:
                    return tree, new_node
                    
        return tree, None

    def sample_free(self):
        # Try 100 times to find free space
        for _ in range(100):
            x = np.random.uniform(self.min_x, self.max_x)
            y = np.random.uniform(self.min_y, self.max_y)
            i, j = self.world_to_grid(x, y)
            if 0 <= i < self.grid.shape[0] and 0 <= j < self.grid.shape[1]:
                if self.grid[i, j] == 0:
                    return (x, y)
        return None

    def steer(self, from_pos, to_pos):
        theta = np.arctan2(to_pos[1] - from_pos[1], to_pos[0] - from_pos[0])
        return (
            from_pos[0] + self.step_size * np.cos(theta),
            from_pos[1] + self.step_size * np.sin(theta)
        )

    def check_collision(self, p1, p2):
        cells = self.get_line_cells(p1, p2)
        for (i, j) in cells:
            if (i < 0 or i >= self.grid.shape[0] or 
                j < 0 or j >= self.grid.shape[1] or 
                self.grid[i, j] == 1):
                return True
        return False

    def get_line_cells(self, p1, p2):
        i1, j1 = self.world_to_grid(*p1)
        i2, j2 = self.world_to_grid(*p2)
        
        cells = []
        dx = abs(j2 - j1)
        dy = abs(i2 - i1)
        sx = 1 if j1 < j2 else -1
        sy = 1 if i1 < i2 else -1
        err = dx - dy
        
        current_j, current_i = j1, i1
        while True:
            cells.append((current_i, current_j))
            if current_j == j2 and current_i == i2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                current_j += sx
            if e2 < dx:
                err += dx
                current_i += sy
        return cells

    def world_to_grid(self, x, y):
        j = int((x - self.min_x) / self.map_info.resolution)
        i = int((y - self.min_y) / self.map_info.resolution)
        return i, j

    def distance(self, p1, p2):
        return np.hypot(p1[0]-p2[0], p1[1]-p2[1])

    def extract_path(self, goal_node):
        path = []
        current = goal_node
        while current is not None:
            path.append(current.pos)
            current = current.parent
        return list(reversed(path))

class RRTNode:
    def __init__(self, pos, parent=None):
        self.pos = pos  # (x, y) in world coordinates
        self.parent = parent

def main(args=None):
    rclpy.init(args=args)
    node = RRTWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()