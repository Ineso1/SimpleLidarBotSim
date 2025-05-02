import numpy as np
from .frontier_detector import FrontierDetector
from .planner import plan_path_to

class FrontierExploration:
    def __init__(self, robot, map_resolution=0.1):
        """
        Initializes the exploration strategy.
        :param robot: The LidarBot instance to perform actions and gather data.
        :param map_resolution: Resolution of the map grid (meters per cell).
        """
        self.robot = robot
        self.map_resolution = map_resolution
        self.frontier_detector = FrontierDetector(robot)
        self.goal = None

    def update(self):
        """Update the exploration strategy based on the robot's Lidar scan."""
        self.robot.perform_scan()
        self.robot.update(0.1)  # Update robot's pose and state
        
        frontiers = self.frontier_detector.detect_frontiers()
        if frontiers:
            self.goal = self.select_goal(frontiers)

    def set_goal(self, frontiers):
        """Selects the closest frontier to the robot's current position."""
        robot_x, robot_y, _ = self.robot.get_pose()
        closest_frontier = min(frontiers, key=lambda f: np.hypot(f[0] - robot_x, f[1] - robot_y))
        return closest_frontier

    def plan_and_move(self):
        """Generate path to the goal and update robot's state."""
        if self.goal is None:
            return None
        
        path = plan_path_to(self.robot.get_pose(), self.goal)
        if path:
            self.robot.set_target(self.goal)
            return path
        return None
