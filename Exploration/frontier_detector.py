import numpy as np

class FrontierDetector:
    def __init__(self, robot):
        """
        Initializes the frontier detector.
        :param robot: The LidarBot instance to gather Lidar scan data.
        """
        self.robot = robot

    def detect_frontiers(self):
        """Detect frontiers based on the robot's current Lidar scan."""
        frontiers = []
        scan = self.robot.scan
        
        # Identify frontiers by checking free space with unknown space around it
        for i, distance in enumerate(scan.ranges):
            # Check if the point is free and surrounded by unknown space
            if distance < scan.range_max:
                x, y, _ = self.robot.get_pose()
                angle = scan.angle_min + i * scan.angle_increment
                frontier_point = self.calculate_point(x, y, angle, distance)
                if self.is_frontier(x, y, frontier_point):
                    frontiers.append(frontier_point)  # Ensure this is a tuple (x, y)
        return frontiers

    def calculate_point(self, x, y, angle, distance):
        """Calculate the coordinates of a Lidar point."""
        x_point = x + distance * np.cos(angle)
        y_point = y + distance * np.sin(angle)
        return x_point, y_point  # Ensure this is a tuple (x, y)

    def is_frontier(self, x, y, frontier_point):
        """Check if the point is near an unknown area (boundary between free and unknown)."""
        # Return True for now as we assume every point is a frontier.
        # In practice, you might check whether the point is near unknown space.
        return True
