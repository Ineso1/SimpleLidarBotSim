import numpy as np

def plan_path_to(start_pose, goal_position, map_resolution=0.1):
    """
    Simple A* path planning to generate a path from the robot to the frontier goal.
    :param start_pose: Robot's current pose (x, y, theta).
    :param goal_position: Goal position (x, y).
    :param map_resolution: Grid resolution (meters per cell).
    :return: List of waypoints or an empty list if no path is found.
    """
    # For simplicity, using a basic straight line as a placeholder for path planning.
    # Real path planning would involve A* or other search algorithms.
    
    start_x, start_y, _ = start_pose
    goal_x, goal_y = goal_position
    
    # Generate a straight line path (for simplicity).
    waypoints = []
    num_points = int(np.hypot(goal_x - start_x, goal_y - start_y) / map_resolution)
    
    for i in range(num_points):
        x = start_x + (goal_x - start_x) * i / num_points
        y = start_y + (goal_y - start_y) * i / num_points
        waypoints.append((x, y))
    
    return waypoints
