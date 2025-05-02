import random
import numpy as np
from LidarBot.my_bot import MyBot

ROBOT_DIAMETER = 0.5  # meters
SAFETY_MARGIN = 0.2   # meters
MAP_SIZE = 5.0        # 5m x 5m map size

def is_within_map(x, y):
    return 0 <= x <= MAP_SIZE and 0 <= y <= MAP_SIZE

def is_clear_of_obstacles(x, y, bot, radius=ROBOT_DIAMETER + SAFETY_MARGIN):
    scan = bot.get_lidar_msg()
    angle_min = scan["angle_min"]
    angle_increment = scan["angle_increment"]
    ranges = scan["ranges"]
    robot_x, robot_y, robot_theta = bot.get_pose()  # Get the robot's current pose

    for angle_idx, distance in enumerate(ranges):
        if np.isinf(distance) or np.isnan(distance):
            continue
        
        angle = angle_min + angle_idx * angle_increment
        angle_relative_to_robot = angle + robot_theta
        
        dx = x - robot_x
        dy = y - robot_y

        if np.sqrt(dx**2 + dy**2) < radius and distance > np.sqrt(dx**2 + dy**2):
            return True  # There's no obstacle in the way

    return False  # Obstacle detected

def generate_random_goal_from_lidar(bot, radius=1.0, max_attempts=30):
    try:
        scan = bot.get_lidar_msg()
        ranges = scan["ranges"]
        angle_min = scan["angle_min"]
        angle_increment = scan["angle_increment"]
    except Exception as e:
        print(f"Error: Failed to get LiDAR scan: {e}")
        return bot.get_pose()[:2], bot.get_pose()[2]

    if not ranges:
        print("Error: LiDAR scan data is empty.")
        return bot.get_pose()[:2], bot.get_pose()[2]  # Fallback to current position

    for _ in range(max_attempts):
        i = random.randint(0, len(ranges) - 1)
        distance = ranges[i]

        if np.isinf(distance) or np.isnan(distance) or distance < (ROBOT_DIAMETER + SAFETY_MARGIN):
            continue

        travel_dist = random.uniform(ROBOT_DIAMETER + SAFETY_MARGIN, min(radius, distance))

        angle = angle_min + i * angle_increment
        x, y, _ = bot.get_pose()
        dx = travel_dist * np.cos(angle)
        dy = travel_dist * np.sin(angle)
        new_x = x + dx
        new_y = y + dy

        if is_within_map(new_x, new_y) and is_clear_of_obstacles(new_x, new_y, bot):
            print(f"Generated goal: ({new_x:.2f}, {new_y:.2f}) at angle {angle:.2f}")
            return (new_x, new_y), angle

    x, y, theta = bot.get_pose()
    print("No valid goal found, staying in place.")
    return (x, y), theta

def run_random_exploration(bot, steps=200, replan_interval=20, radius=1.0):
    for step in range(steps):
        if step % replan_interval == 0:
            pos, angle = generate_random_goal_from_lidar(bot, radius=radius)
            bot.set_goal(position=pos, angle=angle)
        
        bot.step()
