import numpy as np
import matplotlib.pyplot as plt
from LidarBot.my_bot import MyBot


def main():
    # Initialize bot at a known starting location
    initial_pose = (1.0, 1.0, 0.0)
    bot = MyBot(initial_pose=initial_pose)

    # Define a list of waypoints (x, y, theta)
    waypoints = [
        (0.5, 0.5, 0.0),
        (4.5, 0.5, 0.0),
        (4.5, 1.75, 0.0),
        (0.5, 1.75, 0.0),
        (0.5, 4.5, 0.0),
        (4.5, 4.5, 0.0),
        (4.5, 3.0, 0.0),
        (2.5, 3.0, 0.0),
    ]


    for x, y, theta in waypoints:
        bot.set_goal(position=(x, y), angle=theta)
        bot.loop()


    bot.animate2()

if __name__ == "__main__":
    main()
