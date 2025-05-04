import numpy as np
import matplotlib.pyplot as plt
from LidarBot.my_bot2 import MyBot

count = 0
def main():

    bot = MyBot()
    bot.set_goal((4.0, 3.0))  # Set target position
    
    count = count + 1

    if count < 10:    
        bot.loop()  # Start autonomous navigation
    bot.animate2()  # Visualize the results


'''
    initial_pose = (1.0, 1.0, 0.0)
    bot = MyBot(initial_pose=initial_pose)

    waypoints = [
        (0.5, 0.5, 0.0),
        (4.5, 0.5, np.pi/2),
        (4.5, 1.75, np.pi),
        (0.5, 1.75, np.pi/4),
        (0.5, 4.5, 0.0),
        (4.5, 4.5, 0.0),
        (4.5, 3.0, np.pi/2),
        (2.5, 3.0, 0.0),
    ]

    for x, y, theta in waypoints:
        bot.set_goal(position=(x, y), angle=theta)
        bot.loop()

    bot.animate2()
'''

if __name__ == "__main__":
    main()
