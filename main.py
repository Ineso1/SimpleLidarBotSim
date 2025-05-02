import numpy as np
import time
from LidarBot.map_handler import MyMap
from LidarBot.my_bot import MyBot 
from Exploration.exploration_strategy import FrontierExploration

def main():
    # Load the real map to initialize high-level robot interface
    m = MyMap("map1.png", grid_resolution=0.1, desired_size=5, show_grid=True)
    bot = MyBot(map_data=m, initial_pose=(2.0, 3.0, 0.0))

    # Set a static target for now
    bot.set_goal(position=(1.0, 2.0), angle=np.pi/2)

    # Run simulation
    bot.run(steps=200)

    # Animate result
    bot.animate()

if __name__ == "__main__":
    main()
