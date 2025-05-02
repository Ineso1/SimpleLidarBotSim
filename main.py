import numpy as np
import matplotlib.pyplot as plt
from LidarBot.my_bot import MyBot
from Exploration.random_exploration import run_random_exploration

def plot_occupancy_map(occupancy_map):
    """Visualizes the occupancy map."""
    # Create a figure and axis for plotting
    plt.figure(figsize=(6, 6))
    
    # Plot the occupancy map
    # -1: unknown, 0: free, 1: occupied
    plt.imshow(occupancy_map, cmap='gray', origin='lower', interpolation='nearest')
    plt.colorbar(label="Occupancy")
    plt.title("Occupancy Map")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()

def main():
    # Initialize the bot at a chosen starting pose within the 5x5 map
    initial_pose = (1.0, 2.0, np.pi / 2)  # x, y, heading (facing upward)
    bot = MyBot(initial_pose=initial_pose)

    # Parameters for exploration
    total_steps = 300           # Total number of simulation steps
    replan_interval = 20        # How often to choose a new random goal
    exploration_radius = 0.5    # Max distance to attempt traveling each replan

    # Start the random exploration
    run_random_exploration(
        bot=bot,
        steps=total_steps,
        replan_interval=replan_interval,
        radius=exploration_radius
    )

    # Visualize the occupancy map after the exploration
    plot_occupancy_map(bot.occupancy_map)

    # Visualize the path and explored areas
    bot.animate()

if __name__ == "__main__":
    main()
