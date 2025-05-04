import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Constants
l = 0.2  # Distance between wheels

# RRT Parameters
MAX_ITER = 1000
STEP_SIZE = 0.1
GOAL_TOLERANCE = 0.1

# Occupancy Grid parameters
min_x = -4.0
max_x = 3.0
min_y = -3.0
max_y = 3.0
resolution = 0.1  # meters per cell

# Calculate grid dimensions
width = int((max_x - min_x) / resolution) + 1
height = int((max_y - min_y) / resolution) + 1

# Initialize occupancy grid
occupancy_grid = np.zeros((height, width), dtype=np.uint8)

occupancy_grid[40:60, 40:50] = 1
occupancy_grid[20:30, 40:60] = 1
occupancy_grid[10:30, 30:40] = 1
occupancy_grid[10:20, 0:20] = 1


# Define original obstacles [x, y, radius]
original_obstacles = [(-0.5, 0.5, 0.0)]

# Convert original obstacles to occupancy grid
for obstacle in original_obstacles:
    ox, oy, orad = obstacle
    # Calculate grid bounds for the obstacle
    x_min = ox - orad
    x_max = ox + orad
    y_min = oy - orad
    y_max = oy + orad

    # Convert to grid indices
    j_min = max(0, int((x_min - min_x) // resolution))
    j_max = min(width-1, int((x_max - min_x) // resolution))
    i_min = max(0, int((max_y - y_max) // resolution))
    i_max = min(height-1, int((max_y - y_min) // resolution))

    # Iterate over the bounded grid cells
    for j in range(j_min, j_max + 1):
        for i in range(i_min, i_max + 1):
            wx = min_x + j * resolution
            wy = max_y - i * resolution
            if (wx - ox)**2 + (wy - oy)**2 <= orad**2:
                occupancy_grid[i, j] = 1

# Initialize start and goal positions
start = (1, 2)
goal = (-2, -1)

# RRT Node Class
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos  # (x, y)
        self.parent = parent

# Convert world coordinates to grid indices
def world_to_grid(x, y):
    j = int(round((x - min_x) / resolution))
    i = int(round((max_y - y) / resolution))
    return (i, j)

# Get all grid cells along the line between p1 and p2 using Bresenham's algorithm
def get_line_cells(p1, p2):
    i1, j1 = world_to_grid(p1[0], p1[1])
    i2, j2 = world_to_grid(p2[0], p2[1])
    
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

# Collision check using occupancy grid
def collision_check(p1, p2):
    cells = get_line_cells(p1, p2)
    for (i, j) in cells:
        if i < 0 or i >= height or j < 0 or j >= width:
            return True
        if occupancy_grid[i, j] == 1:
            return True
    return False

# RRT Algorithm
def rrt(start, goal):
    tree = [Node(start)]
    
    for _ in range(MAX_ITER):
        if np.random.random() < 0.05:
            rand = goal
        else:
            while True:
                rand_x = np.random.uniform(min_x, max_x)
                rand_y = np.random.uniform(min_y, max_y)
                i, j = world_to_grid(rand_x, rand_y)
                if (0 <= i < height and 0 <= j < width) and occupancy_grid[i, j] == 0:
                    rand = (rand_x, rand_y)
                    break
        
        nearest = min(tree, key=lambda node: (node.pos[0]-rand[0])**2 + (node.pos[1]-rand[1])**2)
        theta = np.arctan2(rand[1]-nearest.pos[1], rand[0]-nearest.pos[0])
        new_pos = (
            nearest.pos[0] + STEP_SIZE*np.cos(theta),
            nearest.pos[1] + STEP_SIZE*np.sin(theta)
        )
        
        if not collision_check(nearest.pos, new_pos):
            new_node = Node(new_pos, nearest)
            tree.append(new_node)
            
            if np.hypot(new_pos[0]-goal[0], new_pos[1]-goal[1]) <= GOAL_TOLERANCE:
                print("Path found!")
                return tree, new_node
    
    print("No path found")
    return tree, None

# Extract path from RRT tree
def extract_path(goal_node):
    path = []
    current = goal_node
    while current is not None:
        path.append(current.pos)
        current = current.parent
    path.reverse()
    path.append(goal)
    return path

# Generate RRT path
tree, goal_node = rrt(start, goal)
if goal_node is None:
    exit()

path = extract_path(goal_node)

# Simulation parameters
dt = 0.01
Tf = 45
current_waypoint = 0
xd, yd = path[current_waypoint]

# Robot initial state
x, y, theta = start[0], start[1], 0
thetaeint = 0  # Integral of orientation error

# Create figure and axes
fig, ax = plt.subplots(figsize=(8,8))
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.grid(True)

# Draw occupancy grid
grid_rgba = np.zeros((height, width, 4))
grid_rgba[occupancy_grid == 1] = [0.5, 0.5, 0.5, 0.3]  # Gray with alpha 0.3
ax.imshow(grid_rgba, extent=(min_x, max_x, min_y, max_y), origin='upper', interpolation='none')

# Draw RRT tree
for node in tree:
    if node.parent:
        ax.plot([node.pos[0], node.parent.pos[0]], 
                [node.pos[1], node.parent.pos[1]], 
                color='green', linewidth=0.5, alpha=0.3)

# Draw path
path_x = [p[0] for p in path]
path_y = [p[1] for p in path]
ax.plot(path_x, path_y, 'r-', linewidth=2, marker='o', markersize=5)
ax.plot(start[0], start[1], 'bo', markersize=10, label='Start')
ax.plot(goal[0], goal[1], 'go', markersize=10, label='Goal')

# Robot visualization elements
robot_body, = ax.plot([], [], 'bo', markersize=7)
robot_heading, = ax.plot([], [], 'r-', linewidth=1)

def update(frame):
    global x, y, theta, thetaeint, current_waypoint, xd, yd
    
    # Check if reached current waypoint
    dist = np.hypot(xd - x, yd - y)
    if dist < 0.1:
        current_waypoint += 1
        if current_waypoint >= len(path):
            return robot_body, robot_heading
        xd, yd = path[current_waypoint]
    
    # Position control
    thetad = np.arctan2(yd - y, xd - x)
    
    # Orientation error
    thetae = theta - thetad
    if thetae > np.pi:
        thetae -= 2*np.pi
    elif thetae < -np.pi:
        thetae += 2*np.pi
    
    # PI controller for orientation
    thetaeint += thetae * dt
    w = -3*thetae - 0.03*thetaeint
    
    # Distance control
    d = np.sqrt((xd - x)**2 + (yd - y)**2)
    V = 1*d
    
    # Wheel velocities
    vl = V - w*l/2
    vr = V + w*l/2
    
    # Update robot state
    V_robot = (vl + vr)/2
    w_robot = (vr - vl)/l
    
    x += V_robot * np.cos(theta) * dt
    y += V_robot * np.sin(theta) * dt
    theta += w_robot * dt
    
    # Update visualization
    robot_body.set_data(x, y)
    heading_line = [
        [x, x + l * np.cos(theta)],
        [y, y + l * np.sin(theta)]
    ]
    robot_heading.set_data(heading_line)
    
    return robot_body, robot_heading

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, Tf, dt),
                   blit=True, repeat=False, interval=20)

plt.legend()
plt.show()
