import numpy as np
import matplotlib.pyplot as plt
import heapq
import random
import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Maze size (must be odd for full walls)
WIDTH, HEIGHT = 21, 21

def initialize_maze():
    return np.ones((HEIGHT, WIDTH), dtype=int)  # 1 for walls, 0 for paths

def carve_maze(maze, x, y):
    maze[y][x] = 0  # Mark as path
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    random.shuffle(directions)
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 1 <= nx < WIDTH-1 and 1 <= ny < HEIGHT-1 and maze[ny][nx] == 1:
            maze[y + dy//2][x + dx//2] = 0  # Remove wall
            carve_maze(maze, nx, ny)

def generate_maze():
    maze = initialize_maze()
    carve_maze(maze, 1, 1)  # Start at (1,1)
    maze[0][1] = 0  # Entrance
    maze[HEIGHT-1][WIDTH-2] = 0  # Exit
    return maze

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Reverse path
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT and maze[neighbor[1]][neighbor[0]] == 0:
                temp_g = g_score[current] + 1
                if neighbor not in g_score or temp_g < g_score[neighbor]:
                    g_score[neighbor] = temp_g
                    f_score[neighbor] = temp_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current
    return []  # No path found

def visualize_maze(maze, path=None, animate=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(maze, cmap='Greens')  # Green maze
    
    rat_img = plt.imread("Rat.png")
    imagebox = OffsetImage(rat_img, zoom=0.03)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False)
    ax.add_artist(ab)
    
    start_time = time.time()
    timer_text = ax.text(0, -2, "Time: 0.0s", fontsize=12, color='black')
    
    if path:
        if animate:
            for i in range(len(path)):
                ab.xybox = path[i][0], path[i][1]
                ax.plot([p[0] for p in path[:i]], [p[1] for p in path[:i]], color='red', linewidth=2)
                elapsed_time = time.time() - start_time
                timer_text.set_text(f"Time: {elapsed_time:.2f}s")
                plt.pause(0.05)
        else:
            ax.plot([p[0] for p in path], [p[1] for p in path], color='red', linewidth=2)
    
    plt.xticks([])
    plt.yticks([])
    plt.show()

maze = generate_maze()
start, goal = (1, 0), (WIDTH-2, HEIGHT-1)
path = astar(maze, start, goal)
visualize_maze(maze, path, animate=True)