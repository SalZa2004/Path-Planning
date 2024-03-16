import cv2
from matplotlib import pyplot as plt
import numpy as np
import heapq


MOVEMENTS = [(1, 0), (-1, 0), (0, 1), (0, -1)] # (up, down, left, right)

# Chosen heuristic function = Euclidean
def euclidean(node, goal):
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

# Image processing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    return blurred_image

def detect_edges(image):
    edges = cv2.Canny(image, 50, 100)
    return edges

def generate_maze(edges):
    maze = np.where(edges > 0,1,0)  
    return maze


GRID_SIZE = 25

def convert_to_grid(maze):
    grid = np.zeros_like(maze)
    for i in range(0, maze.shape[0], GRID_SIZE):
        for j in range(0, maze.shape[1], GRID_SIZE):
            if np.sum(maze[i:i+GRID_SIZE, j:j+GRID_SIZE]) < 1:
                grid[i:i+GRID_SIZE, j:j+GRID_SIZE] = 1  
    return grid

# Path planning using A*
def astar(grid, start, goal):
    open_list = []
    closed_set = set()
    came_from = {}

    g_n = {start: 0}
    f_n = {start: euclidean(start, goal)}

    heapq.heappush(open_list, (f_n[start], start))

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        closed_set.add(current)

        for move in MOVEMENTS:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1] and grid[neighbor] == 0:
                initial_g_n = g_n[current] + 1  

                if neighbor in closed_set and initial_g_n >= g_n.get(neighbor, float('inf')):
                    continue

                if initial_g_n < g_n.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_n[neighbor] = initial_g_n
                    f_n[neighbor] = initial_g_n + euclidean(neighbor, goal)
                    heapq.heappush(open_list, (f_n[neighbor], neighbor))

    return None  


# Main function
def main():
    
    image_path = 'MY2023 rescaled 00.png'  
    preprocessed_image = preprocess_image(image_path)
    
    edges = detect_edges(preprocessed_image)
    
    maze = generate_maze(edges)
     
    grid = convert_to_grid(maze)

    # Define start and goal positions on the grid [Coordinates]
    start = (2424,3671)  
    goal = (955,2775)  

    # Run A* algorithm
    path = astar(grid, start, goal)

    if path:
        print("Path found:", path)
        
        visualize_path(grid, path)
    else:
        print("No path found")

def visualize_path(grid, path, color ='red'):
    plt.imshow(grid, cmap='gray')
    plt.plot([x[1] for x in path], [x[0] for x in path], color='red', linewidth=2)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
