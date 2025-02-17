#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # Set up argument parser to get the CSV file name from the command line
    parser = argparse.ArgumentParser(description='Plot a 2D trajectory with obstacles from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing trajectory data (with columns: time, x, y)')
    args = parser.parse_args()

    # ----- Configuration -----
    # Fixed dimensions for obstacles
    OBSTACLE_WIDTH = 1    # fixed width for all obstacles
    OBSTACLE_HEIGHT = 0.5   # fixed height for all obstacles
    z = 1.05
    y = 0.07
    # Define obstacles as a list of center positions: (center_x, center_y)
    goal = ([3.1, -1.58, z])

    obstacles= ([[1, 1, z], [2, 1, z], [1, -2, z], [2, -2, z], [-1, 0, z], [-1, 0.5, z], [-1, -0.5, z], [-1, 1, z], [-1, -1, z], [0, 1, z], [-1, -1.5, z], [0, -2, z],
                       [5, 0, z], [5, 0.5, z], [5, -0.5, z], [5, 1, z], [5, -1, z], [5, -1.5, z], [3, -2, z], [4, -2, z], [3, 1, z], [4, 1, z]])
    inner_walls = ([[1, y, z], [2, y, z]])
    obstacles.extend(inner_walls)
    # ----- Read CSV Data -----
    # The CSV file is expected to have columns: time, x, y
    df = pd.read_csv(args.csv_file)

    # Extract the trajectory coordinates
    x = df['x']
    y = df['y']

    # ----- Plotting -----
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each obstacle as a rectangle using fixed width and height.
    # Calculate the lower left corner from the center position.
    for center_x, center_y, _ in obstacles:
        lower_left_x = center_x - OBSTACLE_WIDTH / 2
        lower_left_y = center_y - OBSTACLE_HEIGHT / 2
        rect = patches.Rectangle(
            (lower_left_x, lower_left_y), 
            OBSTACLE_WIDTH, 
            OBSTACLE_HEIGHT,
            linewidth=1, 
            edgecolor='black', 
            facecolor='gray', 
            alpha=0.5
        )
        ax.add_patch(rect)

    # Plot the trajectory as a blue line
    ax.plot(x, y, label='Trajectory', color='blue', linewidth=2)

    # Mark the start point (first coordinate) with a green circle
    ax.plot(x.iloc[0], y.iloc[0], marker='o', markersize=8, color='purple', label='Start')

    # Mark the end point (last coordinate) with a red square
    ax.plot(x.iloc[-1], y.iloc[-1], marker='s', markersize=8, color='red', label='End')
    
    goal_x, goal_y, _ = goal
    ax.plot(goal_x, goal_y, marker='*', markersize=12, color='green', label='Goal')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Trajectory with Obstacles')

    # Enable grid and legend
    ax.grid(True)
    ax.legend()

    # Optionally, set equal aspect ratio for the x and y axes
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
