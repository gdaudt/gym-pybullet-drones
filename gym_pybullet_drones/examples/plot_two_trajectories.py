#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    # Set up argument parser to get the CSV file names from the command line
    parser = argparse.ArgumentParser(
        description='Plot 2D trajectories with obstacles from two CSV files.')
    parser.add_argument('csv_file', type=str,
                        help='Path to the first CSV file (columns: time, x, y)')
    parser.add_argument('csv_file2', type=str,
                        help='Path to the second CSV file (columns: time, x, y)')
    args = parser.parse_args()

    # ----- Configuration -----
    # Fixed dimensions for obstacles
    OBSTACLE_WIDTH = 1    # fixed width for all obstacles
    OBSTACLE_HEIGHT = 0.5  # fixed height for all obstacles
    z = 1.05
    # Define obstacles as a list of center positions: (center_x, center_y, z)
    obstacles = [
        [1, 0, z], [2, 0, z], [1, 1, z], [2, 1, z], [1, -2, z], [2, -2, z],
        [-1, 0, z], [-1, 0.5, z], [-1, -0.5, z], [-1, 1, z], [-1, -1, z],
        [0, 1, z], [-1, -1.5, z], [0, -2, z],
        [5, 0, z], [5, 0.5, z], [5, -0.5, z], [5, 1, z], [5, -1, z],
        [5, -1.5, z], [3, -2, z], [4, -2, z], [3, 1, z], [4, 1, z]
    ]

    # ----- Read CSV Data -----
    # Each CSV file is expected to have columns: time, x, y
    df1 = pd.read_csv(args.csv_file)
    df2 = pd.read_csv(args.csv_file2)

    # Extract the trajectory coordinates for the first CSV file
    x1 = df1['x']
    y1 = df1['y']

    # Extract the trajectory coordinates for the second CSV file
    x2 = df2['x']
    y2 = df2['y']

    # ----- Plotting -----
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each obstacle as a rectangle using fixed width and height.
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

    # Plot the first trajectory
    ax.plot(x1, y1, label='Trajectory 1', color='blue', linewidth=2)
    ax.plot(x1.iloc[0], y1.iloc[0], marker='o', markersize=8, color='green', label='Start 1')
    ax.plot(x1.iloc[-1], y1.iloc[-1], marker='s', markersize=8, color='red', label='End 1')

    # Plot the second trajectory
    ax.plot(x2, y2, label='Trajectory 2', color='orange', linewidth=2)
    ax.plot(x2.iloc[0], y2.iloc[0], marker='o', markersize=8, color='black', label='Start 2')
    ax.plot(x2.iloc[-1], y2.iloc[-1], marker='s', markersize=8, color='black', label='End 2')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Trajectories with Obstacles')

    # Enable grid, legend, and equal aspect ratio
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    # Display the plot
    plt.show()

if __name__ == '__main__':
    main()
