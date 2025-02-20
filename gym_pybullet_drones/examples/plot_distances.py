import argparse
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

def split_episodes(file_path):
    """
    Splits the CSV file into episodes based on repeated header lines.
    Assumes that the header line is exactly:
    "time,x,y,z,vx,vy,vz,mean_dist,avg_dist,lowest_dist"
    """
    header_line = "time,x,y,z,vx,vy,vz,mean_dist,avg_dist,lowest_dist"
    episodes = []
    current_episode_lines = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # If we encounter a header line and current_episode_lines is not empty,
            # it's the start of a new episode.
            if line.strip() == header_line and current_episode_lines:
                episodes.append("".join(current_episode_lines))
                current_episode_lines = []
            current_episode_lines.append(line)
    # Append the last episode
    if current_episode_lines:
        episodes.append("".join(current_episode_lines))
    
    return episodes

def plot_distance_metric(episodes_data, metric_name, cmap):
    """
    Plots the given metric (e.g. mean_dist) vs. time for all episodes
    in a separate figure. The time axis is normalized so that every episode starts at zero.
    """
    plt.figure(figsize=(10, 6))
    for idx, ep_data in enumerate(episodes_data):
        df = pd.read_csv(StringIO(ep_data))
        # Normalize time so that each episode starts at time zero.
        df['time_norm'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time_norm'], df[metric_name], color=cmap(idx), 
                 label=f'Episode {idx+1}')
    plt.xlabel('Time (normalized)')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Episode')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot distance metrics from a CSV file containing multiple episodes."
    )
    parser.add_argument("--filename", type=str, help="Path to the CSV file")
    args = parser.parse_args()
    
    episodes_data = split_episodes(args.filename)
    num_episodes = len(episodes_data)
    
    # Create a colormap with as many distinct colors as there are episodes.
    cmap = plt.get_cmap('tab10', num_episodes)
    
    # Plot each metric in a separate figure.
    for metric in ['mean_dist', 'avg_dist', 'lowest_dist']:
        plot_distance_metric(episodes_data, metric, cmap)

if __name__ == '__main__':
    main()
