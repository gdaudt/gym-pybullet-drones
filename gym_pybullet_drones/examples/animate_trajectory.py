#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # needed for 3d plotting

m = 0.27  # kg
g = 9.81  # m/s² (gravity)
max_thrust = 0.6  # m/s² (max additional thrust)
F_max = m * g + m * max_thrust  # Maximum force

dim = 3

# Time parameters for trajectory generation (for simulated trajectories)
t_end = 0.7  # seconds
num_samples = 100  # Number of time steps for simulated trajectory
t = np.linspace(0, t_end, num_samples)

# Fourier series parameters
N = 5  # Number of Fourier terms
omega = 2 * np.pi / t_end  # Base frequency (one full oscillation per time window)

y = 0
z = 1
obstacles = [
    [1, 1, z], [2, 1, z], [1, -2, z], [2, -2, z], [-1, 0, z], [-1, 0.5, z],
    [-1, -0.5, z], [-1, 1, z], [-1, -1, z], [0, 1, z], [-1, -1.5, z], [0, -2, z],
    [5, 0, z], [5, 0.5, z], [5, -0.5, z], [5, 1, z], [5, -1, z], [5, -1.5, z],
    [3, -2, z], [4, -2, z], [3, 1, z], [4, 1, z]
]
inner_walls = [[1, y, z], [2, y, z]]
obstacles = inner_walls

# Obstacle dimensions
obs_size_x = 1
obs_size_y = 0.5
obs_size_z = 2

def generate_trajectories(x0, y0, z0, vx0, vy0, vz0):
    """
    Generate simulated trajectories based on initial conditions.
    Returns an array of shape (num_trajs, num_samples, 3).
    """
    num_trajs = 50  # number of simulated trajectories
    trajs = []
    for _ in range(num_trajs):
        # Generate random Fourier coefficients for x and y
        A_x = np.random.uniform(-F_max, F_max, N)
        B_x = np.random.uniform(-F_max, F_max, N)
        A_y = np.random.uniform(-F_max, F_max, N)
        B_y = np.random.uniform(-F_max, F_max, N)

        # Compute acceleration as Fourier series for x and y (divide by mass to get acceleration)
        a_x = np.sum([A_x[n] * np.cos((n+1) * omega * t) + B_x[n] * np.sin((n+1) * omega * t)
                      for n in range(N)], axis=0) / m
        a_y = np.sum([A_y[n] * np.cos((n+1) * omega * t) + B_y[n] * np.sin((n+1) * omega * t)
                      for n in range(N)], axis=0) / m

        # For 3D, also compute for z
        if dim == 3:
            A_z = np.random.uniform(-F_max, F_max, N)
            B_z = np.random.uniform(-F_max, F_max, N)
            a_z = np.sum([A_z[n] * np.cos((n+1) * omega * t) + B_z[n] * np.sin((n+1) * omega * t)
                          for n in range(N)], axis=0) / m

        # Integration time-step
        dt = t_end / num_samples

        # Integrate acceleration to get velocity (using scalar initial conditions)
        v_x = vx0 + np.cumsum(a_x) * dt
        v_y = vy0 + np.cumsum(a_y) * dt
        if dim == 3:
            v_z = vz0 + np.cumsum(a_z) * dt

        # Integrate velocity to get position
        x = x0 + np.cumsum(v_x) * dt
        y = y0 + np.cumsum(v_y) * dt
        if dim == 3:
            z = z0 + np.cumsum(v_z) * dt        

        trajs.append(np.column_stack([x, y, z]))
    return np.array(trajs)

def main():
    parser = argparse.ArgumentParser(
        description='Animate a 3D trajectory with obstacles and simulated trajectories with interpolation, then save as MP4.')
    parser.add_argument('csv_file', type=str,
                        help='Path to CSV file (columns: time, x, y, z, vx, vy, vz)')
    parser.add_argument('--output', type=str, default='animation.mp4',
                        help='Output MP4 filename (default: animation.mp4)')
    args = parser.parse_args()

    # Load CSV data
    df = pd.read_csv(args.csv_file)
    times = df['time'].values
    xs = df['x'].values
    ys = df['y'].values
    zs = df['z'].values
    vxs = df['vx'].values
    vys = df['vy'].values
    vzs = df['vz'].values

    # Setup figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    margin = 1
    ax.set_xlim([np.min(xs)-margin, np.max(xs)+margin])
    ax.set_ylim([np.min(ys)-margin, np.max(ys)+margin])
    ax.set_zlim([np.min(zs)-margin, np.max(zs)+margin])
    
    # Render obstacles
    for obs in obstacles:
        obs_x, obs_y, obs_z = obs
        ax.bar3d(obs_x - obs_size_x / 2, obs_y - obs_size_y / 2, 0,  
                 obs_size_x, obs_size_y, obs_size_z, color='gray', alpha=0.2)

    # Precompute simulated trajectories for each CSV entry
    print("Precomputing simulated trajectories...")
    precomputed_trajs = {}
    for i in range(len(xs)):
        cur_x, cur_y, cur_z = xs[i], ys[i], zs[i]
        cur_vx, cur_vy, cur_vz = vxs[i], vys[i], vzs[i]
        precomputed_trajs[i] = generate_trajectories(cur_x, cur_y, cur_z, cur_vx, cur_vy, cur_vz)
    print("Precomputation complete!")

    # To interpolate smoothly between time steps, choose an interpolation factor
    interp_factor = 10  # number of extra frames between each CSV step

    # Prepare data for the main trajectory (we'll also interpolate the main trajectory)
    main_traj = np.column_stack([xs, ys, zs])
    num_main_frames = (len(xs)-1) * interp_factor + 1
    interp_indices = np.linspace(0, len(xs)-1, num_main_frames)
    interp_traj = np.empty((num_main_frames, 3))
    # Linear interpolation for main trajectory (for each coordinate)
    for dim_idx in range(3):
        interp_traj[:, dim_idx] = np.interp(interp_indices, np.arange(len(xs)), main_traj[:, dim_idx])

    # Initialize plot elements
    trajectory_line, = ax.plot([], [], [], 'b-', lw=2, label='Trajectory')
    point_scatter = None
    simulated_lines = []  # list to store simulated trajectory line handles
    main_traj_x, main_traj_y, main_traj_z = [], [], []

    def update(frame):
        nonlocal point_scatter, simulated_lines

        # Determine which CSV index we are between and the interpolation factor
        # interp_index will be between 0 and len(xs)-1
        interp_index = interp_indices[frame]
        lower_idx = int(np.floor(interp_index))
        upper_idx = int(np.ceil(interp_index))
        alpha = interp_index - lower_idx  # interpolation parameter
        # Compute new viewing angles (for example, rotate 1 degree per frame)
        new_elev = 30  # fixed elevation (or change over time)
        new_azim = frame*0.2 % 360  # continuously rotating azimuth
        ax.view_init(elev=new_elev, azim=new_azim)

        # Interpolate main trajectory position
        cur_x = (1 - alpha) * xs[lower_idx] + alpha * xs[upper_idx] if upper_idx < len(xs) else xs[lower_idx]
        cur_y = (1 - alpha) * ys[lower_idx] + alpha * ys[upper_idx] if upper_idx < len(ys) else ys[lower_idx]
        cur_z = (1 - alpha) * zs[lower_idx] + alpha * zs[upper_idx] if upper_idx < len(zs) else zs[lower_idx]

        # Append current interpolated position to main trajectory for plotting
        main_traj_x.append(cur_x)
        main_traj_y.append(cur_y)
        main_traj_z.append(cur_z)
        trajectory_line.set_data(main_traj_x, main_traj_y)
        trajectory_line.set_3d_properties(main_traj_z)

        # Update the point mass marker
        if point_scatter is not None:
            point_scatter.remove()
        point_scatter = ax.scatter([cur_x], [cur_y], [cur_z], color='red', s=100)

        # Clear previous simulated trajectory lines
        for line in simulated_lines:
            line.remove()
        simulated_lines.clear()

        # For the simulated trajectories, we can also interpolate between precomputed ones.
        # Determine the two sets of precomputed trajectories (for lower_idx and upper_idx)
        traj_set_lower = precomputed_trajs[lower_idx]
        traj_set_upper = precomputed_trajs[upper_idx] if upper_idx < len(xs) else traj_set_lower

        # For each simulated trajectory (assuming same ordering across precomputed sets)
        num_trajs = traj_set_lower.shape[0]
        # For each simulated trajectory, interpolate all time steps of the simulated path.
        for i in range(num_trajs):
            traj_lower = traj_set_lower[i]  # shape: (num_samples, 3)
            traj_upper = traj_set_upper[i]  # shape: (num_samples, 3)
            # Linear interpolation between the two trajectories
            traj_interp = (1 - alpha) * traj_lower + alpha * traj_upper
            line, = ax.plot(traj_interp[:, 0], traj_interp[:, 1], traj_interp[:, 2],
                            lw=1, alpha=0.7)
            simulated_lines.append(line)

        ax.set_title(f"Time: {np.interp(interp_index, np.arange(len(xs)), times):.2f}")
        return [trajectory_line, point_scatter] + simulated_lines

    # Total number of frames after interpolation
    total_frames = num_main_frames

    anim = FuncAnimation(fig, update, frames=total_frames, interval=50, blit=False)
    
    writer = FFMpegWriter(fps=30, metadata=dict(artist='Your Name'), bitrate=4000)
    anim.save(args.output, writer=writer)
    print(f"Animation saved to {args.output}")

    # Show legend (only once)
    ax.legend()

    # Display the animation
    plt.show()

if __name__ == '__main__':
    main()
