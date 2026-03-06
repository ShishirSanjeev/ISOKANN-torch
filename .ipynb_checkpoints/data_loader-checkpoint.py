import numpy as np
import os

def save_trajectories(trajs, save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    for i, traj in enumerate(trajs):
        np.save(os.path.join(save_dir, f"traj{i}.npy"), traj)

def load_trajectories(file_list):
    trajs = [np.load(f) for f in file_list]
    return np.array(trajs)
