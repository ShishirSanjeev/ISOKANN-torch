import matplotlib.pyplot as plt
import numpy as np

def visualize(data, memberships, koopman_estimates=None):
    n_traj, n_steps, _ = data.shape
    dominant_states = np.argmax(memberships, axis=2)

    time_axis = np.arange(n_steps)
    time_axis_trim = time_axis[:-1]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for i in range(n_traj):
        axs[0].plot(time_axis, data[i, :, 0], alpha=0.7)
    axs[0].set_ylabel("Position")
    axs[0].set_title("Simulated Trajectories")

    for i in range(n_traj):
        y = dominant_states[i]
        x = time_axis if len(y) == n_steps else time_axis_trim
        axs[1].plot(x, y, alpha=0.7)
    axs[1].set_ylabel("Metastable State")
    axs[1].set_xlabel("Time Step")
    axs[1].set_title("Metastable States from ISOKANN")

    plt.tight_layout()
    plt.show()
