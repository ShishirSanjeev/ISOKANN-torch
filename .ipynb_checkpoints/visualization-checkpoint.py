import matplotlib.pyplot as plt
import numpy as np

def visualize(data, memberships, koopman_estimates=None):
    """
    Visualize the simulated trajectories and the metastable states.

    Parameters:
        data: numpy array, shape (n_traj, n_steps, dim)
        memberships: numpy array, shape (n_traj, n_steps, n_states)
        koopman_estimates: optional, numpy array, same shape as memberships
    """
    n_traj, n_steps, dim = data.shape
    n_states = memberships.shape[2]

    # Compute dominant metastable state per trajectory point
    dominant_states = np.argmax(memberships, axis=2)  # shape (n_traj, n_steps)

    time_axis = np.arange(n_steps)
    time_axis_trim = time_axis[:-1]  # For states if they have length n_steps-1

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top plot: Trajectories (plot first dimension as example)
    for i in range(n_traj):
        axs[0].plot(time_axis, data[i, :, 0], alpha=0.7)
    axs[0].set_ylabel("Trajectory (dim 0)")
    axs[0].set_title("Simulated Trajectories")

    # Bottom plot: Metastable states
    for i in range(n_traj):
        # If dominant_states length matches n_steps, plot full
        # Else trim x-axis by 1 to match length
        y = dominant_states[i]
        if len(y) == n_steps:
            axs[1].plot(time_axis, y, alpha=0.7)
        else:
            axs[1].plot(time_axis_trim, y, alpha=0.7)
    axs[1].set_ylabel("Metastable State")
    axs[1].set_xlabel("Time Step")
    axs[1].set_title("Metastable States from ISOKANN")

    plt.tight_layout()
    plt.show()
