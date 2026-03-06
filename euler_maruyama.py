import numpy as np

def doublewell(x):
    return 0.25 * x**4 - 0.5 * x**2

def grad_potential(x):
    return x**3 - x  # Gradient of potential

def euler_maruyama_step(x, dt, beta=1.0):
    noise = np.sqrt(2 * dt / beta) * np.random.randn(*x.shape)
    drift = -grad_potential(x)
    return x + drift * dt + noise

def generate_trajectories(n_traj=10, steps=1000, dt=0.01):
    trajs = []
    for _ in range(n_traj):
        x = np.zeros((steps, 1))
        x[0] = np.random.randn()
        for t in range(1, steps):
            x[t] = euler_maruyama_step(x[t-1], dt)
        trajs.append(x)
    return np.array(trajs)


