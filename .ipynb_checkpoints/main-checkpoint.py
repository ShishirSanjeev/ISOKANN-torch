from euler_maruyama import generate_trajectories
from data_loader import load_trajectories, save_trajectories
from isokann_model import ISOKANN
from koopman_utils import estimate_koopman_monte_carlo
from visualization import visualize

def main():
    # === Step 1: Generate Trajectories ===
    n_traj = 5
    steps = 1000
    dt = 0.01
    trajectories = generate_trajectories(n_traj, steps, dt)

    # Save and load data
    save_trajectories(trajectories, "data")
    file_list = [f"data/traj{i}.npy" for i in range(n_traj)]
    data = load_trajectories(file_list)  # shape (n_traj, steps, dim)

    # === Step 2: Train ISOKANN ===
    input_dim = data.shape[2]
    model = ISOKANN(input_dim=input_dim, hidden_dim=64, output_dim=3)
    model.train(data, n_epochs=500)

    # === Step 3: Predict membership functions ===
    memberships = model.predict(data)  # shape (n_traj*steps, output_dim) or (n_traj, steps, output_dim)

    # === Step 4: Monte Carlo estimation of Koopman operator action ===
    # Prepare shifted data arrays for Koopman estimation
    x_t = data[:, :-1, :].reshape(-1, input_dim)
    x_tp = data[:, 1:, :].reshape(-1, input_dim)

    _, koopman_estimates = estimate_koopman_monte_carlo(model, x_t, x_tp)  # returns membership(x_t), membership(x_tp)

    # === Step 5: Visualize trajectories and metastable states with Koopman residuals ===
    visualize(data, memberships, koopman_estimates=koopman_estimates)

if __name__ == "__main__":
    main()