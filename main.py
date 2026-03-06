from euler_maruyama import generate_trajectories
from data_loader import load_trajectories, save_trajectories
from isokann_model import ISOKANN
from visualization import visualize

def main():
    n_traj = 5
    steps = 1000
    dt = 0.01
    trajectories = generate_trajectories(n_traj, steps, dt)

    save_trajectories(trajectories, "data")
    file_list = [f"data/traj{i}.npy" for i in range(n_traj)]
    data = load_trajectories(file_list)

    input_dim = data.shape[2]
    model = ISOKANN(input_dim=input_dim, hidden_dim=64, output_dim=3)
    model.train(data, n_epochs=500)

    memberships = model.predict(data)

    x_t = data[:, :-1, :].reshape(-1, input_dim)
    x_tp = data[:, 1:, :].reshape(-1, input_dim)
    chi_xt = model.predict(x_t)
    chi_xtp = model.predict(x_tp)

    visualize(data, memberships, koopman_estimates=chi_xtp)

if __name__ == "__main__":
    main()