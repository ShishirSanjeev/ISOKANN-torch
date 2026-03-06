# koopman_utils.py
import numpy as np

def estimate_koopman_monte_carlo(model, x_t, x_tp):
    """
    Estimate the Koopman operator action via Monte Carlo sampling.

    Parameters:
    - model: ISOKANN model with a predict method that takes numpy arrays and returns numpy arrays.
    - x_t: numpy array of shape (num_traj, steps, features) at time t
    - x_tp: numpy array of shape (num_traj, steps, features) at time t + dt

    Returns:
    - chi_xt: membership function evaluated at x_t (numpy array)
    - chi_xtp: membership function evaluated at x_tp (numpy array)
    """
    # Use the model's predict method directly on numpy arrays
    chi_xt = model.predict(x_t)   # shape (num_samples, n_states)
    chi_xtp = model.predict(x_tp)

    # Here you can add further Monte Carlo estimation if needed
    # For now, just returning memberships at x_t and x_tp

    return chi_xt, chi_xtp

