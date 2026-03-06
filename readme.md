# ISOKANN - Torch Implementation 

## Theory

ISOKANN (Invariant Subspace via Koopman Approximation and Neural Networks) is a method for identifying metastable states in stochastic dynamical systems. The core idea is to learn a set of membership functions chi(x) that approximate the slow eigenfunctions of the Koopman operator associated with the system's dynamics. Please see 

The Koopman operator K acts on observables f of the state space by pushing them forward in time: $(Kf)(x) = E[f(x_t+dt) | x_t = x]$. Its leading eigenfunctions capture the slowest-relaxing modes of the system, which correspond to transitions between metastable states. ISOKANN parameterises these eigenfunctions with a neural network and trains it via a self-consistent fixed-point iteration: at each step, the network at time t is regressed onto the network's own output at time t+dt (held fixed as a target), with a shift-scale normalisation applied to prevent collapse.

The system used here is a one-dimensional overdamped Langevin process on a double-well potential V(x) = 0.25x^4 - 0.5x^2:

$$dx_t = -\nabla V(x_t)\, dt + \sqrt{2\beta^{-1}}\, dW_t$$

where beta is the inverse temperature and W_t is a Wiener process. This system has two metastable states near the minima of the potential at x = -1 and x = +1, with rare transitions between them over the energy barrier at x = 0.

---

## Files

### euler_maruyama.py
Defines the double-well potential and its gradient, a single Euler-Maruyama step, and `generate_trajectories`, which runs multiple independent trajectories from random initial conditions.

### data_loader.py
`save_trajectories` writes each trajectory to a `.npy` file. `load_trajectories` reads a list of such files back into a single numpy array of shape `(n_traj, steps, dim)`.

### isokann_model.py
Contains three things:

- `MLP`: a two-layer network with ReLU activation and a Softmax output, so each output dimension can be interpreted as a membership probability summing to one.
- `ISOKANN`: wraps the MLP with a training loop that implements the self-consistent ISOKANN update, and a `predict` method that handles both 2D and 3D input arrays.
- `shift_scale`: centres and normalises the membership output column-wise during training to prevent trivial solutions.

### visualization.py
`visualize` produces a two-panel plot: the raw trajectories on top, and the dominant metastable state (argmax over membership functions) for each trajectory on the bottom.

### main.py
Runs the full pipeline: generate trajectories, save and reload them, train the ISOKANN model, predict memberships, evaluate the membership functions at consecutive time pairs to approximate Koopman action, and visualize the result.


## References 
- [ISOKANN: Invariant subspaces of Koopman operators learned by a neural network](https://pubs.aip.org/aip/jcp/article-abstract/153/11/114109/199583/ISOKANN-Invariant-subspaces-of-Koopman-operators?redirectedFrom=fulltext)
- [ISOKANN - Julia Implementation](https://github.com/axsk/ISOKANN.jl)

