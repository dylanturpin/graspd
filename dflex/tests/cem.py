import numpy as np
from sklearn.mixture import GaussianMixture

def CEM(
    fitness_function,
    prior_particles,
    alpha=0.1,
    qtile=0.25,
    verbose=True,
    gmm_components=None,
    max_iter=30,
):
    """
    Run CEM with quantile bounding which maximizes fitness function.
    :param fitness_function: a function from [particle x param_value] -> [fitness]
    :param prior_particles: a matrix of [particle x param_value]
    :param alpha: the low pass filter alpha on the update of the quantile and posterior distribution
    :param qtile: the quantile of fit samples to include
    :param verbose: whether to print out debugging information
    :param gmm_components: number of kernels in the GMM
    :param max_iter: maximum number of iterations for fitting the GMM at each CEM step
    """
    gmm_verbosity = 2 if verbose else 0
    model = GaussianMixture(
        n_components=gmm_components,
        n_init=5,
        warm_start=True,
        verbose=gmm_verbosity,
        max_iter=max_iter,
    )
    cur_bound = float("-inf")
    mean = None
    var = None
    while True:
        # evaluate fitness
        fitnesses = fitness_function(prior_particles)
        elite_idxs = fitnesses >= cur_bound
        elite_samples = prior_particles[elite_idxs, :]
        elite_fitnesses = fitnesses[elite_idxs]
        if elite_fitnesses != np.array([]):
            quantile_fitnesses = np.quantile(elite_fitnesses, [1 - qtile])[0]
            if cur_bound != float("-inf"):
                cur_bound = alpha * quantile_fitnesses + \
                    (1 - alpha) * cur_bound
            else:
                cur_bound = quantile_fitnesses
            if gmm_components is None:
                cur_mean = np.mean(elite_samples, axis=0)
                cur_var = np.var(elite_samples, axis=0)
                if mean is not None:
                    mean = alpha * cur_mean + (1 - alpha) * mean
                    var = alpha * cur_var + (1 - alpha) * var
                else:
                    mean = cur_mean
                    var = cur_var
                if verbose:
                    print(f"Posterior N({mean},{var}), Bound: {cur_bound}")
            else:
                model.fit(elite_samples)
        if gmm_components is None:
            prior_particles = np.random.multivariate_normal(
                mean, np.diag(var), size=prior_particles.shape[0]
            )
        else:
            # ensure the sum of weights never exceeds 1
            model.weights_ = model.weights_ / np.sum(model.weights_)
            prior_particles = model.sample(prior_particles.shape[0])[0]
            prior_particles[:,3:7] /= np.linalg.norm(prior_particles[:,3:7],axis=1,keepdims=True)
        yield prior_particles