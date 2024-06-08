import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed
from progress_bar import tqdm_joblib

from sampler import LangevinSampler


"""
Draws samples from single sampler by running markov chains sequentially
"""
def draw_samples(sampler, theta0, n_chains=100):
    # dimension of sample
    d = len(np.ravel(np.array(theta0).reshape(-1)))

    # initialise arrays to store moments
    moment_1st = np.zeros((n_chains, d))
    moment_2nd = np.zeros((n_chains, d))

    # initialise array to store samples
    samples = np.zeros((n_chains, d))

    for i in range(n_chains):
        # run a single markov chain
        theta_arr = sampler.sample(theta0, return_arr=True)

        # update moments
        moment_1st[i] = theta_arr.mean(axis=0)
        moment_2nd[i] = (theta_arr ** 2).mean(axis=0)

        # update sample
        samples[i] = theta_arr[-1]

    return samples, moment_1st, moment_2nd


"""
Draws samples from single sampler by running markov chains in parallel. Example of usage:

d = 100
sampler = LangevinSampler(targ='double_well', algo='mTULA', step = 0.0001)
theta0 = np.zeros(d)
results_df = draw_samples_parallel(sampler, theta0, n_chains=250)
"""
def draw_samples_parallel(sampler, theta0, runtime=200, n_chains=100, n_jobs=-1):
    # dimension of sample
    d = len(np.ravel(np.array(theta0).reshape(-1)))

    # running a single markov chain
    def _run_single_markov_chain():
        return pd.DataFrame(
            [sampler.sample(theta0, runtime=runtime)],
            columns=[f'component_{i + 1}' for i in range(d)]
        )

    # run markov chains in parallel
    with tqdm_joblib(tqdm(desc='Markov Chains', total=n_chains)) as progress_bar:
        samples_df_lst = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_markov_chain)() for i in range(n_chains)
        )

    return pd.concat(samples_df_lst, ignore_index=True)


"""
Draws samples from multiple samplers specified by a given parameter grid. Example of usage:

d = 100
param_grid = {
    'targ': ['double_well', 'gaussian', 'gaussian_mixture'],
    'algo': ['ULA', 'MALA', 'TULA', 'mTULA'],
    'step': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    'theta0': [np.zeros(d)],
    'n_chains': 250,
    'Sigma': [np.eye(d)],
    'a': [2*np.ones(d)/(np.sqrt(d))]
}
results_df = convergence_results(param_grid)
"""
def convergence_results(param_grid, n_jobs=-1):
    # fill in missing parameters
    for key in ['Sigma', 'a', 'alpha', 'lambd', 'tau']:
        if param_grid.get(key) == None:
            param_grid[key] = []

    # expand parameter grid
    param_grid_expanded = list(itertools.product(
        [v for v in param_grid['targ'] if not v in ['gaussian', 'gaussian_mixture', 'ginzburg_landau']],
        param_grid['algo'],
        param_grid['step'],
        [None],
        [None],
        [None],
        [None],
        [None]
    )) + list(itertools.product(
        [v for v in param_grid['targ'] if v == 'gaussian'],
        param_grid['algo'],
        param_grid['step'],
        param_grid['Sigma'],
        [None],
        [None],
        [None],
        [None]
    )) + list(itertools.product(
        [v for v in param_grid['targ'] if v == 'gaussian_mixture'],
        param_grid['algo'],
        param_grid['step'],
        [None],
        param_grid['a'],
        [None],
        [None],
        [None]
    )) + list(itertools.product(
        [v for v in param_grid['targ'] if v == 'ginzburg_landau'],
        param_grid['algo'],
        param_grid['step'],
        [None],
        [None],
        param_grid['alpha'],
        param_grid['lambd'],
        param_grid['tau']
    ))

    # function that runs markov chain for a single configuration
    def _convergence_results_single_config(targ, algo, step, Sigma, a, alpha, lambd, tau):
        # intialise empty list to store results
        results_df_lst = []

        # initialise sampler
        sampler = LangevinSampler(targ, algo, step=step, Sigma=Sigma, a=a, alpha=alpha, lambd=lambd, tau=tau)

        # iterate over initial starting points
        for theta0 in param_grid['theta0']:
            # number of independent Markov chains
            n_chains = param_grid['n_chains']

            # dimension of samples
            d = len(np.ravel(np.array(theta0).reshape(-1)))

            # draw samples and compute moment estimates
            samples, moment_1st, moment_2nd = draw_samples(sampler, theta0, n_chains=n_chains)

            # store results in dataframe
            results_df = pd.DataFrame({
                'target': [targ] * n_chains,
                'algo': [algo] * n_chains,
                'step': [step] * n_chains
            })

            results_df.loc[:, 'theta0'] = [theta0] * n_chains
            results_df.loc[:, 'Sigma'] = [Sigma] * n_chains
            results_df.loc[:, 'a'] = [a] * n_chains
            results_df.loc[:, 'alpha'] = [alpha] * n_chains
            results_df.loc[:, 'lambd'] = [lambd] * n_chains
            results_df.loc[:, 'tau'] = [tau] * n_chains
            results_df.loc[:, [f'moment_1st_{i + 1}' for i in range(d)]] = moment_1st
            results_df.loc[:, [f'moment_2nd_{i + 1}' for i in range(d)]] = moment_2nd
            results_df.loc[:, [f'component_{i + 1}' for i in range(d)]] = samples
            results_df_lst.append(results_df)

        return pd.concat(results_df_lst)

    # run parameter configurations in parallel
    with tqdm_joblib(tqdm(desc='Simulation', total=len(param_grid_expanded))) as progress_bar:
        results_df_lst = Parallel(n_jobs=n_jobs)(
            delayed(_convergence_results_single_config)(
                targ, algo, step, Sigma, a, alpha, lambd, tau
            ) for (targ, algo, step, Sigma, a, alpha, lambd, tau) in param_grid_expanded
        )

    return pd.concat(results_df_lst)