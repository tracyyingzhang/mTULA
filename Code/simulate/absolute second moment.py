#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import itertools
import pandas as pd
                                    
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import quad
from scipy.special import gamma
from scipy.stats import wasserstein_distance
import math


# In[2]:


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object

    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


# In[3]:


TARG_LST = [
    'double_well',
    'gaussian',
    'gaussian_mixture'
]

ALGO_LST = [
    'ULA',
    'MALA',
    'TULA',
    'mTULA'
]

class TULALangevinSampler:
    def __init__(self, targ, algo, step=0.001, beta=1, Sigma=None, a=None, alpha=None, lambd=None, tau=None):
        assert targ in TARG_LST
        assert algo in ALGO_LST

        self.targ = targ
        self.algo = algo
        self.beta = beta
        self.step = step
        self.adjust = (algo == 'MALA')

        self.Sigma = Sigma  # for gaussian target
        self.a = a  # for mixed gaussian target

        # ginzburg_landau parameters
        self.alpha = alpha
        self.lambd = lambd
        self.tau = tau


        if targ == 'double_well':
            self.r = 2

        elif targ == 'gaussian' or targ == 'gaussian_mixture':
            self.r = 0

        elif targ == 'ginzburg_landau':
            self.r = 2


    def _potential(self, theta):
        if self.targ == 'double_well':
            return (1 / 4) * np.dot(theta, theta)**2 - (1/2) * np.dot(theta, theta)

        elif self.targ == 'gaussian':
            return (1 / 2) * ((theta.dot(np.linalg.inv(self.Sigma))).dot(theta))

        elif self.targ == 'gaussian_mixture':
            return (1 / 2) * (np.linalg.norm(theta-self.a)**2) - np.log(1 + np.exp(-2*np.dot(theta, self.a)))

        elif self.targ == 'ginzburg_landau':
            theta_mat = theta.reshape([int(np.cbrt(len(theta)))]*3)

            return ((1-self.tau)/2) * ((theta_mat**2).sum()) + \
                   (self.tau * self.lambd/4) * ((theta_mat**4).sum()) + \
                   (self.tau * self.alpha/2) * sum([((np.roll(theta_mat, -1, axis) - theta_mat) ** 2).sum() for axis in range(3)])


    def _gradient(self, theta):
        if self.targ == 'double_well':
            return (np.dot(theta, theta) - 1) * theta

        elif self.targ == 'gaussian':
            return np.linalg.inv(self.Sigma).dot(theta)

        elif self.targ == 'gaussian_mixture':
            return theta - self.a + 2 * self.a / (1 + np.exp(2*np.dot(theta, self.a)))

        elif self.targ == 'ginzburg_landau':
            theta_mat = theta.reshape([int(np.cbrt(len(theta)))]*3)
            grad_mat = (1-self.tau)*theta_mat + \
                       (self.tau * self.lambd)*(theta_mat**3) + \
                       (self.tau * self.alpha) * (
                            6 * theta_mat - \
                            np.roll(theta_mat, -1, axis=0) - \
                            np.roll(theta_mat, -1, axis=1) -
                            np.roll(theta_mat, -1, axis=2) - \
                            np.roll(theta_mat, 1, axis=0) - \
                            np.roll(theta_mat, 1, axis=1) - \
                            np.roll(theta_mat, 1, axis=2)
                        )
            return grad_mat.reshape(-1)


    def _gradient_tamed(self, theta):
        if self.algo == 'ULA' or self.algo == 'MALA':
            return self._gradient(theta)

        elif self.algo == 'TULA':
            grad = self._gradient(theta)
            return grad / (1 + (self.step) * np.linalg.norm(grad))

        elif self.algo == 'mTULA':
            return self._gradient(theta) / ((1 + self.step*(np.dot(theta, theta)**self.r))**0.5)


    def sample(self, theta0, n_iter=10**5, n_burnin=10**4, return_arr=False, runtime=200):
        # if runtime is specified
        if runtime is not None:
            # replace the number of iterations and burn-in samples such that step*n_iter is constant
            n_iter = int(runtime/self.step)
            n_burnin = n_iter

        # flatten array to 1d
        theta = np.ravel(np.array(theta0).reshape(-1))

        # obtain dimension
        d = len(theta)

        # initialise array to store samples after burn-in period
        if return_arr:
            theta_arr = np.zeros(n_iter + n_burnin)

        # run algorithm
        for n in np.arange(n_iter + n_burnin):

            # proposal
            proposal = theta - self.step * self._gradient_tamed(theta) + np.sqrt(
                2 * self.step / self.beta) * np.random.normal(size=d)

            # if metropolis-hastings version is run
            if self.adjust:
                # potential at current iteration and proposal
                U_proposal = self._potential(proposal)
                U_theta = self._potential(theta)

                # (tamed) gradient at current iteration and proposal
                h_proposal = self._gradient_tamed(proposal)
                h_theta = self._gradient_tamed(theta)

                # logarithm of acceptance probability
                log_acceptance_prob = -self.beta * (U_proposal - U_theta) + \
                                      (1 / (4 * self.step)) * (np.linalg.norm(
                    proposal - theta + self.step * h_theta)**2 - np.linalg.norm(
                    theta - proposal + self.step * h_proposal)**2)

                # determine acceptance
                if np.log(np.random.uniform(size=1)) <= log_acceptance_prob:
                    theta = proposal

            # if not, then an unadjusted version is run
            else:
                theta = proposal

            # include samples after burn-in in final output
            if return_arr:
                theta_arr[n] = theta[0]

        return theta if (not return_arr) else theta_arr


# In[4]:


def draw_samples_parallel(sampler, theta0, runtime=200, n_chains=100, n_jobs=-1):
    d = len(np.ravel(np.array(theta0).reshape(-1)))
    sampler.d = d
    
    def _run_single_markov_chain():
        last_iteration_sample = sampler.sample(theta0, return_arr=False, runtime=runtime)
        return pd.DataFrame(
            last_iteration_sample.reshape(1, -1),
            columns=[f'component_{i + 1}' for i in range(len(last_iteration_sample))]
        )
        
    samples_df_lst = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_markov_chain)() for i in tqdm(range(n_chains), desc='Markov Chains')
    )

    return pd.concat(samples_df_lst, ignore_index=True)


# In[71]:


step_sizes = [0.001]
d = 100
theta0 = 10*np.ones(d)
no_chains = 250


# In[72]:


mtula_results_dict = {}
for step in step_sizes:
    print(f"Running Langevin sampler for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    mtula_sampler = TULALangevinSampler(targ='double_well', algo='mTULA', step=step)
    
    # Draw samples in parallel
    mtula_results_df = draw_samples_parallel(mtula_sampler, theta0, n_chains=no_chains)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    mtula_results_dict[step] = mtula_results_df


# In[73]:


# Define the absolute second moment function
def absolute_second_moment(data):
    norms_squared = np.sum(data**2, axis=1)  # Calculate the squared norm of each row
    return np.mean(norms_squared)  # Average the squared norms over the 250 samples

results_mtula = mtula_results_dict[step]
# Compute the absolute second moment
absolute_second_moment_mtula = absolute_second_moment(results_mtula.to_numpy())
# Print the absolute second moment
print(absolute_second_moment_mtula)


# In[58]:


tula_results_dict = {}
for step in step_sizes:
    print(f"Running Langevin sampler for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    tula_sampler = TULALangevinSampler(targ='double_well', algo='TULA', step=step)
    
    # Draw samples in parallel
    tula_results_df = draw_samples_parallel(tula_sampler, theta0, n_chains=no_chains)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    tula_results_dict[step] = tula_results_df

results_tula = tula_results_dict[step]
# Compute the absolute second moment
absolute_second_moment_tula = absolute_second_moment(results_tula.to_numpy())

# Print the absolute second moment
print(absolute_second_moment_tula)


# In[55]:


ula_results_dict = {}
for step in step_sizes:
    print(f"Running Langevin sampler for step size: {step}")
    
    # Create a LangevinSampler instance with the specified step size
    ula_sampler = TULALangevinSampler(targ='double_well', algo='ULA', step=step)
    
    # Draw samples in parallel
    ula_results_df = draw_samples_parallel(ula_sampler, theta0, n_chains=no_chains)
    
    # Store the results DataFrame in the dictionary with the step size as the key
    ula_results_dict[step] = ula_results_df

results_ula = ula_results_dict[step]
# Compute the absolute second moment
absolute_second_moment_ula = absolute_second_moment(results_ula.to_numpy())

# Print the absolute second moment
print(absolute_second_moment_ula)


# In[ ]:




