import numpy as np


TARG_LST = [
    'double_well',
    'gaussian',
    'gaussian_mixture',
    'ginzburg_landau'
]

ALGO_LST = [
    'ULA',
    'MALA',
    'TULA',
    'mTULA'
]


class LangevinSampler:
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
            theta_arr = np.zeros((n_iter, d))

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
            if (n >= n_burnin) and return_arr:
                theta_arr[n - n_burnin] = theta

        return theta if (not return_arr) else theta_arr