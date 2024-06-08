import numpy as np

from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import gamma


class DistributionInfo:
    def __init__(self, name, d, Sigma=None, a=None):
        self.name = name
        self.d = d
        self.Sigma = Sigma
        self.a = a

        # auxiliary constant for double well distribution
        u = lambda r: np.exp(-(1 / 4) * (r ** 2) + (1 / 2) * r)
        double_well_integrand = lambda r: (r ** (self.d / 2 - 1)) * u(r)
        self.double_well_aux = quad(double_well_integrand, 0, np.inf)[0]


    def marginal_density(self, x, component=1):
        if self.name == 'double_well':
            u_tilde = lambda r: np.exp(-(1 / 4) * (r + x ** 2) ** 2 + (1 / 2) * (r + x ** 2))
            integrand = lambda r: (r ** ((self.d - 3) / 2)) * u_tilde(r) / (
                        self.double_well_aux * gamma((self.d - 1) / 2) * np.sqrt(np.pi) / gamma(self.d / 2))

            return quad(integrand, 0, np.inf)[0]

        elif self.name == 'gaussian':
            return norm.pdf(x, loc=0, scale=np.sqrt(self.Sigma[component - 1, component - 1]))

        elif self.name == 'gaussian_mixture':
            return (1 / (2 * np.sqrt(2 * np.pi))) * (np.exp(-((x - self.a[component - 1]) ** 2) / 2) + np.exp(
                -((x + self.a[component - 1]) ** 2) / 2))


    def moment_1st(self, component=1):
        if self.name in ['double_well', 'gaussian', 'gaussian_mixture']:
            return 0


    def moment_2nd(self, component=1):
        if self.name == 'double_well':
            u = lambda r: np.exp(-(1 / 4) * (r ** 2) + (1 / 2) * r)
            integrand = lambda r: (r ** (self.d / 2)) * u(r) / (self.double_well_aux * self.d)

            return quad(integrand, 0, np.inf)[0]

        elif self.name == 'gaussian':
            return self.Sigma[component - 1, component - 1] ** 2

        elif self.name == 'gaussian_mixture':
            return 1