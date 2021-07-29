import numpy as np
from scipy.special import gammaln

# 各分布の対数尤度とパラメータ偏微分値を計算

# normal distribution
def normal_log_likelihood(x, mu, var):
    return ((-(1/2) * np.log(2 * np.pi * var)) - ((x - mu)**2 / (2 * var))).sum()

def normal_derivative(x, mu, var):
    mu_d = ((x - mu) / np.sqrt(var)).sum()
    var_d = ((-1 / (2 * var)) + (((x - mu)**2) / (2 * (var**2) ))).sum()
    return mu_d, var_d

# poisson distribution
def poisson_log_likelihood(x, lambda_):
    return (-lambda_ + ( x * np.log(lambda_)) - gammaln(x + 1)).sum()

def poisson_derivative(x, lambda_):
    return (1 + (x / lambda_))

# cauchy distribution
def cauchy_log_likelihood(x, mu, tau):
    # return (5 * np.log(tau)) - (10 * np.log(np.pi)) - (np.log( ((x - mu) ** 2) + tau )).sum()
    return (-np.log(np.pi) + (np.log(tau) / 2) - (np.log( ((x - mu) ** 2) + tau ))).sum()

def cauchy_derivative(x, mu, tau):
    mu_d = ((2*x - 2*mu) / (((x - mu)**2 ) + tau)).sum()
    tau_d = ((1 / 2*tau) - ( 1 / (((x - mu)**2) + tau))).sum()
    return mu_d, tau_d
