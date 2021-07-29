import numpy as np
from scipy.special import gammaln, digamma, xlogy

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

# negative binomial distribution (r = number of failures, p = success probability)
def nbinom_log_likelihood(x, r, p):
    return np.log(gammaln(x + r) / (gammaln(x + 1) * gammaln(r))) + xlogy(x, p) + xlogy(r, (1 - p))

def nbinom_derivative(x, r, p):
    r_d = (digamma(x + r) * digamma(1 / (x + 1)) * digamma(1 / r)) + np.log(1 - p)
    p_d = (x / p) - (r / (1 - p))

# cauchy distribution
def cauchy_log_likelihood(x, mu, tau):
    return (-np.log(np.pi) + (np.log(tau) / 2) - (np.log( ((x - mu) ** 2) + tau ))).sum()

def cauchy_derivative(x, mu, tau):
    mu_d = ((2*x - 2*mu) / (((x - mu)**2 ) + tau)).sum()
    tau_d = ((1 / 2*tau) - ( 1 / (((x - mu)**2) + tau))).sum()
    return mu_d, tau_d
