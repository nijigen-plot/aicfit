import numpy as np
# 各分布の対数尤度とパラメータ偏微分値を計算

def cauchy_log_likelihood(x, mu, tau):
    return (5 * np.log(tau)) - (10 * np.log(np.pi)) - (np.log( ((x - mu) ** 2) + tau )).sum()

def cauchy_derivative(x, mu, tau):
    mu_d = 2 * ((x - mu) / (((x - mu) ** 2) + tau)).sum()
    tau_d = (5 / tau) - (1 / ( ((x - mu) ** 2) + tau)).sum()
    return mu_d, tau_d


