# フィッティングに使用する分布を指定
import numpy as np
import pandas as pd
from itertools import product
from aicfit import calc

def quasi_newton_method(sample, first_params, first_log_likelihood, first_derivative_scores, log_calc, derivative_calc, c, df_out=False):
    finish_flag = False
    # 最大リピート回数
    repeat_number = 100
    # パラメータ数
    param_number = len(first_params)
    # ステップ幅
    lambda_sample = np.array(list(product(np.linspace(-1, 1, 201), repeat=param_number)))
    # パラメータ群
    x = np.array(first_params)
    H = np.linalg.inv(np.eye(len(first_params)))
    g = np.array(first_derivative_scores)
    params = [x]
    aics = [-2 * first_log_likelihood + 2 * param_number]
    derivatives = [g]
    # 探索
    for i in range(repeat_number):
        hk = H * g
        derivative_result = np.array([ derivative_calc( sample, *(x - np.dot(lk, hk))) for lk in lambda_sample])
        fx_array = np.array([ np.abs(i).sum() for i in derivative_result])
        min_fx = fx_array.min()
        min_lambda = lambda_sample[np.argmin(fx_array)]
        log_likelihood = log_calc( sample, *(x - np.dot(min_lambda, hk)))
        next_x = x - np.dot(min_lambda, hk)
        delta_x = next_x - x
        x = next_x
        next_g = np.array(derivative_calc( sample, *x ))
        delta_g = next_g - g
        g = next_g
        # DFP calc
        parts1 = H
        parts2 = (np.dot(delta_x, delta_x.T) / np.dot(delta_x.T, delta_g))
        parts3 = (np.dot(H @ delta_g @ delta_g.T, H) / (delta_g.T @ H @ delta_g))
        next_H = parts1 + parts2 - parts3
        H = next_H
        # append
        params.append(x)
        aics.append(-2 * log_likelihood + 2 * param_number)
        derivatives.append(g)
        while min_fx < 0.001:
            print('パラメータ偏微分値が0.001を下回った為探索を終了しました')
            finish_flag = True
            break
        if finish_flag:
            break
    if df_out:
        d = pd.concat([pd.DataFrame(params), pd.DataFrame(aics), pd.DataFrame(derivatives)], axis=1)
        d.columns = c
        return d
    else:
        return params[-1], aics[-1], derivatives[-1]
        

class normal:
    def __init__(self, x, mu, var):
        self.x = x
        self.mu = mu
        self.var = var
        self.log_likelihood = calc.normal_log_likelihood(self.x, self.mu, self.var)
        self.mu0_derivative_score, self.var0_derivative_score = calc.normal_derivative(self.x, self.mu, self.var)
        self.column_name = ['mu', 'var', 'aic', 'mu_derivative', 'var_derivative']
    
    def fit(self, df_out=False):
        return quasi_newton_method(self.x, [self.mu, self.var], self.log_likelihood, [self.mu0_derivative_score, self.var0_derivative_score], calc.normal_log_likelihood, calc.normal_derivative, self.column_name, df_out)

class poisson:
    def __init__(self, x, lambda_):
        self.x = x
        self.lambda_ = lambda_
        self.log_likelihood = calc.poisson_log_likelihood(self.x, self.lambda_)
        self.lambda_0_derivative_score = calc.poisson_derivative(self.x, self.lambda_)
        self.column_name = ['lambda', 'aic', 'lambda_derivative']
    
    def fit(self, df_out=False):
        return quasi_newton_method(self.x, [self.lambda_], self.log_likelihood, [self.lambda_0_derivative_score], calc.poisson_log_likelihood, calc.poisson_derivative, self.column_name, df_out)

class nbinom:
    def __init__(self, x, r, p):
        self.x = x
        self.r = r
        self.p = p
        self.log_likelihood = calc.nbinom_log_likelihood(self.x, self.r, self.p)
        self.r0_derivative_score, self.p0_derivative_score = calc.nbinom_derivative(self.x, self.r, self.p)
        self.column_name = ['r', 'p', 'aic','r_derivative', 'p_derivative']
    
    def fit(self, df_out=False):
        return quasi_newton_method(self.x, [self.r, self.p], self.log_likelihood, [self.r0_derivative_score, self.p0_derivative_score], calc.nbinom_log_likelihood, calc.nbinom_derivative, self.column_name, df_out)


class cauchy:
    def __init__(self, x, mu, tau):
        self.x = x
        self.mu = mu
        self.tau = tau
        self.log_likelihood = calc.cauchy_log_likelihood(self.x, self.mu, self.tau)
        self.mu0_derivative_score, self.tau0_derivative_score = calc.cauchy_derivative(self.x, self.mu, self.tau)
        self.column_name = ['mu', 'tau', 'aic', 'mu_derivative', 'tau_derivative']

    def fit(self, df_out=False):
        return quasi_newton_method(self.x, [self.mu, self.tau], self.log_likelihood, [self.mu0_derivative_score, self.tau0_derivative_score], calc.cauchy_log_likelihood, calc.cauchy_derivative, self.column_name, df_out)
