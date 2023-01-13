import numpy as np
import torch
import scipy.stats as stats

from utils import is_positive_definite, flat_sigma_to_sigma_matrix, sigma_matrix_to_flat_sigma
from init import init
from likelyhoods import log_likelyhood_ys, log_g_sigma, q_sigma

s, w, x, y, z, sigma_matrix, gamma, beta_0, beta_1, g_0, G_0, beta_0_init, B_0 = init()

def sample_sigma(old_sigma_matrix, s, w, x, z, beta_0, beta_1, gamma, g_0, G_0):
    old_flat_sigma = sigma_matrix_to_flat_sigma(old_sigma_matrix)
    old_log_g_sigma_likely = log_g_sigma(w, s, x, z, beta_0, beta_1, gamma, old_flat_sigma, g_0, G_0)
    old_q_sigma_likely = q_sigma(w, s, x, z, beta_0, beta_1, gamma, old_sigma_matrix, g_0, G_0, sample=False)

    # generation of new_sigma 
    new_sigma_flat = q_sigma(w, s, x, z, beta_0, beta_1, gamma, old_sigma_matrix,
                             g_0, G_0, sample=True)
    new_sigma_matrix = flat_sigma_to_sigma_matrix(new_sigma_flat)
    if not is_positive_definite(new_sigma_matrix):
        return old_sigma_matrix
    new_log_g_sigma_likely = log_g_sigma(w, s, x, z, beta_0, beta_1, gamma, new_sigma_flat, g_0, G_0)
    new_q_sigma_likely = q_sigma(w, s, x, z, beta_0, beta_1, gamma, new_sigma_matrix, g_0, G_0, sample=False)

    # acceptance ratio
    g_sigma_likely_ratio = torch.exp(new_log_g_sigma_likely - old_log_g_sigma_likely)
    acceptance_ratio = g_sigma_likely_ratio*new_q_sigma_likely/old_q_sigma_likely

    if acceptance_ratio > np.random.uniform():
        return new_sigma_matrix
    else:
        return old_sigma_matrix

def sample_zi_star(w, x, s, z, gamma, beta_0, beta_1, sigma_matrix):
    '''zi_star = si_star, z_i[1 - si]
    '''
    si_star = np.zeros(len(s))
    for i in range(len(s)):
        treatment = s[i]
        if treatment == 0:
            si_star[i] = stats.truncnorm.rvs(a=-np.inf, b=0,
                                             loc=np.dot(w[i], gamma), scale=1)
            z[1][i] = stats.norm.rvs(loc=np.dot(x[i], beta_1),
                                     scale=sigma_matrix[2][2]**2)
        else:
            si_star[i] = stats.truncnorm.rvs(a=0, b=np.inf,
                                             loc=np.dot(w[i], gamma), scale=1)
            z[0][i] = stats.norm.rvs(loc=np.dot(x[i], beta_0),
                                     scale=sigma_matrix[1][1]**2)
    return si_star, z

def sample_beta(w, x, s, z, gamma, beta_0,
                beta_1, sigma_matrix, B_0, beta_0_init, s_star):
    new_z = np.zeros((len(s), 3))
    new_z[:, 0], new_z[:,1], new_z[:, 2] = s_star, z[0], z[1]

    big_X = np.zeros((len(s), 13, 3))
    big_X[:, 0:5, 0], big_X[:, 5:9, 1], big_X[:, 9:13, 2] = w, x, x

    sigma_inv = np.linalg.inv(sigma_matrix)
    B_0_inv = np.linalg.inv(B_0)


    B = B_0_inv + np.sum([np.dot(np.dot(big_X[i], sigma_inv),np.transpose(big_X[i]))
                        for i in range(len(big_X))])
    B = np.linalg.inv(B)
    B = (B + np.transpose(B)) / 2 # Force numerical symetrie

    beta_hat = np.dot(B, np.dot(B_0_inv, beta_0_init)
                        + np.sum([np.dot(np.dot(big_X[i], sigma_inv), new_z[i])
                        for i in range(len(big_X))]))

    new_beta = stats.multivariate_normal.rvs(mean=beta_hat, cov=B)
    return new_beta


def super_update(w, x, s, z, gamma,  sigma_matrix, beta_0, beta_1, g_0, G_0, beta_0_init, B_0):
    new_sigma = sample_sigma(sigma_matrix, s, w, x, z, beta_0, beta_1, gamma, g_0, G_0)
    s_star, z = sample_zi_star(w, x, s, z, gamma, beta_0, beta_1, new_sigma)
    new_beta = sample_beta(w, x, s, z, gamma, beta_0, beta_1, new_sigma, B_0, beta_0_init, s_star)
    return new_sigma, new_beta

new_sigma, new_beta = super_update(w, x, s, z, gamma,  sigma_matrix, beta_0, beta_1, g_0, G_0, beta_0_init, B_0)
