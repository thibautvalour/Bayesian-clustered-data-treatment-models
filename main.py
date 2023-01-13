import numpy as np
import torch
import scipy.stats as stats
import time 

from init import init
from samplers import sample_sigma, sample_zi_star, sample_beta, super_update

s, w, x, y, z, sigma_matrix, gamma, beta_0, beta_1, g_0, G_0, beta_0_init, B_0 = init()
results = {'beta_0': [beta_0], 'beta_1': [beta_1], 'sigma': [sigma_matrix], 'gamma': [gamma]}

def main_gaussian(w, x, s, z, gamma,  sigma_matrix, beta_0,
                  beta_1, g_0, G_0, beta_0_init, B_0, n_iterations):
    start = time.time()
    for i in range(n_iterations):
        temp_start = time.time()
        sigma_matrix, beta = super_update(w, x, s, z, gamma,  sigma_matrix, beta_0,
                                          beta_1, g_0, G_0, beta_0_init, B_0)
        sigma_matrix = torch.tensor(sigma_matrix)
        beta = torch.tensor(beta)
        gamma, beta_0, beta_1 = beta[:5], beta[5:9], beta[9:]
        results['beta_0'].append(beta_0)
        results['beta_1'].append(beta_1)
        results['sigma'].append(sigma_matrix)
        results['gamma'].append(gamma)
        np.save('results.npy', results)
        if i%10 == 0:
            step_time = round(time.time() - temp_start, 2)
            total_time = round((time.time() - start)/60)
            print(f"Iteration {i} done in {step_time} seconds. Total time: {total_time} minutes.")
    return beta, sigma_matrix

if __name__ == "__main__":
    new_sigma, new_beta = main_gaussian(w, x, s, z, gamma,  sigma_matrix, beta_0,
                                        beta_1, g_0, G_0, beta_0_init, B_0, 21)
