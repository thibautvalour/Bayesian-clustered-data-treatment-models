import numpy as np
from utils import is_positive_definite, flat_sigma_to_sigma_matrix

def init():

    # load the data
    data_dict_try = np.load('../data/data_dict.npy', allow_pickle=True).item()
    s, w, x, y, z = data_dict_try['s'], data_dict_try['w'], data_dict_try['x'], data_dict_try['y'], data_dict_try['z']
    n = 500
    s, w, x, y = s[:n], w[:n], x[:n], y[:n]
    z[0], z[1] = z[0][:n], z[1][:n]

    # parameters
    z_mean = np.array([z[1][i] if s[i] else z[0][i] for i in range(len(s))]).mean()
    beta_0_mean, beta_1_mean = z_mean.mean()/(4*x.mean(axis=0)), z_mean/(4*x.mean(axis=0))
    gamma_mean = 1/(5*np.mean(w, axis=0))
    
    beta_0_init = np.concatenate([beta_0_mean, beta_1_mean, gamma_mean])
    B_0 = np.diag([1] * 4 * 2 + [0]*5)

    # sample from multivariate normal distribution
    beta = np.random.multivariate_normal(beta_0_init, B_0)

    g_0 = np.array([0, 0, z_mean, z_mean])
    G_0 = np.identity(len(g_0))
    
    sigma_flat = np.random.multivariate_normal(g_0, G_0)
    sigma_matrix = flat_sigma_to_sigma_matrix(sigma_flat)
    
    while not is_positive_definite(sigma_matrix):
        sigma_flat = np.random.multivariate_normal(g_0, G_0)
        sigma_matrix = flat_sigma_to_sigma_matrix(sigma_flat)
    
    return s, w, x, y, z, sigma_matrix, beta, g_0, G_0, beta_0_init, B_0
