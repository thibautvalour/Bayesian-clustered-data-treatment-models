import numpy as np
from utils import is_positive_definite, flat_sigma_to_sigma_matrix

def init():

    k_0, k_1, p = 4, 4, 5
    k = k_0 + k_1 + p
    
    beta_0_init = np.zeros(k)
    B_0 = np.identity(k)
    # sample from multivariate normal distribution
    beta = np.random.multivariate_normal(beta_0_init, B_0)

    g_0 = np.array([1, 1, 1, 1])
    G_0 = np.identity(g_0.shape[0])
    
    sigma_flat = np.random.multivariate_normal(g_0, G_0)
    sigma_matrix = flat_sigma_to_sigma_matrix(sigma_flat)
    
    while not is_positive_definite(sigma_matrix):
        sigma_flat = np.random.multivariate_normal(g_0, G_0)
        sigma_matrix = flat_sigma_to_sigma_matrix(sigma_flat)
    
    # load the data
    data_dict_try = np.load('../data/data_dict.npy', allow_pickle=True).item()  
    s, w, x, y, z = data_dict_try['s'], data_dict_try['w'], data_dict_try['x'], data_dict_try['y'], data_dict_try['z']
    
    return beta, sigma_flat, s, w, x, y, z