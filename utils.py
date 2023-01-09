import numpy as np

def is_positive_definite(A):
    return int(np.all(np.linalg.eigvals(A) > 0))

def flat_sigma_to_sigma_matrix(sigma_flat):
    sigma_matrix = np.zeros((3, 3))
    sigma_matrix[0, 0] = 1
    sigma_matrix[1, 1] = sigma_flat[2]
    sigma_matrix[2, 2] = sigma_flat[3]

    sigma_matrix[0, 1] = sigma_flat[0]
    sigma_matrix[1, 0] = sigma_flat[0]

    sigma_matrix[0, 2] = sigma_flat[1]
    sigma_matrix[2, 0] = sigma_flat[1]
    
    sigma_matrix[1, 2] = 0
    sigma_matrix[2, 1] = 0

    return sigma_matrix

def sigma_matrix_to_flat_sigma(sigma_matrix):
    sigma_flat = np.zeros(4)
    sigma_flat[0] = sigma_matrix[0, 1]
    sigma_flat[1] = sigma_matrix[0, 2]
    sigma_flat[2] = sigma_matrix[1, 1]
    sigma_flat[3] = sigma_matrix[2, 2]
    return sigma_flat