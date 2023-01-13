import torch
import math
import scipy.stats as stats

from utils import is_positive_definite, flat_sigma_to_sigma_matrix, sigma_matrix_to_flat_sigma

def log_likelyhood_ys(w, x, s, z, gamma, beta_0, beta_1, sigma_matrix):
    
    log_prod = 0
    for i in range(len(s)):
        treatment = s[i]
        if treatment == 0:
            mean = torch.dot(x[i], beta_0)
            std = sigma_matrix[1][1]**2
            value = ((torch.dot(-w[i], gamma) - (sigma_matrix[0][1]/sigma_matrix[1][1])
                      * (z[0][i] - torch.dot(x[i], beta_0)))
                      / (1-sigma_matrix[0][1]**2/sigma_matrix[1][1])**0.5)
        else:
            mean = torch.dot(x[i], beta_1)
            std = sigma_matrix[2][2]**2
            value = ((torch.dot(-w[i], gamma) - (sigma_matrix[0][2]/sigma_matrix[2][2])
                      * (z[1][i] - torch.dot(x[i], beta_1)))
                      / (1-sigma_matrix[0][2]**2/sigma_matrix[2][2])**0.5)
            
        log_fn = torch.distributions.normal.Normal(mean, std).log_prob(z[treatment][i])
        log_normal_cdf = torch.log(0.5 * (1 + torch.erf((value) / math.sqrt(2))))

        log_prod += torch.max(torch.tensor([log_fn, + log_normal_cdf,
                              10**-300]))

    return log_prod

def log_g_sigma(w, s, x, z, beta_0, beta_1, gamma, flat_sigma, g_0, G_0):

    sigma_matrix = flat_sigma_to_sigma_matrix(flat_sigma)

    # likelihood of ys given sigma
    first_term = log_likelyhood_ys(w, x, s, z, gamma, beta_0, beta_1, sigma_matrix)

    # density of sigma given g_0, G_0
    second_term = torch.distributions.multivariate_normal.MultivariateNormal(g_0, G_0).log_prob(flat_sigma)

    third_term = 0 if is_positive_definite(sigma_matrix) else -float('inf')

    result = first_term + second_term + third_term
    return result

def q_sigma(w, s, x, z, beta_0, beta_1, gamma, sigma_matrix, g_0, G_0, sample=True):

    def sub_minus_log_g_sigma(flat_sigma):
        return log_g_sigma(w, s, x, z, beta_0, beta_1, gamma, flat_sigma, g_0, G_0)

    sigma_flat = sigma_matrix_to_flat_sigma(sigma_matrix)

    H = torch.autograd.functional.hessian(sub_minus_log_g_sigma, sigma_flat)
    V = -torch.linalg.inv(H)
    mu = -torch.diag(H)

    if sample:
        # sample from multivariate t density
        return torch.tensor(stats.multivariate_t.rvs(loc=mu, shape=V))
    else:
        # return likelyhood of sigma
        return stats.multivariate_t.pdf(sigma_flat, mu, V)
