import numpy as np
import scipy.stats as stats


def log_likelyhood_ys(w, x, s, z, gamma, beta_0, beta_1, sigma_matrix):
    
    log_prod = 0
    for i in range(len(s)):
        treatment = s[i]
        if treatment == 0:
            fn = stats.norm.pdf(z[0][i],  loc=np.dot(x[i], beta_0), 
                                scale=sigma_matrix[1][1]**2)
            normal_cdf = stats.norm.cdf((np.dot(-w[i], gamma) - (sigma_matrix[0][1]/sigma_matrix[1][1])
                                         * (z[0][i] - np.dot(x[i], beta_0)))
                                         / (1-sigma_matrix[0][1]**2/sigma_matrix[1][1])**0.5)
        else:
            fn = stats.norm.pdf(z[1][i],  loc=np.dot(x[i], beta_1), 
                                scale=sigma_matrix[2][2]**2)
            normal_cdf = stats.norm.cdf((np.dot(-w[i], gamma) - (sigma_matrix[0][2]/sigma_matrix[2][2])
                                         * (z[1][i] - np.dot(x[i], beta_1)))
                                         / (1-sigma_matrix[0][2]**2/sigma_matrix[2][2])**0.5)
                    
        log_prod += np.log(max(fn * normal_cdf, 10**-300))

    return log_prod