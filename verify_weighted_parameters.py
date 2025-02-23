#!/usr/bin/env python3

import random



def gaussian(mu, sigma):
    return random.gauss(mu,sigma) 

def mean(x):
    return sum(x) / len(x)

def variance(x):
    mu = mean(x)
    return sum([(xi - mu) ** 2 for xi in x]) / len(x)
    
if __name__ == "__main__":
    # chosen parameters
    mu1 = 2
    sigma1 = 2
    mu2 = 5
    sigma2 = 4
    n1 = 1000
    n2 = 10000
    
    # generate data according to the chosen parameters
    x1 = [gaussian(mu1, sigma1) for _ in range(n1)]
    x2 = [gaussian(mu2, sigma2) for _ in range(n2)]

    # compute the estimated parameters
    print(mean(x1), variance(x1))
    print(mean(x2), variance(x2))

    # check our math on calculated the weighted average mean (using mu and n as
    # sufficient statistics)
    mu_combined = (n1 * mean(x1) + n2 * mean(x2)) / (n1 + n2)
    print(mean(x1 + x2), mu_combined)
    assert abs(mean(x1 + x2) - mu_combined) < 1e-10

    # check our math on calculated the weighted average variance (using mu, n,
    # sum(xi^2), # and sum(xi) as sufficient # statistics)
    sum1_2 = sum([xi ** 2 for xi in x1])
    sum1 = sum(x1)

    sum2_2 = sum([xi ** 2 for xi in x2])
    sum2 = sum(x2)

    sigma2_combined = (sum1_2 + sum2_2 - 2 * mu_combined * (sum1 + sum2) + (n1 + n2) * (mu_combined ** 2)) / (n1 + n2)

    sigmas_combined= sum1_2+sum2_2-mu_combined-(mu1*sum1)+(mu1*n1)-(mu2*sum2)+(mu2*n2)
    print(sum1_2+sum2_2,sum1,sum2)
    print(variance(x1 + x2), sigma2_combined, sigmas_combined)
    assert abs(variance(x1 + x2) - sigma2_combined) < 1e-10
