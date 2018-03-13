# -*- coding: utf-8 -*-
"""
Gaussian Process with 2. order taylor expansion to propagate the variance

@author: Helge-André Langåker
"""

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")

import casadi as ca

dir_data = '../data/'
dir_parameters = '../parameters/'


def gp_taylor_approx(invK, X, F, hyper, D, inputmean, inputcov):
    #hyper = ca.MX.log(hyper)
    E     = len(invK)
    n     = ca.MX.size(F[:, 1])[0]
    mean  = ca.MX.zeros(E, 1)
    var = ca.MX.zeros(D, 1)
    v     = X - ca.repmat(inputmean, n, 1)
    covar = ca.MX.zeros(E, E)
    covariance = ca.MX.zeros(E, E)
    d_mean = ca.MX.zeros(E, 1)
    dd_var = ca.MX.zeros(E, E)

    for a in range(E):
        ell = hyper[a, :D]
        w = 1 / ell**2
        sf2 = ca.MX(hyper[a, D]**2)
        iK = ca.MX(invK[a])
        alpha = ca.mtimes(iK, F[:, a])
        kss = sf2
        ks = ca.MX.zeros(n, 1)

        for i in range(n):
            dist = ca.sum2((X[i, :] - inputmean)**2 / ell**2)
            ks[i] = sf2 * ca.MX.exp(-.5 * dist)

        invKks = ca.mtimes(iK, ks)
        mean[a] = ca.mtimes(ks.T, alpha)
        var[a] = kss - ca.mtimes(ks.T, invKks)
        d_mean[a] = ca.mtimes(ca.transpose(w[a] * v[:, a] * ks), alpha)

    for d in range(1):
        for e in range(1):
            dd_var1a = ca.mtimes(ca.transpose(v[:, d] * ks), iK)
            dd_var1b = ca.mtimes(dd_var1a, v[e] * ks)
            dd_var2 = ca.mtimes(ca.transpose(v[d] * v[e] * ks), invKks)
            dd_var[d, e] = -2 * w[d] * w[e] * (dd_var1b + dd_var2)
            if d == e:
                dd_var[d, e] = dd_var[d, e] + 2 * w[d] * (kss - var[d])

    # covariance with noisy input
    for a in range(E):
        covar1 = ca.mtimes(d_mean, d_mean.T)
        covar[0, 0] = inputcov[a]
        covariance[a, a] = var[a] + ca.trace(ca.mtimes(covar, .5 * dd_var + covar1))
        #covariance[a, a] = var[a] + covar[0, 0] * .5 * dd_var[0, 0] + ca.trace(covar1)
        #covariance[a] = var[a] + ca.trace(ca.mtimes(inputcov, .5 * dd_var + covar1))
    return [mean, covariance]