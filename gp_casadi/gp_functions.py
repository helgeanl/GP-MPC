# -*- coding: utf-8 -*-
"""
Gaussian Process
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")

import numpy as np
import casadi as ca


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    dist = ca.sum2((x - z)**2 / ell**2)
    return sf2 * ca.MX.exp(-.5 * dist)


def calc_cov_matrix_casadi(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = ca.SX.size(X)
    for i in range(D):
        x = X[:, i].reshape((n, 1))
        dist = (ca.sum2(x**2).reshape((-1, 1)) + ca.sum2(x**2) -
                2 * ca.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * ca.SX.exp(-.5 * dist)


def gp(invK, hyp, X, Y, z):
    """ Gaussian Process

    # Arguments
        invK: Inverse covariance matrix of size NxN.
        hyp: Array with hyperparameters [ell_1 .. ell_D sf sn], where D is the
            number of inputs to the GP.
        X: Training data matrix with inputs of size NxD.
        Y: Training data matrix with outpyts of size NxE, with E number of outputs.
        z: Input to the GP of size 1xD

    # Returns
        mean: The estimated mean.
        var: The estimated variance
    """

    E = len(invK)
    n, D = ca.MX.size(X)

    mean  = ca.MX.zeros(E, 1)
    var  = ca.MX.zeros(E, 1)
    for a in range(E):
        ell = ca.MX(hyp[a, 0:D])
        sf2 = ca.MX(hyp[a, D]**2)
        #m = hyp[a, D + 2]
        kss = covSEard(z, z, ell, sf2)
        ks = ca.MX.zeros(n, 1)

        for i in range(n):
            ks[i] = covSEard(X[i, :], z, ell, sf2)
        #ks = repmat()
        ksK = ca.mtimes(ks.T, invK[a])

        mean[a] = ca.mtimes(ksK, Y[:, a])
        var[a] = kss - ca.mtimes(ksK, ks)

    return mean, var


def gp_taylor_approx(invK, X, F, hyper, inputmean, inputvar):
    """ Gaussian Process with Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a second order taylor for estimating the variance.

    # Arguments
        invK: Array with the inverse covariance matrices of size DxNxN, with D
            number of inputs to the GP and N number of training points.
        X: Training data matrix with inputs of size NxD.
        F: Training data matrix with outpyts of size NxE, with E number of outputs.
        hyper: Array with hyperparameters [ell_1 .. ell_D sf sn].
        inputmean: Mean from the last GP iteration of size 1xE
        inputvar: Variance from the last GP iteration of size 1xE

    # Returns
        mean: Array with estimated mean of size Ex1.
        covariance: The estimated covariance matrix with the output variance in the
                    diagonal of size ExE.
    """
    E     = len(invK)
    n, D     = ca.MX.size(X)
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
        covar[0, 0] = inputvar[a]
        covariance[a, a] = var[a] + ca.trace(ca.mtimes(covar, .5 * dd_var + covar1))
        #covariance[a, a] = var[a] + covar[0, 0] * .5 * dd_var[0, 0] + ca.trace(covar1)
        #covariance[a] = var[a] + ca.trace(ca.mtimes(inputcov, .5 * dd_var + covar1))
    return [mean, covariance]


def gp_exact_moment(invK, X, F, hyper, D, inputmean, inputcov):
    """ Gaussian Process with Exact Moment Matching

    The first and second moments are used to compute the mean and covariance of the
    posterior distribution with a stochastic input distribution. This assumes a
    zero prior mean function and the squared exponential kernel.

    # Arguments
        invK: Array with the inverse covariance matrices of size DxNxN, with D
            number of inputs to the GP and N number of training points.
        X: Training data matrix with inputs of size NxD.
        F: Training data matrix with outpyts of size NxE, with E number of outputs.
        hyper: Array with hyperparameters [ell_1 .. ell_D sf sn].
        inputmean: Mean from the last GP iteration of size 1xD
        inputcov: Covariance matrix from the last GP iteration of size ExE

    # Returns
        mean: Array of the output mean of size Ex1.
        covariance: Covariance matrix of size ExE.
    """

    hyper = ca.log(hyper)
    E     = len(invK)
    n     = ca.MX.size(F[:, 1])[0]
    mean  = ca.MX.zeros(E, 1)
    beta  = ca.MX.zeros(n, E)
    log_k = ca.MX.zeros(n, E)
    v     = X - ca.repmat(inputmean, n, 1)

    #invK = MX(invK)
    covariance = ca.MX.zeros(E, E)

    A = ca.SX.sym('A', inputcov.shape)
    [Q, R2] = ca.qr(A)
    determinant = ca.Function('determinant', [A], [ca.exp(ca.trace(ca.log(R2)))])

    for a in range(E):
        beta[:, a] = ca.mtimes(invK[a], F[:, a])
        iLambda   = ca.diag(ca.exp(-2 * hyper[a, :D]))
        R  = inputcov + ca.diag(ca.exp(2 * hyper[a, :D]))
        iR = ca.mtimes(iLambda, (ca.MX.eye(D) - ca.solve((ca.MX.eye(D) + ca.mtimes(inputcov, iLambda)), (ca.mtimes(inputcov, iLambda)))))
        T  = ca.mtimes(v, iR)
        c  = ca.exp(2 * hyper[a, D]) / ca.sqrt(determinant(R)) * ca.exp(ca.sum2(hyper[a, :D]))
        q2 = c * ca.exp(-ca.sum2(T * v) * 0.5)
        qb = q2 * beta[:, a]
        mean[a] = ca.sum1(qb)
        t  = ca.repmat(ca.exp(hyper[a, :D]), n, 1)
        v1 = v / t
        log_k[:, a] = 2 * hyper[a, D] - ca.sum2(v1 * v1) * 0.5

    # covariance with noisy input
    for a in range(E):
        ii = v / ca.repmat(ca.exp(2 * hyper[a, :D]), n, 1)
        for b in range(a + 1):
            R = ca.mtimes(inputcov, ca.diag(ca.exp(-2 * hyper[a, :D]) + ca.exp(-2 * hyper[b, :D]))) + ca.MX.eye(D)

            t = 1.0 / ca.sqrt(determinant(R))
            ij = v / ca.repmat(ca.exp(2 * hyper[b, :D]), n, 1)
            Q = ca.exp(ca.repmat(log_k[:, a], 1, n) + ca.repmat(ca.transpose(log_k[:, b]), n, 1) + maha(ii, -ij, ca.solve(R, inputcov * 0.5), n))
            A = ca.mtimes(beta[:, a], ca.transpose(beta[:, b]))
            if b == a:
                A = A - invK[a]
            A = A * Q
            covariance[a, b] = t * ca.sum2(ca.sum1(A))
            covariance[b, a] = covariance[a, b]
        covariance[a, a] = covariance[a, a] + ca.exp(2 * hyper[a, D])
    covariance = covariance - ca.mtimes(mean, ca.transpose(mean))

    return [mean, covariance]


def maha(a1, b1, Q1, n):
    """Calculate the Mahalanobis distance"""
    aQ = ca.mtimes(a1, Q1)
    bQ = ca.mtimes(b1, Q1)
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, n) + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), n, 1) - 2 * ca.mtimes(aQ, ca.transpose(b1))
    return K1
