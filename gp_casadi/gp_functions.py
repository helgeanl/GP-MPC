# -*- coding: utf-8 -*-
"""
Gaussian Process
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")

import casadi as ca


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)




def get_mean_function(hyper, X, func='zero'):
    """ Get mean function
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = xT*diag(a)*x + bT*x + c
    """

    N, Nx = X.shape
    X_s = ca.SX.sym('x', N, Nx)
    Z_s = ca.MX.sym('x', N, Nx)
    m = ca.SX(N, 1)
    hyp_s = ca.SX.sym('hyper', hyper.shape)
    if func == 'zero':
        a = ca.SX(1,1)
        meanF = ca.Function('mean', [X_s, hyp_s], [a])
    elif func == 'const':
        a =  hyp_s[-1]
        for i in range(N):
            m[i] = a
        meanF = ca.Function('mean', [X_s, hyp_s], [m])
    elif func == 'linear':
        a = hyp_s[-Nx-1:-1]
        b = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[i, :].T) + b
        meanF = ca.Function('mean', [X_s, hyp_s], [m])
    elif func == 'polynomial':
        a = hyp_s[-2*Nx-1:-Nx-1]
        b = hyp_s[-Nx-1:-1]
        c = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[i, :].T**2) + ca.mtimes(b, X_s[i, :].T) + c
        meanF = ca.Function('mean', [X_s, hyp_s], [m])
    else:
        raise NameError('No mean function called: ' + func)

    return ca.Function('mean', [Z_s], [meanF(Z_s, hyper)])


def gp(invK, X, Y, hyper, inputmean,  meanFunc='zero', log=False):
    """ Gaussian Process

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size (N x Nx), with Nx number of
            inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
        inputmean: Input to the GP of size (1 x Nx)

    # Returns
        mean: The estimated mean.
        var: The estimated variance
    """
    if log:
        X = ca.log(X)
        Y = ca.log(Y)
        inputmean = ca.log(inputmean)

    Ny = len(invK)
    N, Nx = ca.MX.size(X)

    mean  = ca.MX.zeros(Ny, 1)
    var  = ca.MX.zeros(Ny, 1)

    # Casadi symbols
    x_s = ca.SX.sym('x', Nx)
    z_s = ca.SX.sym('z', Nx)
    ell_s = ca.SX.sym('ell', Nx)
    sf2_s = ca.SX.sym('sf2')
    covSE = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s],
                          [covSEard(x_s, z_s, ell_s, sf2_s)])

    for output in range(Ny):
        m = get_mean_function(hyper[output, :], inputmean, func=meanFunc)
        ell = ca.MX(hyper[output, 0:Nx])
        sf2 = ca.MX(hyper[output, Nx]**2)

        kss = covSE(inputmean, inputmean, ell, sf2)
        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = covSE(X[i, :], inputmean, ell, sf2)

        ksK = ca.mtimes(ks.T, invK[output])

        mean[output] = ca.mtimes(ksK, Y[:, output] - m(inputmean)) + m(inputmean)
        var[output] = kss - ca.mtimes(ksK, ks)
    
    if log:
        mean = ca.exp(mean)
        var = ca.exp(var)
    return mean, var


def gp_taylor_approx(invK, X, Y, hyper, inputmean, inputvar,
                     meanFunc='zero', diag=False, log=False):
    """ Gaussian Process with Taylor Approximation

    This uses a first order taylor for the mean evaluation (a normal GP mean),
    and a second order taylor for estimating the variance.

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size NxNx, with Nx number of
            inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
        inputmean: Mean from the last GP iteration of size (1 x Nx)
        inputvar: Variance from the last GP iteration of size (1 x Ny)

    # Returns
        mean: Array with estimated mean of size (Ny x 1).
        covariance: The estimated covariance matrix with the output variance in the
                    diagonal of size (Ny x Ny).
    """
    if log:
        X = ca.log(X)
        Y = ca.log(Y)
        inputmean = ca.log(inputmean)

    Ny         = len(invK)
    N, Nx      = ca.MX.size(X)
    mean       = ca.MX.zeros(Ny, 1)
    var        = ca.MX.zeros(Nx, 1)
    v          = X - ca.repmat(inputmean, N, 1)
    covar      = ca.MX.zeros(Ny, Ny)
    variance   = ca.MX.zeros(Ny)
    covariance = ca.MX.zeros(Ny, Ny)
    d_mean     = ca.MX.zeros(Ny, 1)
    dd_var     = ca.MX.zeros(Ny, Ny)

    # Casadi symbols
    x_s     = ca.SX.sym('x', Nx)
    z_s     = ca.SX.sym('z', Nx)
    ell_s   = ca.SX.sym('ell', Nx)
    sf2_s   = ca.SX.sym('sf2')
    covSE   = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s],
                          [covSEard(x_s, z_s, ell_s, sf2_s)])

    for a in range(Ny):
        ell = hyper[a, :Nx]
        w = 1 / ell**2
        sf2 = ca.MX(hyper[a, Nx]**2)
        m = get_mean_function(hyper[a, :], inputmean, func=meanFunc)
        iK = ca.MX(invK[a])
        alpha = ca.mtimes(iK, Y[:, a] - m(inputmean)) + m(inputmean)
        kss = sf2

        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = covSE(X[i, :], inputmean, ell, sf2)

        invKks = ca.mtimes(iK, ks)
        mean[a] = ca.mtimes(ks.T, alpha)
        var[a] = kss - ca.mtimes(ks.T, invKks)
        d_mean[a] = ca.mtimes(ca.transpose(w[a] * v[:, a] * ks), alpha)

        for d in range(Ny):
            for e in range(Ny):
                dd_var1a = ca.mtimes(ca.transpose(v[:, d] * ks), iK)
                dd_var1b = ca.mtimes(dd_var1a, v[e] * ks)
                dd_var2 = ca.mtimes(ca.transpose(v[d] * v[e] * ks), invKks)
                dd_var[d, e] = -2 * w[d] * w[e] * (dd_var1b + dd_var2)
                if d == e:
                    dd_var[d, e] = dd_var[d, e] + 2 * w[d] * (kss - var[d])

        covar1 = ca.mtimes(d_mean, d_mean.T)
        covar[0, 0] = inputvar[a]
        if diag:
            variance[a] = var[a] + ca.trace(ca.mtimes(covar, .5 * dd_var + covar1))
        else:
            covariance[a, a] = var[a] + ca.trace(ca.mtimes(covar, .5 * dd_var + covar1))

    if log:
        mean = ca.exp(mean)
        variance = ca.exp(variance)
        
    if diag:
        return [mean, variance]
    else:
        return [mean, covariance]


def gp_exact_moment(invK, X, Y, hyper, inputmean, inputcov):
    """ Gaussian Process with Exact Moment Matching

    The first and second moments are used to compute the mean and covariance of the
    posterior distribution with a stochastic input distribution. This assumes a
    zero prior mean function and the squared exponential kernel.

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size NxNx, with Nx number of
            inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
        inputmean: Mean from the last GP iteration of size (1 x Nx)
        inputcov: Covariance matrix from the last GP iteration of size (Ny x Ny)

    # Returns
        mean: Array of the output mean of size (Ny x 1).
        covariance: Covariance matrix of size (Ny x Ny).
    """

    hyper = ca.log(hyper)
    Ny     = len(invK)
    N, Nx     = ca.MX.size(X)
    mean  = ca.MX.zeros(Ny, 1)
    beta  = ca.MX.zeros(N, Ny)
    log_k = ca.MX.zeros(N, Ny)
    v     = X - ca.repmat(inputmean, N, 1)

    covariance = ca.MX.zeros(Ny, Ny)

    A = ca.SX.sym('A', inputcov.shape)
    [Q, R2] = ca.qr(A)
    determinant = ca.Function('determinant', [A], [ca.exp(ca.trace(ca.log(R2)))])

    for a in range(Ny):
        beta[:, a] = ca.mtimes(invK[a], Y[:, a])
        iLambda   = ca.diag(ca.exp(-2 * hyper[a, :Nx]))
        R  = inputcov + ca.diag(ca.exp(2 * hyper[a, :Nx]))
        iR = ca.mtimes(iLambda, (ca.MX.eye(Nx) - ca.solve((ca.MX.eye(Nx) + ca.mtimes(inputcov, iLambda)), (ca.mtimes(inputcov, iLambda)))))
        T  = ca.mtimes(v, iR)
        c  = ca.exp(2 * hyper[a, Nx]) / ca.sqrt(determinant(R)) * ca.exp(ca.sum2(hyper[a, :Nx]))
        q2 = c * ca.exp(-ca.sum2(T * v) * 0.5)
        qb = q2 * beta[:, a]
        mean[a] = ca.sum1(qb)
        t  = ca.repmat(ca.exp(hyper[a, :Nx]), N, 1)
        v1 = v / t
        log_k[:, a] = 2 * hyper[a, Nx] - ca.sum2(v1 * v1) * 0.5

    # covariance with noisy input
    for a in range(Ny):
        ii = v / ca.repmat(ca.exp(2 * hyper[a, :Nx]), N, 1)
        for b in range(a + 1):
            R = ca.mtimes(inputcov, ca.diag(ca.exp(-2 * hyper[a, :Nx]) + ca.exp(-2 * hyper[b, :Nx]))) + ca.MX.eye(Nx)
            t = 1.0 / ca.sqrt(determinant(R))
            ij = v / ca.repmat(ca.exp(2 * hyper[b, :Nx]), N, 1)
            Q = ca.exp(ca.repmat(log_k[:, a], 1, N) + ca.repmat(ca.transpose(log_k[:, b]), N, 1) + maha(ii, -ij, ca.solve(R, inputcov * 0.5), N))
            A = ca.mtimes(beta[:, a], ca.transpose(beta[:, b]))
            if b == a:
                A = A - invK[a]
            A = A * Q
            covariance[a, b] = t * ca.sum2(ca.sum1(A))
            covariance[b, a] = covariance[a, b]
        covariance[a, a] = covariance[a, a] + ca.exp(2 * hyper[a, Nx])
    covariance = covariance - ca.mtimes(mean, ca.transpose(mean))

    return [mean, covariance]


def maha(a1, b1, Q1, N):
    """Calculate the Mahalanobis distance"""
    aQ = ca.mtimes(a1, Q1)
    bQ = ca.mtimes(b1, Q1)
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, N) + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), N, 1) - 2 * ca.mtimes(aQ, ca.transpose(b1))
    return K1
