# -*- coding: utf-8 -*-
"""
Gaussian Process functions
Copyright (c) 2018, Helge-André Langåker, Eric Bradford
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel
        Copyright (c) 2018, Helge-André Langåker
    """
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)


def get_mean_function(hyper, X, func='zero'):
    """ Get mean function
        Copyright (c) 2018, Helge-André Langåker

    # Arguments:
        hyper: Matrix with hyperperameters.
        X: Input vector or matrix.
        func: Option for mean function:
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = aT*x^2 + bT*x + c
    # Returns:
         CasADi mean function [m(X, hyper)]
    """

    Nx, N = X.shape
    X_s = ca.SX.sym('x', Nx, N)
    Z_s = ca.MX.sym('x', Nx, N)
    m = ca.SX(N, 1)
    hyp_s = ca.SX.sym('hyper', *hyper.shape)
    if func == 'zero':
        meanF = ca.Function('zero_mean', [X_s, hyp_s], [m])
    elif func == 'const':
        a =  hyp_s[-1]
        for i in range(N):
            m[i] = a
        meanF = ca.Function('const_mean', [X_s, hyp_s], [m])
    elif func == 'linear':
        a = hyp_s[-Nx-1:-1].reshape((1, Nx))
        b = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[:,i]) + b
        meanF = ca.Function('linear_mean', [X_s, hyp_s], [m])
    elif func == 'polynomial':
        a = hyp_s[-2*Nx-1:-Nx-1].reshape((1,Nx))
        b = hyp_s[-Nx-1:-1].reshape((1,Nx))
        c = hyp_s[-1]
        for i in range(N):
            m[i] = ca.mtimes(a, X_s[:, i]**2) + ca.mtimes(b, X_s[:,i]) + c
        meanF = ca.Function('poly_mean', [X_s, hyp_s], [m])
    else:
        raise NameError('No mean function called: ' + func)

    return ca.Function('mean', [Z_s], [meanF(Z_s, hyper)])


def build_gp(invK, X, hyper, alpha, chol, meanFunc='zero'):
    """ Build Gaussian Process function
        Copyright (c) 2018, Helge-André Langåker

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size (N x Nx), with Nx number of
            inputs to the GP.
        alpha: Training data matrix with invK time outputs of size (Ny x N).
        hyper: Array with hyperparame|ters [ell_1 .. ell_Nx sf sn].

    # Returns
        mean:     GP mean casadi function [mean(z)]
        var:      GP variance casadi function [var(z)]
        covar:    GP covariance casadi function [covar(z) = diag(var(z))]
        mean_jac: Casadi jacobian of the GP mean function [jac(z)]
    """


    Ny = len(invK)
    X = ca.SX(X)
    N, Nx = ca.SX.size(X)

    mean  = ca.SX.zeros(Ny, 1)
    var   = ca.SX.zeros(Ny, 1)

    # Casadi symbols
    x_s        = ca.SX.sym('x', Nx)
    z_s        = ca.SX.sym('z', Nx)
    m_s        = ca.SX.sym('m')
    ell_s      = ca.SX.sym('ell', Nx)
    sf2_s      = ca.SX.sym('sf2')
    X_s        = ca.SX.sym('X', N, Nx)
    ks_s       = ca.SX.sym('ks', N)
    v_s       = ca.SX.sym('v', N)
    kss_s      = ca.SX.sym('kss')
    alpha_s    = ca.SX.sym('alpha', N)

    covSE = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s],
                          [covSEard(x_s, z_s, ell_s, sf2_s)])

    ks = ca.SX.zeros(N, 1)
    for i in range(N):
        ks[i] = covSE(X_s[i, :], z_s, ell_s, sf2_s)
    ks_func = ca.Function('ks', [X_s, z_s, ell_s, sf2_s], [ks])

    mean_i_func = ca.Function('mean', [ks_s, alpha_s, m_s],
                            [ca.mtimes(ks_s.T, alpha_s) + m_s])

    L_s = ca.SX.sym('L', ca.Sparsity.lower(N))
    v_func = ca.Function('v', [L_s, ks_s], [ca.solve(L_s, ks_s)])

    var_i_func  = ca.Function('var', [v_s, kss_s,],
                            [kss_s - v_s.T @ v_s])

    for output in range(Ny):
        ell      = ca.SX(hyper[output, 0:Nx])
        sf2      = ca.SX(hyper[output, Nx]**2)
        alpha_a  = ca.SX(alpha[output])
        ks       = ks_func(X_s, z_s, ell, sf2)
        v        = v_func(chol[output], ks)
        m = get_mean_function(ca.MX(hyper[output, :]), z_s, func=meanFunc)
        mean[output] = mean_i_func(ks, alpha_a, m(z_s))
        var[output]  = var_i_func(v, sf2)


    mean_temp  = ca.Function('mean_temp', [z_s, X_s], [mean])
    var_temp   = ca.Function('var_temp',  [z_s, X_s], [var])

    mean_func  = ca.Function('mean', [z_s], [mean_temp(z_s, X)])
    covar_func = ca.Function('var',  [z_s], [ca.diag(var_temp(z_s, X))])
    var_func = ca.Function('var',  [z_s], [var_temp(z_s, X)])

    mean_jac_z = ca.Function('mean_jac_z', [z_s],
                                      [ca.jacobian(mean_func(z_s), z_s)])

    return mean_func, var_func, covar_func, mean_jac_z


def build_TA_cov(mean, covar, jac, Nx, Ny):
    """ Build 1st order Taylor approximation of covariance function
        Copyright (c) 2018, Helge-André Langåker

    # Arguments:
        mean: GP mean casadi function [mean(z)]
        covar: GP covariance casadi function [covar(z)]
        jac: Casadi jacobian of the GP mean function [jac(z)]
        Nx: Number of inputs to the GP
        Ny: Number of ouputs from the GP

    # Return:
        cov: Casadi function with the approximated covariance
             function [cov(z, covar_x)].
    """
    cov_z  = ca.SX.sym('cov_z', Nx, Nx)
    z_s    = ca.SX.sym('z', Nx)
    jac_z = jac(z_s)
    cov    = ca.Function('cov', [z_s, cov_z],
                      [covar(z_s) + jac_z @ cov_z @ jac_z.T])

    return cov


def gp(invK, X, Y, hyper, inputmean,  alpha=None, meanFunc='zero', log=False):
    """ Gaussian Process
        Copyright (c) 2018, Helge-André Langåker

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N),
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size (N x Nx), with Nx number of
            inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        hyper: Array with hyperparame|ters [ell_1 .. ell_Nx sf sn].
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
    x_s     = ca.SX.sym('x', Nx)
    z_s     = ca.SX.sym('z', Nx)
    ell_s   = ca.SX.sym('ell', Nx)
    sf2_s   = ca.SX.sym('sf2')

    invK_s  = ca.SX.sym('invK', N, N)
    Y_s     = ca.SX.sym('Y', N)
    m_s     = ca.SX.sym('m')
    ks_s    = ca.SX.sym('ks', N)
    kss_s   = ca.SX.sym('kss')
    ksT_invK_s = ca.SX.sym('ksT_invK', 1, N)
    alpha_s = ca.SX.sym('alpha', N)

    covSE = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s],
                          [covSEard(x_s, z_s, ell_s, sf2_s)])

    ksT_invK_func = ca.Function('ksT_invK', [ks_s, invK_s],
                           [ca.mtimes(ks_s.T, invK_s)])

    if alpha is not None:
        mean_func = ca.Function('mean', [ks_s, alpha_s],
                           [ca.mtimes(ks_s.T, alpha_s)])
    else:
        mean_func = ca.Function('mean', [ksT_invK_s, Y_s],
                           [ca.mtimes(ksT_invK_s, Y_s)])

    var_func  = ca.Function('var', [kss_s, ksT_invK_s, ks_s],
                            [kss_s - ca.mtimes(ksT_invK_s, ks_s)])

    for output in range(Ny):
        m = get_mean_function(hyper[output, :], inputmean, func=meanFunc)
        ell = ca.MX(hyper[output, 0:Nx])
        sf2 = ca.MX(hyper[output, Nx]**2)

        kss = covSE(inputmean, inputmean, ell, sf2)
        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = covSE(X[i, :], inputmean, ell, sf2)

        ksT_invK = ksT_invK_func(ks, ca.MX(invK[output]))
        if alpha is not None:
            mean[output] = mean_func(ks, ca.MX(alpha[output]))
        else:
            mean[output] = mean_func(ksT_invK, Y[:, output])
        var[output] = var_func(kss, ks, ksT_invK)

    if log:
        mean = ca.exp(mean)
        var = ca.exp(var)

    covar = ca.diag(var)
    return mean, covar


def gp_taylor_approx(invK, X, Y, hyper, inputmean, inputcovar,
                     meanFunc='zero', diag=False, log=False):
    """ Gaussian Process with Taylor Approximation
        Copyright (c) 2018, Helge-André Langåker

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
    covar_temp      = ca.MX.zeros(Ny, Ny)

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

        #BUG: This don't take into account the covariance between states
        for d in range(Ny):
            for e in range(Ny):
                dd_var1a = ca.mtimes(ca.transpose(v[:, d] * ks), iK)
                dd_var1b = ca.mtimes(dd_var1a, v[e] * ks)
                dd_var2 = ca.mtimes(ca.transpose(v[d] * v[e] * ks), invKks)
                dd_var[d, e] = -2 * w[d] * w[e] * (dd_var1b + dd_var2)
                if d == e:
                    dd_var[d, e] = dd_var[d, e] + 2 * w[d] * (kss - var[d])

        mean_mat = ca.mtimes(d_mean, d_mean.T)
        covar_temp[0, 0] = inputcovar[a, a]
        covariance[a, a] = var[a] + ca.trace(ca.mtimes(covar_temp, .5
                                         * dd_var + mean_mat))

    return [mean, covariance]



def gp_exact_moment(invK, X, Y, hyper, inputmean, inputcov):
    """ Gaussian Process with Exact Moment Matching
    Copyright (c) 2018, Eric Bradford, Helge-André Langåker

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
        inputcov: Covariance matrix from the last GP iteration of size (Nx x Nx)

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

    #TODO: Fix that LinsolQr don't work with the extended graph?
    A = ca.SX.sym('A', inputcov.shape)
    [Q, R2] = ca.qr(A)
    determinant = ca.Function('determinant', [A], [ca.exp(ca.trace(ca.log(R2)))])

    for a in range(Ny):
        beta[:, a] = ca.mtimes(invK[a], Y[:, a])
        iLambda   = ca.diag(ca.exp(-2 * hyper[a, :Nx]))
        R  = inputcov + ca.diag(ca.exp(2 * hyper[a, :Nx]))
        iR = ca.mtimes(iLambda, (ca.MX.eye(Nx) - ca.solve((ca.MX.eye(Nx)
                + ca.mtimes(inputcov, iLambda)), (ca.mtimes(inputcov, iLambda)))))
        T  = ca.mtimes(v, iR)
        c  = ca.exp(2 * hyper[a, Nx]) / ca.sqrt(determinant(R)) \
                * ca.exp(ca.sum2(hyper[a, :Nx]))
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
            R = ca.mtimes(inputcov, ca.diag(ca.exp(-2 * hyper[a, :Nx])
                + ca.exp(-2 * hyper[b, :Nx]))) + ca.MX.eye(Nx)
            t = 1.0 / ca.sqrt(determinant(R))
            ij = v / ca.repmat(ca.exp(2 * hyper[b, :Nx]), N, 1)
            Q = ca.exp(ca.repmat(log_k[:, a], 1, N)
                + ca.repmat(ca.transpose(log_k[:, b]), N, 1)
                + maha(ii, -ij, ca.solve(R, inputcov * 0.5), N))
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
    """Calculate the Mahalanobis distance
    Copyright (c) 2018, Eric Bradford
    """
    aQ = ca.mtimes(a1, Q1)
    bQ = ca.mtimes(b1, Q1)
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, N) \
            + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), N, 1) \
            - 2 * ca.mtimes(aQ, ca.transpose(b1))
    return K1
