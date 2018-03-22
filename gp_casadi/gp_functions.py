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

"""
                'zero':       m = 0 
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = xT*diag(a)*x + bT*x + c
"""

def meanlinear(a, b, X):
    return a * X + b

def gp(invK, X, Y, hyp, z,  meanFunc='zero'):
    """ Gaussian Process

    # Arguments
        invK: Array with the inverse covariance matrices of size (Ny x N x N), 
            with Ny number of outputs from the GP and N number of training points.
        X: Training data matrix with inputs of size NxNx, with Nx number of
            inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny).
        hyp: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
        z: Input to the GP of size 1xD

    # Returns
        mean: The estimated mean.
        var: The estimated variance
    """

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
        if meanFunc == 'zero':
            m = ca.MX(1,1)
        elif meanFunc == 'const':
            m =  hyp[output, -1]
        elif meanFunc == 'linear':
            print('Linear mean function')
            a_s = ca.SX.sym('a', Nx)
            b_s = ca.SX.sym('b')
            X_s = ca.SX.sym('X', 1, Nx)
            meanF = ca.Function('mean', [X_s, a_s, b_s], 
                              [ca.sum2(ca.transpose(a_s * X_s.T)) + b_s])
            a = hyp[output, -Nx-1:-1]
            b = hyp[output, -1]
            m = meanF(z, a, b)
        elif meanFunc == 'polynomial':
            a_s = ca.SX.sym('a', Nx)
            b_s = ca.SX.sym('b', Nx)
            c_s = ca.SX.sym('b')
            X_s = ca.SX.sym('X', 1, Nx)
            meanF = ca.Function('mean', [X_s, a_s, b_s, c_s], 
                              [ca.sum2(ca.transpose(a_s * X_s.T**2)) + 
                               ca.sum2(ca.transpose(b_s * X_s.T)) + c_s])
            a = hyp[output, -2*Nx-1:-Nx-1]
            b = hyp[output, -Nx-1:-1]
            c = hyp[output, -1]
            m = meanF(z, a, b, c)
            
        ell = ca.MX(hyp[output, 0:Nx])
        sf2 = ca.MX(hyp[output, Nx]**2)
        #m = hyp[a, D + 2]
        
        kss = covSE(z, z, ell, sf2)
        
        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = covSE(X[i, :], z, ell, sf2)

        ksK = ca.mtimes(ks.T, invK[output])

        mean[output] = ca.mtimes(ksK, Y[:, output] - m) + m
        var[output] = kss - ca.mtimes(ksK, ks)

    return mean, var


def gp_taylor_approx(invK, X, Y, hyper, inputmean, inputvar):
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
    Ny     = len(invK)
    N, Nx     = ca.MX.size(X)
    mean  = ca.MX.zeros(Ny, 1)
    var = ca.MX.zeros(Nx, 1)
    v     = X - ca.repmat(inputmean, N, 1)
    covar = ca.MX.zeros(Ny, Ny)
    covariance = ca.MX.zeros(Ny, Ny)
    d_mean = ca.MX.zeros(Ny, 1)
    dd_var = ca.MX.zeros(Ny, Ny)
    
    # Casadi symbols
    x_s = ca.SX.sym('x', Nx)
    z_s = ca.SX.sym('z', Nx)
    ell_s = ca.SX.sym('ell', Nx)
    sf2_s = ca.SX.sym('sf2')
    covSE = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s], 
                          [covSEard(x_s, z_s, ell_s, sf2_s)])

    for a in range(Ny):
        ell = hyper[a, :Nx]
        w = 1 / ell**2
        sf2 = ca.MX(hyper[a, Nx]**2)
        iK = ca.MX(invK[a])
        alpha = ca.mtimes(iK, Y[:, a])
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
        covariance[a, a] = var[a] + ca.trace(ca.mtimes(covar, .5 * dd_var + covar1))

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
