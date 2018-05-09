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

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)



def get_mean_function(hyper, X, func='zero'):
    """ Get mean function
                'zero':       m = 0
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = aT*x^2 + bT*x + c
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
    x_s = ca.SX.sym('x', Nx)
    z_s = ca.SX.sym('z', Nx)
    ell_s = ca.SX.sym('ell', Nx)
    sf2_s = ca.SX.sym('sf2')
    covSE = ca.Function('covSE', [x_s, z_s, ell_s, sf2_s],
                          [covSEard(x_s, z_s, ell_s, sf2_s)])
    
    invK_s = ca.SX.sym('invK', N, N)
    ks_s = ca.SX.sym('ks', N)
    ksK_func = ca.Function('ksK', [ks_s, invK_s], [ca.mtimes(ks_s.T, invK_s)])
    
    ksK_s = ca.SX.sym('ksK', 1, N)
    kss_s = ca.SX.sym('kss')
    Y_s = ca.SX.sym('Y', N)
    m_s = ca.SX.sym('m')
    mean_func = ca.Function('mean', [ksK_s, Y_s, m_s], 
                       [ca.mtimes(ksK_s, Y_s - m_s + m_s)])
    
    
    var_func  = ca.Function('var', [kss_s, ks_s, ksK_s], 
                            [kss_s - ca.mtimes(ksK_s, ks_s)])

    for output in range(Ny):
        m = get_mean_function(hyper[output, :], inputmean, func=meanFunc)
        ell = ca.MX(hyper[output, 0:Nx])
        sf2 = ca.MX(hyper[output, Nx]**2)

        kss = covSE(inputmean, inputmean, ell, sf2)
        ks = ca.MX.zeros(N, 1)
        for i in range(N):
            ks[i] = covSE(X[i, :], inputmean, ell, sf2)
        
        ksK = ksK_func(ks, invK[output])

        mean[output] = mean_func(ksK, Y[:, output], m(inputmean))
        var[output] = var_func(kss, ks, ksK)
    
    if log:
        mean = ca.exp(mean)
        var = ca.exp(var)
        
    covar = ca.diag(var)
    return mean, covar


def gp_taylor_approx(invK, X, Y, hyper, inputmean, inputcovar,
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
    """Calculate the Mahalanobis distance"""
    aQ = ca.mtimes(a1, Q1)
    bQ = ca.mtimes(b1, Q1)
    K1  = ca.repmat(ca.sum2(aQ * a1), 1, N) \
            + ca.repmat(ca.transpose(ca.sum2(bQ * b1)), N, 1) \
            - 2 * ca.mtimes(aQ, ca.transpose(b1))
    return K1



def predict(X, Y, invK, hyper, x0, u, sim_system):

    # Predict future
    N = X.shape[0]
    Ny = len(invK)
    Nx = X.shape[1]
    Nu = Nx - Ny

    initVar = hyper[:, Nx + 1]**2
    covar = ca.SX.eye(Nx) * 1e-5
    dt = 30
    simTime = 150.0
    Nt = int(simTime / dt)

    mu_EM = np.zeros((Nt, Ny))
    var_EM = np.zeros((Nt, Ny))
    

    mu_ME = np.zeros((Nt, Ny))
    var_ME = np.zeros((Nt, Ny))

    mu_TA = np.zeros((Nt, Ny))
    var_TA = np.zeros((Nt, Ny))

    Y_s = ca.MX.sym('Y', N, Ny)
    X_s = ca.MX.sym('X', N, Nx)
    hyp_s = ca.MX.sym('hyp', hyper.shape)
    z_s = ca.MX.sym('z', 1, Nx)
    cov_s = ca.MX.sym('cov', Nx, Nx)


    gp_EM = ca.Function('gp', [X_s, Y_s, hyp_s, z_s, cov_s],
                        gp_exact_moment(invK, X_s, Y_s, hyp_s, z_s, cov_s))
    gp_TA = ca.Function('gp_taylor_approx', [X_s, Y_s, hyp_s, z_s, cov_s],
                        gp_taylor_approx(invK, X_s, Y_s, hyp_s, z_s, cov_s))
    gp_simple = ca.Function('gp_simple', [X_s, Y_s, hyp_s, z_s], 
                            gp(invK, X_s, Y_s, hyp_s, z_s, meanFunc='zero'))

    mu = x0
    covar[:Ny, :Ny] = ca.diag(initVar)
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, covar_x = gp_EM(X, Y, hyper, z, covar)
        mu, covar_x = mu.full(), covar.full()
        mu.shape, covar_x.shape = (Ny), (Ny, Ny)
        mu_EM[t, :], var_EM[t, :] = mu, np.diag(covar_x)
        covar[:Ny, :Ny] = covar_x

    mu = x0
    var = initVar
    covar[:Ny, :Ny] = ca.diag(initVar)
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, covar_x = gp_TA(X, Y, hyper, z, var)
        mu, covar_x = mu.full(), covar_x.full()
        mu.shape, covar.shape = (Ny), (Ny, Ny)
        mu_TA[t, :], var_TA[t, :] = mu, np.diag(covar_x)
        covar[:Ny, :Ny] = covar_x

    mu = x0
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, covar_x = gp_simple(X, Y, hyper, z)
        mu, covar_x = mu.full(), covar_x.full()
        mu.shape, covar_x.shape = (Ny), (Ny, Ny)
        mu_ME[t, :], var_ME[t, :] = mu, np.diag(covar_x)

    t = np.linspace(0.0, simTime, Nt)
    Y_sim = sim_system(x0, u, simTime, dt)


    plt.figure()
    fontP = FontProperties()
    fontP.set_size('small')
    for i in range(Ny):
        plt.subplot(2, 2, i + 1)
        mu_EM_i = mu_EM[:, i]
        mu_TA_i = mu_TA[:, i]
        mu_ME_i = mu_ME[:, i]

        sd_EM_i = np.sqrt(var_EM[:, i])
        sd_TA_i = np.sqrt(var_TA[:, i])
        sd_ME_i = np.sqrt(var_ME[:, i])

        plt.gca().fill_between(t.flat, mu_EM_i - 2 * sd_EM_i, mu_EM_i +
               2 * sd_EM_i, color="#555555", label='95% conf interval EM')
        plt.gca().fill_between(t.flat, mu_TA_i - 2 * sd_TA_i, mu_TA_i +
               2 * sd_TA_i, color="#FFFaaa", label='95% conf interval TA')
        plt.gca().fill_between(t.flat, mu_ME_i - 2 * sd_ME_i, mu_ME_i +
               2 * sd_ME_i, color="#bbbbbb", label='95% conf interval ME')

        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_EM_i, label='95% conf interval EM')
        #plt.errorbar(t, mu_TA_i, yerr=2 * sd_TA_i, label='95% conf interval TA')
        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_ME_i, label='95% conf interval ME')

        plt.plot(t, Y_sim[:, i], 'b-', label='Simulation')
        plt.plot(t, mu_EM_i, 'rx', label='GP Excact moment')
        plt.plot(t, mu_TA_i, 'kx', label='GP Taylor Approx')
        plt.plot(t, mu_ME_i, 'yx', label='GP Mean Equivalence')

        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(prop=fontP)
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()

    plt.figure()
    u_temp = np.vstack((u, u[-1, :]))
    for i in range(Nu):
        plt.subplot(2, 1, i + 1)
        plt.step(t, u_temp[:, i], 'k', where='post')
        plt.ylabel('Flow  ' + str(i + 1) + ' [ml/s]')
        plt.suptitle('Control inputs', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()
    return mu_EM, var_EM