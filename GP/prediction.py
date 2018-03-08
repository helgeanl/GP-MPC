# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: helgeanl
"""

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
path.append(r"../Simulation")

#from scipy.stats import norm as norms
#from pylab import *
import matplotlib.pyplot as plt
import numpy as np
#from numpy.linalg import inv, cholesky
#import matplotlib.pyplot as plt
#import time
#from math import sqrt
from Simulation_data_Four_Tank import sim_system
from noisyGP import GP_noisy_input
import casadi as ca
import pyDOE
#from scipy.stats.distributions import lognorm
from scipy.optimize import minimize


dir_data = '../Data/'
dir_parameters = '../Parameters/'


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    #dist = 0
    #for i in range(len(x)):
    #    dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    dist = np.sum((x - z)**2 / ell**2)
    return sf2 * np.exp(-.5 * dist)


def covSEard2(x, z, ell, sf2):
    """ GP squared exponential kernel """
    #dist = 0
    #for i in range(ca.SX.size(x)[0]):
    #    dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    dist = ca.sum2((x - z)**2 / ell**2)
    return sf2 * ca.MX.exp(-.5 * dist)


def calc_cov_matrix(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def calc_cov_matrix_casadi(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = ca.SX.size(X)
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (ca.sum2(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def cov_gradient(x, z, ell, sf2):
    print("grad")


def gp(hyp, invK, X, Y, u):
    n, D = X.shape
    ell = hyp[:D]
    sf2 = hyp[D]**2
    #m   = hyp[D + 2]
    kss = covSEard(u, u, ell, sf2)
    ks = np.zeros(n)
    for i in range(n):
        ks[i] = covSEard(X[i, :], u, ell, sf2)
    ksK = np.dot(ks.T, invK)
    mu = np.dot(ksK, Y)
    s2 = kss - np.dot(ksK, ks)
    return mu, s2


def gp_casadi(invK, hyp, X, Y, z):
    E = len(invK)
    n = ca.MX.size(X[:, 1])[0]
    D = ca.MX.size(X[1, :])[1]

    mean  = ca.MX.zeros(E, 1)
    var  = ca.MX.zeros(E, 1)
    for a in range(E):
        ell = ca.MX(hyp[a, 0:D])
        sf2 = ca.MX(hyp[a, D]**2)
        m = 0 #hyp[a, D + 2]
        kss = covSEard2(z, z, ell, sf2)
        ks = ca.MX.zeros(n, 1)

        for i in range(n):
            ks[i] = covSEard2(X[i, :] - m, z - m, ell, sf2)
        #ks = repmat()
        ksK = ca.mtimes(ks.T, invK[a])

        mean[a] = ca.mtimes(ksK, Y[:, a] - m) + m
        var[a] = kss - ca.mtimes(ksK, ks)

    return mean, var


def gp_taylor_approx(invK, X, F, hyper, D, inputmean, inputcov):
    #hyper = ca.MX.log(hyper)
    E     = len(invK)
    n     = ca.MX.size(F[:, 1])[0]
    mean  = ca.MX.zeros(E, 1)
    var = ca.MX.zeros(E, 1)
    v     = X - ca.repmat(inputmean, n, 1)
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
            ks[i] = covSEard2(X[i, :], inputmean, ell, sf2)

        invKks = ca.mtimes(iK, ks)
        mean[a] = ca.mtimes(ks.T, alpha)
        var[a] = kss - ca.mtimes(ks.T, invKks)
        d_mean[a] = ca.mtimes(ca.transpose(w[a] * v[:, a] * ks), alpha)

    for d in range(D)
        for e in range(D):
            dd_var1a = ca.mtimes(ca.transpose(v[:, d] * ks), iK)
            dd_var1b = ca.mtimes(dd_var1a, v[e] * ks)
            dd_var2 = ca.mtimes(ca.transpose(v[d] * v[e] * ks), invKks)
            dd_var[d, e] = -2 * w[d] * w[e] * (dd_var1b + dd_var2)
            if d == e:
                dd_var[d, e] = dd_var[d, e] + 2 * w[d] * (kss - var[d])

    # covariance with noisy input
    for a in range(E):
        covar1 = ca.mtimes(d_mean, d_mean.T)
        print(ca.MX.size(covar1))
        covariance[a, a] = var[a] + ca.trace(ca.mtimes(inputcov, .5 * dd_var + covar1))

    return [mean, covariance]


# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y):
    """ Objective function """
    # Calculate NLL
    n, D = X.shape
    #h1 = D + 1      # number of hyperparameters from covariance
    #h2 = 1          # number of hyperparameters from likelihood
    #h3 = 1
    hyper = np.exp(hyper)
    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    #m   = hyper[D + 2]
    K   = np.zeros((n, n))
    K = calc_cov_matrix(X, ell, sf2)
    K = K + lik * np.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(3) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))

    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK + n * 0.5 * np.log(2 * np.pi)
    return NLL


def calc_NLL_casadi(hyper, X, Y):
    """ Objective function """
    # Calculate NLL
    n, D = ca.MX.size(X)

    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    m   = hyper[D + 2]
    K   = ca.MX.zeros(n, n)

    for i in range(n):
        for j in range(n):
            K[i, j] = covSEard2(X[i, :] - m, X[j, :] - m, ell, sf2)

    K = K + lik * ca.MX.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric

    A = ca.SX.sym('A', ca.MX.size(K))
    #L = ca.chol(A)      # Should check for PD !!!
    cholesky = ca.Function('Cholesky', [A], [ca.chol(A)])
    L = cholesky(K).T

    logK = 2 * ca.sum1(ca.MX.log(ca.fabs(ca.diag(L))))

    invLy = ca.solve(L, Y - m)
    alpha = ca.solve(L.T, invLy)
    NLL = 0.5 * ca.mtimes(Y.T - m, alpha) + 0.5 * logK + n * 0.5 * ca.log(2 * ca.pi)   # - log_priors
    return NLL


def calc_dNLL(x, sign=1.0):
    """ Derivative of objective function """
    dfdx0 = sign * (-2 * x[0] + 2 * x[1] + 2)
    dfdx1 = sign * (2 * x[0] - 4 * x[1])
    return np.array([dfdx0, dfdx1])


def train_gp(X, Y, state):
    n, D = X.shape
    stdX = np.std(X[:, state])
    stdF = np.std(Y)
    #meanF = np.mean(Y)
    h = 2
    lb = np.zeros(D + h)
    ub = np.zeros(D + h)
    lb[:D] = stdX / 20
    ub[:D] = stdX * 20
    lb[D]  = stdF / 5
    ub[D]  = stdF * 5
    lb[D + 1] = 10**-3 / 10
    ub[D + 1] = 10**-3 * 10
    #lb[D + 2] = meanF / 10
    #ub[D + 2] = meanF * 10
    bounds = np.hstack((lb.reshape(D + h, 1), ub.reshape(D + h, 1)))

    options = {'disp': True, 'maxiter': 100}
    multistart = 10

    hyper_init = pyDOE.lhs(D + h, samples=multistart, criterion='maximin')

    # Scale control inputs to correct range
    obj = np.zeros((multistart, 1))
    hyp_opt_loc = np.zeros((multistart, D + h))
    for i in range(multistart):
        hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
        hyper_init[i, D + 1] = 10**-3        # Noise
        #hyper_init[i, D + 2] = meanF                # Mean of F

        res = minimize(calc_NLL, np.log(hyper_init[i, :]), args=(X, Y),
                       method='SLSQP', options=options, bounds=np.log(bounds), tol=10**-12)
        obj[i] = res.fun
        hyp_opt_loc[i, :] = np.exp(res.x)
    hyp_opt = hyp_opt_loc[np.argmin(obj)]

    return hyp_opt


def train_gp_casadi(X, Y, state, meanFunc=0):
    n, D = X.shape
    #number_of_states = len(invK)
    #number_of_inputs = X.shape[1]

    stdX = np.std(X[:, state])
    stdF = np.std(Y)
    meanF = np.mean(Y)
    lb = np.zeros(D + 3)
    ub = np.zeros(D + 3)
    lb[:D]    = stdX / 3
    ub[:D]    = stdX * 3
    lb[D]     = stdF / 3
    ub[D]     = stdF * 3
    lb[D + 1] = 10**-3 / 10
    ub[D + 1] = 10**-3 * 10

    if meanFunc == 0:
        lb[D + 2] = 0
        ub[D + 2] = 100
    elif meanFunc == 'const':
        lb[D + 2] = meanF / 5
        ub[D + 2] = meanF * 5

    #bounds = np.hstack((lb.reshape(D + 3, 1), ub.reshape(D + 3, 1)))

    # NLP solver options
    opts = {}
    #opts["expand"] = True
    #opts["max_iter"] = 100
    opts["verbose"] = True
    # opts["linear_solver"] = "ma57"
    # opts["hessian_approximation"] = "limited-memory"
    multistart = 2

    hyper_init = pyDOE.lhs(D + 3, samples=multistart, criterion='maximin')

    F   = ca.MX(Y)  # ca.MX.sym('F', npoints, 1)
    Xt  = ca.MX(X)  # ca.MX.sym('X', npoints, 6)
    hyp = ca.MX.sym('hyp', (1, D + 3))

    #NLL = ca.Function('NLL', [hyp, Xt, F], [calc_NLL_casadi(hyp, Xt, F)])

    NLL = {'x': hyp, 'f': calc_NLL_casadi(hyp, Xt, F)}
    Solver = ca.nlpsol('Solver', 'ipopt', NLL, opts)
    #return NLL.call([hyper[0,:],X,Y]), calc_NLL(hyper[0,:], X, Y)

    # Scale control inputs to correct range
    obj = np.zeros((multistart, 1))
    hyp_opt_loc = np.zeros((multistart, D + 3))
    for i in range(multistart):
        hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
        hyper_init[i, D + 1] = 10**-3        # Noise
        if meanFunc == 0:
            hyper_init[i, D + 2] = 0
        elif meanFunc == 'const':
            hyper_init[i, D + 2] = meanF          # Mean of F
        res = Solver(x0=hyper_init[i, :], lbx=lb, ubx=ub)
        obj[i] = res['f']
        hyp_opt_loc[i, :] = res['x']
    hyp_opt = res['x']  # hyp_opt_loc[np.argmin(obj)]

    # Zero mean
    #if mean == 0:
    #    hyp_opt[:, D + 2] = 0

    return hyp_opt

#res = minimize(func, [-1.0,1.0], args=(-1.0,), jac=func_deriv,
#               constraints=cons, method='SLSQP', options={'disp': True})


def predict(X, Y, invK, hyper, x0, u):
    # Predict future
    #npoints = X.shape[0]
    number_of_states = len(invK)

    simTime = 300
    deltat = 3
    simPoints = simTime / deltat

    x_n = np.concatenate([x0, u])
    mu_n = np.zeros((simPoints, number_of_states))
    s_n = np.zeros((simPoints, number_of_states))

    for dt in range(simPoints):
        for state in range(number_of_states):
            mu_n[dt, state], s_n[dt, state] = gp(hyper[state, :], invK[state, :, :],
                                                 X, Y[:, state], x_n)
        x_n = np.concatenate((mu_n[dt, :], u))

    t = np.linspace(0.0, simPoints, simPoints)
    u_matrix = np.zeros((simPoints, 2))
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]

    Y_sim = sim_system(x0, u_matrix, simTime, deltat)

    plt.figure()
    plt.clf()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        mu = mu_n[:, i]
        plt.plot(t, Y_sim[:, i], 'b-')
        plt.plot(t, mu, 'r--')
        sd = np.sqrt(s_n[:, i])
        plt.gca().fill_between(t.flat, mu - 2 * sd, mu + 2 * sd, color="#dddddd")
        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(['Simulation', 'Prediction', '95% conf interval'])
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
    plt.show()
    return mu_n, s_n


def predict_casadi(X, Y, invK, hyper, x0, u):
    # Predict future
    npoints = X.shape[0]
    number_of_states = len(invK)
    number_of_inputs = X.shape[1]

    simTime = 300
    deltat = 3
    simPoints = simTime / deltat

    z_n = np.concatenate([x0, u])
    z_n.shape = (1, number_of_inputs)
    mu_n = np.zeros((simPoints, number_of_states))
    var_n = np.zeros((simPoints, number_of_states))
    covariance = np.zeros((number_of_inputs, number_of_inputs))

    z_n2 = np.concatenate([x0, u])
    z_n2.shape = (1, number_of_inputs)
    mu_n2 = np.zeros((simPoints, number_of_states))
    var_n2 = np.zeros((simPoints, number_of_states))

    z_n3 = np.concatenate([x0, u])
    z_n3.shape = (1, number_of_inputs)
    mu3_n = np.zeros((simPoints, number_of_states))
    var3_n = np.zeros((simPoints, number_of_states))
    covariance2 = np.zeros((number_of_states, number_of_states))

    D = number_of_inputs
    F = ca.MX.sym('F', npoints, number_of_states)
    Xt = ca.MX.sym('X', npoints, number_of_inputs)
    hyp = ca.MX.sym('hyp', hyper.shape)
    z = ca.MX.sym('z', z_n.shape)
    cov = ca.MX.sym('cov', covariance.shape)
    cov2 = ca.MX.sym('cov2', covariance2.shape)

    gp_EM = ca.Function('gp', [Xt, F, hyp, z, cov], GP_noisy_input(invK, Xt, F, hyp, D, z, cov))
    gp_TA = ca.Function('gp_taylor_approx', [Xt, F, hyp, z, cov2], gp_taylor_approx(invK, Xt, F, hyp, D, z, cov2))
    gp_simple = ca.Function('gp_simple', [Xt, F, hyp, z], gp_casadi(invK, hyp, Xt, F, z))

    for dt in range(simPoints):
        mu, cov = gp_EM.call([X, Y, hyper, z_n, covariance])
        mu, cov = mu.full(), cov.full()
        mu.shape, cov.shape = (number_of_states), (number_of_states, number_of_states)
        mu_n[dt, :], var_n[dt, :] = mu, np.diag(cov)
        z_n = ca.vertcat(mu, u).T
        covariance[:number_of_states, :number_of_states] = cov

    for dt in range(simPoints):
        mu, var = gp_simple.call([X, Y, hyper, z_n2])
        mu, var = mu.full(), var.full()
        mu.shape, var.shape = (number_of_states), (number_of_states)
        mu_n2[dt, :], var_n2[dt, :] = mu, var
        z_n2 = ca.vertcat(mu, u).T

    covar2 = np.zeros((4, 4))
    for dt in range(simPoints):
        mu, cov = gp_TA.call([X, Y, hyper, z_n3, covar2])
        mu, cov = mu.full(), cov.full()
        mu.shape, cov.shape = (number_of_states), (number_of_states, number_of_states)
        mu3_n[dt, :], var3_n[dt, :] = mu, np.diag(cov)
        z_n3 = ca.vertcat(mu, u).T
        #covariance[:number_of_states, :number_of_states] = cov
        covar2 = cov

    t = np.linspace(0.0, simTime, simPoints)
    u_matrix = np.zeros((simPoints, 2))
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]
    Y_sim = sim_system(x0, u_matrix, simTime, deltat)

    plt.figure()
    plt.clf()
    for i in range(number_of_states):
        plt.subplot(2, 2, i + 1)
        mu = mu_n[:, i]
        mu2 = mu_n2[:, i]
        mu3 = mu3_n[:, i]

        sd = np.sqrt(var_n[:, i])
        plt.gca().fill_between(t.flat, mu - 2 * sd, mu + 2 * sd, color="#dddddd")
        sd2 = np.sqrt(var_n2[:, i])
        plt.gca().fill_between(t.flat, mu2 - 2 * sd2, mu2 + 2 * sd2, color="#bbbbbb")
        sd2 = np.sqrt(var3_n[:, i])
        plt.gca().fill_between(t.flat, mu3 - 2 * sd2, mu3 + 2 * sd2, color="#aaaaaa")
        #plt.errorbar(t, mu, yerr=2 * sd)
        plt.plot(t, Y_sim[:, i], 'b-')
        plt.plot(t, mu, 'r--')
        plt.plot(t, mu2, 'y--')
        plt.plot(t, mu3, 'k--')

        labels = ['Simulation', 'GP Excact moment', 'GP Mean Equivalence', '95% conf interval 1', '95% conf interval 2']
        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(labels)
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()
    return mu_n, var_n


if __name__ == "__main__":
    X = np.loadtxt(dir_data + 'X_matrix_tank')
    Y = np.loadtxt(dir_data + 'Y_matrix_tank')
    optimize = False
    n, D = X.shape
    E = Y.shape[1]
    #npoints = X.shape[0]
    invK = np.zeros((E, n, n))
    h = 2
    #K1, K2 = train_gp_casadi(X, Y[:, 0], 0)

    if optimize:
        hyper = np.zeros((E, D + h))
        n, D = X.shape
        for i in range(E):
            hyper[i, :] = train_gp(X, Y[:, i], i)  # train_gp_casadi(X, Y[:, i], i, meanFunc=0)
            K = calc_cov_matrix(X - hyper[i, D + h - 1], hyper[i, :D], hyper[i, D]**2)
            K = K + hyper[i, D + 1]**2 * np.eye(n)  # Add noise variance to diagonal
            K = (K + K.T) * 0.5   # Make sure matrix is symmentric
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definit, adding jitter!")
                K = K + np.eye(n) * 1e-8
                L = np.linalg.cholesky(K)
            invL = np.linalg.solve(L, np.eye(n))
            invK[i, :, :] = np.linalg.solve(L.T, invL)    # np.linalg.inv(K)
            np.savetxt(dir_parameters + 'invK' + str(i + 1), invK[i, :, :], delimiter=',')
        np.savetxt(dir_parameters + 'hyper_opt', hyper, delimiter=',')

    else:
        hyper = np.loadtxt(dir_parameters + 'hyper_opt', delimiter=',')
        for i in range(E):
            invK[i, :, :] = np.loadtxt(dir_parameters + 'invK' + str(i + 1), delimiter=',')
            #hyper[i, -1] = 0  # np.mean(Y[:, i])

    u = np.array([50, 50])
    x0 = np.array([10, 20, 30, 40])
    mu, var  = predict_casadi(X, Y, invK, hyper, x0, u)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
