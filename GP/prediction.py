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


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    for i in range(len(x)):
        dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    return sf2 * np.exp(-.5 * dist)


def calc_cov_matrix(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def cov_gradient(x, z, ell, sf2):
    print("grad")


def gp(hyp, invK, X, Y, u):
    ell = hyp[0:-3]
    sf2 = hyp[-3]**2
    npoints = len(X)
    kss = covSEard(u, u, ell, sf2)
    ks = np.zeros(npoints)
    for i in range(npoints):
        ks[i] = covSEard(X[i, :], u, ell, sf2)
    ksK = np.dot(ks.T, invK)
    mu = np.dot(ksK, Y)
    s2 = kss - np.dot(ksK, ks)
    return mu, s2


def gp_casadi(hyp, invK, X, Y, z):
    ell = ca.MX(hyp[0:-3])
    sf2 = ca.MX(hyp[-3]**2)
    #X = ca.MX(X)
    #Y = ca.MX(Y)
    #u = ca.MX(u)
    npoints = len(X)
    kss = covSEard(z, z, ell, sf2)
    ks = ca.MX.zeros(npoints, 1)

    for i in range(npoints):
        ks[i] = covSEard(X[i, :], z, ell, sf2)
    #ks = repmat()
    ksK = ca.mtimes(ks.T, invK)

    mu = ca.mtimes(ksK, Y)
    s2 = kss - ca.mtimes(ksK, ks)

    return mu, s2


# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y, sign=-1.0):
    """ Objective function """
    # Calculate NLL
    n, D = X.shape
    #h1 = D + 1      # number of hyperparameters from covariance
    #h2 = 1          # number of hyperparameters from likelihood
    #h3 = 1
    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    K   = np.zeros((n, n))

    K = calc_cov_matrix(X, ell, sf2)
    K = K + lik * np.eye(n)

    K = (K + K.T) / 2   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(3) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))

    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK + n / 2 * np.log(2 * np.pi)
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
    meanF = np.mean(Y)
    lb = np.zeros(D + 3)
    ub = np.zeros(D + 3)
    lb[:D] = stdX / 10
    ub[:D] = stdX * 10
    lb[D] = stdF / 10
    ub[D] = stdF * 10
    lb[D + 1] = 10**-3 / 10
    ub[D + 1] = 10**-3 * 10
    lb[D + 2] = meanF / 4
    ub[D + 2] = meanF * 4
    bounds = np.hstack((lb.reshape(D + 3, 1), ub.reshape(D + 3, 1)))

    options = {'disp': True, 'maxiter': 100}
    multistart = 5

    hyper_init = pyDOE.lhs(D + 3, samples=multistart, criterion='maximin')

    # Scale control inputs to correct range
    obj = np.zeros((multistart, 1))
    hyp_opt_loc = np.zeros((multistart, D + 3))
    for i in range(multistart):
        hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
        hyper_init[i, D + 1] = 10**-3        # Noise
        hyper_init[i, D + 2] = meanF                # Mean of F

        res = minimize(calc_NLL, hyper_init[i, :], args=(X, Y, -1.0,),
                       method='SLSQP', options=options, bounds=bounds, tol=10**-12)
        obj[i] = res.fun
        hyp_opt_loc[i, :] = res.x
    hyp_opt = hyp_opt_loc[np.argmin(obj)]

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

    D = number_of_inputs
    F = ca.MX.sym('F', npoints, number_of_states)
    Xt = ca.MX.sym('X', npoints, number_of_inputs)
    hyp = ca.MX.sym('hyp', hyper.shape)
    z = ca.MX.sym('z', z_n.shape)
    cov = ca.MX.sym('cov', covariance.shape)

    gp_EM = ca.Function('gp', [Xt, F, hyp, z, cov], GP_noisy_input(invK, Xt, F, hyp, D, z, cov))

    for dt in range(simPoints):
        mu, cov = gp_EM.call([X, Y, hyper, z_n, covariance])
        mu, cov = mu.full(), cov.full()
        mu.shape, cov.shape = (number_of_states), (number_of_states, number_of_states)
        mu_n[dt, :], var_n[dt, :] = mu, np.diag(cov)
        z_n = ca.vertcat(mu, u).T
        covariance[:number_of_states, :number_of_states] = cov

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
        plt.plot(t, Y_sim[:, i], 'b-')
        plt.plot(t, mu, 'r--')
        sd = np.sqrt(var_n[:, i])
        plt.gca().fill_between(t.flat, mu - 2 * sd, mu + 2 * sd, color="#dddddd")
        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(['Simulation', 'Prediction', '95% conf interval'])
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
    plt.show()
    return mu_n, var_n


if __name__ == "__main__":
    X = np.loadtxt('../Data/' + 'X_matrix_tank')
    Y = np.loadtxt('../Data/' + 'Y_matrix_tank')
    #hyper = np.loadtxt('Parameters/' + 'hyper_opt', delimiter=',')
    npoints = X.shape[0]
    invK = np.zeros((4, npoints, npoints))
    hyper = np.zeros((4, 9))
    n, D = X.shape
    for i in range(4):
        #invK[i, :, :] = np.loadtxt('invK' + str(i + 1), delimiter=',')

        hyper[i, :] = train_gp(X, Y[:, i], i)
        K = calc_cov_matrix(X, hyper[i, :D], hyper[i, D]**2)
        invK[i, :, :] = np.linalg.inv(K)
        np.savetxt('../Parameters/' + 'invK' + str(i + 1), invK[:, :, i], delimiter=',')
    np.savetxt('../Parameters/' + 'hyper_opt', hyper, delimiter=',')

    u = np.array([50, 50])
    x0 = np.array([10, 20, 30, 40])
    mu, var  = predict_casadi(X, Y, invK, hyper, x0, u)
    mu2, var2  = predict(X, Y, invK, hyper, x0, u)
