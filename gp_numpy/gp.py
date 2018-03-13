# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 16:04:36 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014.01.10")

import pyDOE
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Simulation_data_Four_Tank import sim_system

dir_data = '../data/'
dir_parameters = '../parameters/'


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    #dist = 0
    #for i in range(len(x)):
    #    dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    dist = np.sum((x - z)**2 / ell**2)
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
    #hyper = np.exp(hyper)
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
        print(L)
        K = K + np.eye(3) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))

    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = 0.5 * np.dot(Y.T, alpha) + 0.5 * logK
    return NLL


def train_gp(X, Y, state):
    n, D = X.shape
    stdX = np.std(X[:, state])
    stdF = np.std(Y)
    #meanF = np.mean(Y)
    h = 2
    lb = np.zeros(D + h)
    ub = np.zeros(D + h)
    lb[:D] = stdX / 10
    ub[:D] = stdX * 10
    lb[D]  = stdF / 10
    ub[D]  = stdF * 10
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
        hyper_init[i, D + 1] = 10**-3      # Noise
        #hyper_init[i, D + 2] = meanF                # Mean of F

        res = minimize(calc_NLL, hyper_init[i, :], args=(X, Y),
                       method='SLSQP', options=options, bounds=bounds, tol=10**-12)
        obj[i] = res.fun
        hyp_opt_loc[i, :] = res.x
    hyp_opt = hyp_opt_loc[np.argmin(obj)]

    return hyp_opt


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


if __name__ == "__main__":
    X = np.loadtxt(dir_data + 'X_matrix_tank')
    Y = np.loadtxt(dir_data + 'Y_matrix_tank')
    optimize = True
    n, D = X.shape  # Number of sampling points and inputs
    E = Y.shape[1]  # Number of outputs

    invK = np.zeros((E, n, n))
    h = 2
    #K1, K2 = train_gp_casadi(X, Y[:, 0], 0)
   # X, Y = standardize(X, Y, [0, 0, 0, 0,0,0], [80, 80, 80, 80,100,100])

    #lbx = np.array([.0, .0, .0, .0, .0, .0])
    #ubx = np.array([80., 80., 80., 80., 100., 100.])
    #lby = np.array([0., 0., 0., 0.])
    #uby = np.array([80., 80., 80., 80.])
    #X = scale_min_max(X, lbx, ubx)
    #meanY = np.mean(Y)
    #stdY = np.std(Y)
    #Y = scale_min_max(Y, lby, uby)
    #Y = scale_gaussian(Y, meanY, stdY)


    if optimize:
        hyper = np.zeros((E, D + h))
        n, D = X.shape
        for i in range(E):
            hyper[i, :] = train_gp(X, Y[:, i], i)
            K = calc_cov_matrix(X, hyper[i, :D], hyper[i, D]**2)
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

    u = np.array([50., 50.])
    x0 = np.array([10., 20., 30., 40.])
    mu, var  = predict(X, Y, invK, hyper, x0, u)