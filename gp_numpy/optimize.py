# -*- coding: utf-8 -*-
"""
# Copyright (c) 2018
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014")

import pyDOE
import numpy as np
from scipy.optimize import minimize


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
    #K   = np.zeros((n, n))
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
    multistart = 1

    hyper_init = pyDOE.lhs(D + h, samples=multistart)

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
    print("numpy")
    return hyp_opt
