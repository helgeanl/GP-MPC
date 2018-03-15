# -*- coding: utf-8 -*-
"""
Optimize hyperparameters
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
import casadi as ca
import time


# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y):
    """ Objective function

    Calculate the negative log likelihood function using Casadi SX symbols.

    # Arguments:
        hyper: Array with hyperparameters [ell_1 .. ell_D sf sn], where D is the
            number of inputs to the GP.
        X: Training data matrix with inputs of size NxD.
        Y: Training data matrix with outpyts of size NxE, with E number of outputs.

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """

    # Calculate NLL
    n, D = ca.SX.size(X)
    print(D)
    ell = hyper[:D]
    sf2 = hyper[D]
    sn = hyper[D + 1]

    K   = ca.SX(n, n)
    for i in range(n):
        for j in range(n):
            dist = ca.sum2((X[i, :] - X[j, :])**2 * ell)
            K[i, j] = sf2 * ca.SX.exp(-.5 * dist)

    K = K + sn * ca.SX.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric

    L = ca.chol(K).T
    logK = 2 * ca.sum1(ca.SX.log(ca.fabs(ca.diag(L))))

    invLy = ca.solve(L, Y)
    alpha = ca.solve(L.T, invLy)
    NLL = 0.5 * ca.mtimes(Y.T, alpha) + 0.5 * logK  # - log_priors
    return NLL


def train_gp(X, Y, state, meanFunc='zero'):
    """ Train hyperparameters

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process.

    # Arguments:
        X: Training data matrix with inputs of size NxD, where D is the number
            of inputs to the GP.
        Y: Training data matrix with outpyts of size NxE, with E number of outputs.
        state: The index of the wanted state (#### REMOVE ####)
        meanFunc: String with the name of the wanted mean function.
            Possible options: 'zero', 'const', 'linear', 'polynomial'

    # Return:
        hyp_pot: Array with the optimal hyperparameters [ell_1 .. ell_D sf sn].
    """

    timeStart = time.time()

    n, D = X.shape
    stdX = np.std(X[:, state])
    stdF = np.std(Y)
    meanF = np.mean(Y)

    if meanFunc == 'zero':
        h = 2
    else:
        h = 3

    lb = np.zeros(D + h)
    ub = np.zeros(D + h)
    lb[:D]    = stdX / 10
    ub[:D]    = stdX * 10
    lb[D]     = stdF / 10
    ub[D]     = stdF * 10
    lb[D + 1] = 10**-3 / 10
    ub[D + 1] = 10**-3 * 10

    if meanFunc == 'const':
        lb[D + 2] = meanF / 5
        ub[D + 2] = meanF * 5

    # NLP solver options
    opts = {}
    opts["expand"] = False
    #opts["max_iter"] = 100
    opts["verbose"] = False
    opts["ipopt.print_level"] = 0
    opts["ipopt.tol"] = 1e-12
    #opts["ipopt.linear_solver"] = "ma27"
    #opts["ipopt.hessian_approximation"] = "limited-memory"
    multistart = 1

    timeNLLStart = time.time()

    # Symbols
    hyp = ca.SX.sym('hyp', 1, D + h)
    F   = ca.SX.sym('F', n, 1)
    Xt  = ca.SX.sym('X', n, D)
    NLL_func = ca.Function('NLL', [hyp, Xt, F], [calc_NLL(hyp, Xt, F)])
    nlp = {'x': hyp, 'f': NLL_func(hyp, X, Y)}

    timeNLLEnd = time.time()

    timeCreatingSolverStart = time.time()
    Solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    timeCreatingSolverEnd = time.time()

    obj = np.zeros((multistart, 1))
    hyp_opt_loc = np.zeros((multistart, D + h))
    hyp_opt = np.zeros((D + h))
    hyper_init = pyDOE.lhs(D + h, samples=multistart)

    for i in range(multistart):
        hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
        hyper_init[i, D + 1] = 10**-3        # Noise

        if meanFunc == 'const':
            hyper_init[i, D + 2] = meanF     # Mean of F
        timeSolveStart = time.time()

        #res = res = Solver(x0=hyper_init[0, :], lbx=lb, ubx=ub, p=params_)
        res = res = Solver(x0=hyper_init[0, :], lbx=lb, ubx=ub)
        obj[i] = res['f']
        hyp_opt_loc[i, :] = res['x']

    hyp_opt[:D + h] = hyp_opt_loc[np.argmin(obj)]

    timeEnd = time.time()
    print("------------------ Time [s] -------------------")
    print("Defining NLL: \t\t", (timeNLLEnd - timeNLLStart))
    print("Creating solver: \t", (timeCreatingSolverEnd - timeCreatingSolverStart))
    print("Time to before solve: \t", (timeSolveStart - timeStart))
    print("Time solving: \t\t", (timeEnd - timeSolveStart))
    print("Total time: \t\t", (timeEnd - timeStart))
    print("-----------------------------------------------")
    return hyp_opt
