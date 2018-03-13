# -*- coding: utf-8 -*-
"""
Optimize hyperparameters
@author: Helge-André Langåker
"""

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014.01.10")

import pyDOE
import numpy as np
import casadi as ca


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
    ell = hyper[:D]
    sf2 = hyper[D]**2
    lik = hyper[D + 1]**2
    if ca.SX.size(hyper)[0] == 9:
        m   = hyper[D + 2]
    else:
        m = 0
    K   = ca.SX.zeros(n, n)

    for i in range(n):
        for j in range(n):
            dist = ca.sum2((X[i, :] - X[j, :])**2 / ell**2)
            K[i, j] = sf2 * ca.SX.exp(-.5 * dist)

    K = K + lik * ca.SX.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric

   # A = ca.SX.sym('A', ca.MX.size(K))
    #L = ca.chol(A)      # Should check for PD !!!
    #cholesky = ca.Function('Cholesky', [A], [ca.chol(A)])
    #L = cholesky(K).T
    L = ca.chol(K).T
    #L = ca.chol(K)[1]
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
    n, D = X.shape
    #number_of_states = len(invK)
    #number_of_inputs = X.shape[1]

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
    #opts["linear_solver"] = "ma27"
    # opts["hessian_approximation"] = "limited-memory"
    multistart = 2

    hyper_init = pyDOE.lhs(D + h, samples=multistart, criterion='maximin')

    F   = ca.MX.sym('F', n, 1)
    Xt  = ca.MX.sym('X', n, 6)
    hyp = ca.MX.sym('hyp', (1, D + h))

    NLL_func = ca.Function('NLL', [hyp], [calc_NLL(hyp, Xt, F)])
    #NLL_SX = NLL_mx.expand()
    #NLL = {'x': hyp, 'f': calc_NLL(hyp, Xt, F)}
    NLL = {'x': hyp, 'f': NLL_func(hyp, ca.MX(X), ca.MX(Y))}
    Solver = ca.nlpsol('Solver', 'ipopt', NLL, opts)

    # Scale control inputs to correct range
    obj = np.zeros((multistart, 1))
    hyp_opt_loc = np.zeros((multistart, D + h))
    hyp_opt = np.zeros((D + h))

    for i in range(multistart):
        hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
        hyper_init[i, D + 1] = 10**-3        # Noise
        if meanFunc == 'const':
            hyper_init[i, D + 2] = meanF     # Mean of F
        res = Solver(x0=hyper_init[i, :], lbx=lb, ubx=ub)
        obj[i] = res['f']
        hyp_opt_loc[i, :] = res['x']
    hyp_opt[:D + h] = hyp_opt_loc[np.argmin(obj)]

    return hyp_opt
