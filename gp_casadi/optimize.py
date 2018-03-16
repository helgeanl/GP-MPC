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
from scipy.spatial import distance

# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y, squaredist):
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
    n, D = ca.MX.size(X)
    ell = hyper[:D]
    sf2 = hyper[D]**2
    sn2 = hyper[D + 1]**2

    ell_s   = ca.SX.sym('ell', D)
    sf2_s   = ca.SX.sym('x', 1)
    
    squaredist_s = ca.SX.sym('Km', n, n*D)
    K_s = ca.SX(n, n) # ca.SX.sym('K', n, n)
    # Calculate covariance matrix
    for i in range(D):
        K_s = squaredist_s[:, (i  * n):(i + 1) * n] * 1 / ell_s[i]**2 + K_s
    K_s = sf2_s * ca.SX.exp(-.5 * K_s)
    fill_K = ca.Function('K', [squaredist_s, ell_s, sf2_s], [K_s])
    K = fill_K(squaredist, ell, sf2)
    #sqrtK = sqrt(K)
    #expK = exp(-sqrtK)
    
    #k = sf2_s * ca.exp(-.5 * ca.sum2((x - z)**2 / ell_s**2))
    #SEard = ca.Function('SEard', [x, z, ell_s, sf2_s], [k])
    #R = SEard.mapaccum('R',n).mapaccum('R',n)
    #K = ca.MX(n, n)
    #for i in range(n):
    #    for j in range(n):
            #dist = ca.sum2((X[i, :] - X[j, :])**2 * ell2)
            #K[i, j] = sf2 * ca.SX.exp(-.5 * dist)
    #        K[i, j] = SEard(X[i, :].T, X[j, :].T, ell2, sf2)

    #K = K + sn2 * ca.MX.eye(n)
    #K = (K + K.T) * 0.5   # Make sure matrix is symmentric

    A = ca.SX.sym('A', ca.MX.size(K))
    cholesky = ca.Function('cholesky', [A], [ca.chol(A).T])
    L = cholesky(K)

    B = 2 * ca.sum1(ca.SX.log(ca.diag(A)))
    log_determinant = ca.Function('log_det', [A], [B])
    log_detK = log_determinant(K)

    Y_s = ca.SX.sym('Y', ca.MX.size(Y))
    L_s = ca.SX.sym('L', ca.Sparsity.lower(n))
    sol = ca.Function('sol', [L_s, Y_s], [ca.solve(L_s, Y_s)])
    invLy = sol(L, Y)
    #invLy = ca.solve(L, Y)
    #alpha = ca.solve(L.T, invLy)
    
    invLy_s = ca.SX.sym('invLy', ca.MX.size(invLy))
    sol2 = ca.Function('sol2', [L_s, invLy_s],[ca.solve(L_s.T, invLy_s)])
    alpha = sol2(L, invLy)
    
    alph = ca.SX.sym('alph', ca.MX.size(alpha))
    det = ca.SX.sym('det')
    NLL = ca.Function('NLL', [Y_s, alph, det], [0.5 * ca.mtimes(Y_s.T, alph) + 0.5 * det])
    #NLL = 0.5 * ca.mtimes(Y.T, alpha) + 0.5 * logK  # - log_priors
    return NLL(Y, alpha, log_detK)


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
    lb[:D]    = stdX / 20
    ub[:D]    = stdX * 20
    lb[D]     = stdF / 20
    ub[D]     = stdF * 20
    lb[D + 1] = 10**-5 / 10
    ub[D + 1] = 10**-5 * 10

    if meanFunc == 'const':
        lb[D + 2] = meanF / 5
        ub[D + 2] = meanF * 5

    # NLP solver options
    opts = {}
    opts["expand"] = True
    #opts["max_iter"] = 100
    opts["verbose"] = False
    opts["ipopt.print_level"] = 0
    opts["ipopt.tol"] = 1e-12
    #opts["ipopt.linear_solver"] = "ma27"
    #opts["ipopt.hessian_approximation"] = "limited-memory"
    multistart = 1

    timeNLLStart = time.time()

    # Symbols
    hyp_s = ca.MX.sym('hyp', 1, D + h)
    ell_s   = ca.SX.sym('ell', D)
    sf2_s   = ca.SX.sym('sf2', 1)
    t   = ca.SX.sym('t')
    
    Y_s   = ca.MX.sym('Y', n, 1)
    X_s  = ca.MX.sym('X', n, D)
    squaredist_s = ca.SX.sym('squaredist', n, n * D)
    #squaredist_s2 = ca.MX.sym('squaredist', n, n * D)

    squaredist = np.zeros((n, n * D))
    for i in range(D):
        d = distance.pdist(X[:, i].reshape(n, 1), 'sqeuclidean')
        squaredist[:, (i * n):(i + 1) * n] = distance.squareform(d)


    ############################################################
    ell = hyp_s[:D]
    sf2 = hyp_s[D]**2
    sn2 = hyp_s[D + 1]**2
    squaredist = ca.MX(squaredist)
    Y = ca.MX(Y)
    X = ca.MX(X)
    
    K_init = ca.SX.sym('K1', n, n)
    K_s = ca.SX.sym('K', n, n)
    # Calculate covariance matrix
    K_s = K_init
    for i in range(D):
        K_s = squaredist_s[:, (i  * n):(i + 1) * n] / ell_s[i]**2 + K_s
    K_s = sf2_s * ca.SX.exp(-.5 * K_s)
    K = ca.MX.eye(n)
    fill_K = ca.Function('K2', [squaredist_s, ell_s, sf2_s, K_init], [K_s])

    K1 = ca.MX(n, n)
    K = fill_K(squaredist, ell, sf2, K1)
    #sqrtK = sqrt(K)
    #expK = exp(-sqrtK)

    K = K + sn2 * ca.MX.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    
    
    
    A = ca.SX.sym('A', ca.MX.size(K))
    cholesky = ca.Function('cholesky', [A], [ca.chol(A).T])
    L = cholesky(K)

    B = 2 * ca.sum1(ca.SX.log(ca.diag(A)))
    log_determinant = ca.Function('log_det', [A], [B])
    log_detK = log_determinant(K)

    Y_s = ca.SX.sym('Y', ca.MX.size(Y))
    L_s = ca.SX.sym('L', ca.Sparsity.lower(n))
    #linear_solver = ca.Linsol("linear_solver", "csparse", L_s.Sparsity())
    sol = ca.Function('sol', [L_s, Y_s], [ca.solve(L_s, Y_s)])
    invLy = sol(L, Y)
    

    #invLy = ca.solve(L, Y)
    #alpha = ca.solve(L.T, invLy)
    
    invLy_s = ca.SX.sym('invLy', ca.MX.size(invLy))
    Lt_s = ca.SX.sym('Lt', ca.Sparsity.upper(n))
    #linear_solver = ca.Linsol("linear_solver", "csparse", Lt_s)
    sol2 = ca.Function('sol2', [Lt_s, invLy_s],[ca.solve(Lt_s, invLy_s)])
    alpha = sol2(L.T, invLy)
    
    alph = ca.SX.sym('alph', ca.MX.size(alpha))
    det = ca.SX.sym('det')
    
    NLL = ca.Function('NLL', [Y_s, alph, det], [0.5 * ca.mtimes(Y_s.T, alph) + 0.5 * det])
    #NLL = 0.5 * ca.mtimes(Y.T, alpha) + 0.5 * logK  # - log_priors
    #NLL(Y, alpha, log_detK)
    
    ##########################################################
    #NLL_func = ca.Function('NLL', [hyp_s, X_s, Y_s, squaredist_s],
    #                       [calc_NLL(hyp_s, X_s, Y_s, squaredist_s)])
    #nlp = {'x': hyp_s, 'f': NLL_func(hyp_s, X, Y, squaredist)}
    #nlp = {'x': hyp_s, 'f': calc_NLL(hyp_s, ca.MX(X), ca.MX(Y), ca.MX(squaredist))}
    nlp = {'x': hyp_s, 'p':t, 'f':NLL(Y, alpha, log_detK)}
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
        hyper_init[i, D + 1] = 10**-5        # Noise

        if meanFunc == 'const':
            hyper_init[i, D + 2] = meanF     # Mean of F
        timeSolveStart = time.time()

        #res = res = Solver(x0=hyper_init[0, :], lbx=lb, ubx=ub, p=params_)
        res = res = Solver(x0=hyper_init[0, :], lbx=lb, ubx=ub, p=1)
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
