# -*- coding: utf-8 -*-
"""
Optimize hyperparameters
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014")

import pyDOE
import numpy as np
import casadi as ca
from scipy.spatial import distance


# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, Y, squaredist):
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

    n = ca.MX.size(Y)[0]
    D = ca.MX.size(hyper)[1] - 2
    ell = hyper[:D]
    sf2 = hyper[D]**2
    sn2 = hyper[D + 1]**2

    # Calculate covariance matrix
    K_s = ca.SX.sym('K_s', n, n)
    sqdist = ca.SX.sym('sqd', n, n)
    elli = ca.SX.sym('elli')
    ki = ca.Function('ki', [sqdist, elli, K_s], [sqdist / elli**2 + K_s])
    K1 = ca.MX(n, n)
    for i in range(D):
        K1 = ki(squaredist[:, (i  * n):(i + 1) * n], ell[i], K1)

    sf2_s   = ca.SX.sym('sf2')
    exponent   = ca.SX.sym('exp', n, n)
    K_exp = ca.Function('K', [exponent, sf2_s], [sf2_s * ca.SX.exp(-.5 * exponent)])
    K2 = K_exp(K1, sf2)

    K = K2 + sn2 * ca.MX.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric

    A = ca.SX.sym('A', ca.MX.size(K))
    cholesky = ca.Function('cholesky', [A], [ca.chol(A).T])
    L = cholesky(K)

    B = 2 * ca.sum1(ca.SX.log(ca.diag(A)))
    log_determinant = ca.Function('log_det', [A], [B])
    log_detK = log_determinant(L)

    Y_s = ca.SX.sym('Y', ca.MX.size(Y))
    L_s = ca.SX.sym('L', ca.Sparsity.lower(n))
    sol = ca.Function('sol', [L_s, Y_s], [ca.solve(L_s, Y_s)])
    invLy = sol(L, Y)

    invLy_s = ca.SX.sym('invLy', ca.MX.size(invLy))
    sol2 = ca.Function('sol2', [L_s, invLy_s], [ca.solve(L_s.T, invLy_s)])
    alpha = sol2(L, invLy)

    alph = ca.SX.sym('alph', ca.MX.size(alpha))
    det = ca.SX.sym('det')
    NLL = ca.Function('NLL', [Y_s, alph, det], [0.5 * ca.mtimes(Y_s.T, alph) + 0.5 * det])
    return NLL(Y, alpha, log_detK)


def train_gp(X, Y, meanFunc='zero'):
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
    num_outputs = Y.shape[1]


    if meanFunc == 'zero':
        h = 2
    else:
        h = 3
    
    # Create solver
    Y_s          = ca.MX.sym('Y', n)
    hyp_s        = ca.MX.sym('hyp', 1, D + h)
    squaredist_s = ca.MX.sym('sqdist', n, n * D)
    param_s = ca.horzcat(squaredist_s, Y_s)

    NLL_func = ca.Function('NLL', [hyp_s, Y_s, squaredist_s],
                           [calc_NLL(hyp_s, Y_s, squaredist_s)])
    nlp = {'x': hyp_s, 'f': NLL_func(hyp_s, Y_s, squaredist_s), 'p': param_s}
    
    # NLP solver options
    opts = {}
    opts["expand"]              = True
    #opts["ipopt.max_iter"]      = 100
    opts["verbose"]             = False
    opts["ipopt.print_level"]   = 0
    #opts["ipopt.tol"]           = 1e-12
    #opts["ipopt.linear_solver"] = "ma27"
    #opts["ipopt.hessian_approximation"] = "limited-memory"
    Solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    hyp_opt = np.zeros((num_outputs, D + h))
    for output in range(num_outputs):
        print('Optimizing hyperparameters for state ' + str(output))
        stdX      = np.std(X)
        stdF      = np.std(Y[:, output])
        meanF     = np.mean(Y)
        lb        = np.zeros(D + h)
        ub        = np.zeros(D + h)
        lb[:D]    = stdX / 20
        ub[:D]    = stdX * 20
        lb[D]     = stdF / 20
        ub[D]     = stdF * 20
        lb[D + 1] = 10**-5 / 10
        ub[D + 1] = 10**-5 * 10
    
        if meanFunc == 'const':
            lb[D + 2] = meanF / 5
            ub[D + 2] = meanF * 5
    
        multistart = 1
    
        squaredist = np.zeros((n, n * D))
        for i in range(D):
            d = distance.pdist(X[:, i].reshape(n, 1), 'sqeuclidean')
            squaredist[:, (i * n):(i + 1) * n] = distance.squareform(d)
        param = ca.horzcat(squaredist, Y[:, output])
        
        obj = np.zeros((multistart, 1))
        hyp_opt_loc = np.zeros((multistart, D + h))
        
        hyper_init = pyDOE.lhs(D + h, samples=multistart)
    
        for i in range(multistart):
            hyper_init[i, :] = hyper_init[i, :] * (ub - lb) + lb
            hyper_init[i, D + 1] = 10**-5        # Noise
    
            if meanFunc == 'const':
                hyper_init[i, D + 2] = meanF
    
            res = res = Solver(x0=hyper_init[0, :], lbx=lb, ubx=ub, p=param)
            obj[i] = res['f']
            hyp_opt_loc[i, :] = res['x']
    
        hyp_opt[output, :D + h] = hyp_opt_loc[np.argmin(obj)]


    return hyp_opt
