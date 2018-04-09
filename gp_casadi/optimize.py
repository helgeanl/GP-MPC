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

import time
import pyDOE
import numpy as np
import casadi as ca
from scipy.spatial import distance
from . gp_functions import get_mean_function

# -----------------------------------------------------------------------------
# Optimization of hyperperameters as a constrained minimization problem
# -----------------------------------------------------------------------------
def calc_NLL(hyper, X, Y, squaredist, meanFunc='zero'):
    """ Objective function

    Calculate the negative log likelihood function using Casadi SX symbols.

    # Arguments:
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn], where Nx is the
            number of inputs to the GP.
        X: Training data matrix with inputs of size (N x Nx).
        Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.

    # Returns:
        NLL: The negative log likelihood function (scalar)
    """

    N, Nx = ca.MX.size(X)
    ell = hyper[:Nx]
    sf2 = hyper[Nx]**2
    sn2 = hyper[Nx + 1]**2

    m = get_mean_function(hyper, X, func=meanFunc)

    # Calculate covariance matrix
    K_s = ca.SX.sym('K_s',N, N)
    sqdist = ca.SX.sym('sqd', N, N)
    elli = ca.SX.sym('elli')
    ki = ca.Function('ki', [sqdist, elli, K_s], [sqdist / elli**2 + K_s])
    K1 = ca.MX(N, N)
    for i in range(Nx):
        K1 = ki(squaredist[:, (i  * N):(i + 1) * N], ell[i], K1)

    sf2_s   = ca.SX.sym('sf2')
    exponent   = ca.SX.sym('exp', N, N)
    K_exp = ca.Function('K', [exponent, sf2_s], [sf2_s * ca.SX.exp(-.5 * exponent)])
    K2 = K_exp(K1, sf2)

    K = K2 + sn2 * ca.MX.eye(N)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric

    A = ca.SX.sym('A', ca.MX.size(K))
    cholesky = ca.Function('cholesky', [A], [ca.chol(A).T])
    L = cholesky(K)

    B = 2 * ca.sum1(ca.SX.log(ca.diag(A)))
    log_determinant = ca.Function('log_det', [A], [B])
    log_detK = log_determinant(L)

    Y_s = ca.SX.sym('Y', ca.MX.size(Y))
    L_s = ca.SX.sym('L', ca.Sparsity.lower(N))
    sol = ca.Function('sol', [L_s, Y_s], [ca.solve(L_s, Y_s)])
    invLy = sol(L, Y - m(X))

    invLy_s = ca.SX.sym('invLy', ca.MX.size(invLy))
    sol2 = ca.Function('sol2', [L_s, invLy_s], [ca.solve(L_s.T, invLy_s)])
    alpha = sol2(L, invLy)

    alph = ca.SX.sym('alph', ca.MX.size(alpha))
    det = ca.SX.sym('det')
    NLL = ca.Function('NLL', [Y_s, alph, det], [0.5 * ca.mtimes(Y_s.T, alph) + 0.5 * det])
    return NLL(Y - m(X), alpha, log_detK)


def train_gp(X, Y, meanFunc='zero', hyper_init=None, multistart=1):
    """ Train hyperparameters

    Maximum likelihood estimation is used to optimize the hyperparameters of
    the Gaussian Process.

    # Arguments:
        X: Training data matrix with inputs of size (N x Nx), where Nx is the number
            of inputs to the GP.
        Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.
        meanFunc: String with the name of the wanted mean function.
            Possible options: 
                'zero':       m = 0 
                'const':      m = a
                'linear':     m(x) = aT*x + b
                'polynomial': m(x) = xT*diag(a)*x + bT*x + c
            
    # Return:
        hyp_pot: Array with the optimal hyperparameters [ell_1 .. ell_Nx sf sn].
    """

    N, Nx = X.shape
    Ny = Y.shape[1]
    # Counting mean function parameters
    if meanFunc == 'zero':
        h_m = 0
    elif meanFunc == 'const':
        h_m = 1
    elif meanFunc == 'linear':
        h_m = Nx + 1
    elif meanFunc == 'polynomial':
        h_m = 2 * Nx + 1
    else:
        raise NameError('No mean function called: ' + meanFunc)
    
    h_ell   = Nx    # Number of length scales parameters
    h_sf    = 1     # Standard deviation function
    h_sn    = 1     # Standard deviation noise
    num_hyp = h_ell + h_sf + h_sn + h_m

    # Create solver
    Y_s          = ca.MX.sym('Y', N)
    X_s          = ca.MX.sym('X', N, Nx)
    hyp_s        = ca.MX.sym('hyp', 1, num_hyp)
    squaredist_s = ca.MX.sym('sqdist', N, N * Nx)
    param_s      = ca.horzcat(squaredist_s, Y_s)

    NLL_func = ca.Function('NLL', [hyp_s, X_s, Y_s, squaredist_s],
                           [calc_NLL(hyp_s, X_s, Y_s, squaredist_s, 
                                     meanFunc=meanFunc)])
    nlp = {'x': hyp_s, 'f': NLL_func(hyp_s, X, Y_s, squaredist_s), 'p': param_s}
    
    # NLP solver options
    opts = {}
    opts['expand']              = True
    opts['print_time']          = False
    opts['verbose']             = False
    opts['ipopt.print_level']   = 0
    opts['ipopt.linear_solver'] = 'ma27'
    opts['ipopt.max_cpu_time'] = 1
    #opts["ipopt.max_iter"]     = 100
    #opts["ipopt.tol"]          = 1e-12
    #opts["ipopt.hessian_approximation"] = "limited-memory"
    Solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)

    hyp_opt = np.zeros((Ny, num_hyp))
    print('\n----------------------------------------')
    for output in range(Ny):
        print('Optimizing hyperparameters for state %d:' % output)
        stdX      = np.std(X)
        stdF      = np.std(Y[:, output])
        meanF     = np.mean(Y)
        lb        = np.zeros(num_hyp)
        ub        = np.zeros(num_hyp)
        lb[:Nx]    = stdX / 20
        ub[:Nx]    = stdX * 20
        lb[Nx]     = stdF / 20
        ub[Nx]     = stdF * 20
        lb[Nx + 1] = 10**-5 / 10
        ub[Nx + 1] = 10**-5 * 10

        if meanFunc == 'const':
            lb[-1] = meanF / 5
            ub[-1] = meanF * 5
        elif meanFunc != 'zero':
            lb[-1] = meanF / 5
            ub[-1] = meanF * 5
            lb[-h_m:-1] = 0
            ub[-h_m:-1] = 2

        if hyper_init is None:
            hyp_init = pyDOE.lhs(num_hyp, samples=1).flatten()
            hyp_init = hyp_init * (ub - lb) + lb
        else:
            hyp_init = hyper_init[output, :]

        squaredist = np.zeros((N, N * Nx))
        for i in range(Nx):
            d = distance.pdist(X[:, i].reshape(N, 1), 'sqeuclidean')
            squaredist[:, (i * N):(i + 1) * N] = distance.squareform(d)
        param = ca.horzcat(squaredist, Y[:, output])

        obj = np.zeros((multistart, 1))
        hyp_opt_loc = np.zeros((multistart, num_hyp))
    
        for i in range(multistart):
            solve_time = -time.time()
            res = res = Solver(x0=hyp_init, lbx=lb, ubx=ub, p=param)
            status = Solver.stats()['return_status']
            obj[i] = res['f']
            hyp_opt_loc[i, :] = res['x']
            hyp_init = res['x']
            solve_time += time.time()
            print("\t%d: %s - %f s" % (i, status, solve_time))

        hyp_opt[output, :] = hyp_opt_loc[np.argmin(obj)]
    print('----------------------------------------')

    return hyp_opt
