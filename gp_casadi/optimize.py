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

import time
import pyDOE
import numpy as np
import casadi as ca
from scipy.spatial import distance
from . gp_functions import get_mean_function, gp

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


def train_gp(X, Y, meanFunc='zero', hyper_init=None, lam_x0=None, log=False, 
             multistart=1, solver_opts=None):
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
    build_solver_time = -time.time()
    if log:
        X = np.log(X)
        Y = np.log(Y)

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
    opts['ipopt.print_level']   = 1
    opts["ipopt.tol"]          = 1e-8
    if solver_opts is not None:
            opts.update(solver_opts)

    warm_start = False
    if hyper_init is not None:
        opts['ipopt.warm_start_init_point'] = 'yes'
        warm_start = True
    Solver = ca.nlpsol('Solver', 'ipopt', nlp, opts)
    
    hyp_opt = np.zeros((Ny, num_hyp))
    lam_x_opt = np.zeros((Ny, num_hyp))
    invK = np.zeros((Ny, N, N))
    
    build_solver_time += time.time()
    print('\n________________________________________')
    print('# Time to build optimizer: %f sec' % build_solver_time)
    print('----------------------------------------')
    for output in range(Ny):
        print('Optimizing hyperparameters for state %d:' % output)
        stdX      = np.std(X)
        stdF      = np.std(Y[:, output])
        meanF     = np.mean(Y)
        lb        = np.zeros(num_hyp)
        ub        = np.zeros(num_hyp)
        #ub[:]     = np.inf
        lb[:Nx]    = stdX / 20
        ub[:Nx]    = stdX * 20
        lb[Nx]     = stdF / 20
        ub[Nx]     = stdF * 20
        lb[Nx + 1] = 10**-5 / 10
        ub[Nx + 1] = 10**-5 * 10

        if meanFunc is 'const':
            lb[-1] = meanF / 5
            ub[-1] = meanF * 5
        elif meanFunc is not 'zero':
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
        lam_x_opt_loc = np.zeros((multistart, num_hyp))

        for i in range(multistart):
            solve_time = -time.time()
            if warm_start:
                res = res = Solver(x0=hyp_init, lam_x0=lam_x0[output],
                                   lbx=lb, ubx=ub, p=param)
            else:
                res = res = Solver(x0=hyp_init, lbx=lb, ubx=ub, p=param)
            status = Solver.stats()['return_status']
            obj[i]              = res['f']
            hyp_opt_loc[i, :]   = res['x']
            lam_x_opt_loc       = res['lam_x']
            solve_time += time.time()
            print("\t%s - %f s" % (status, solve_time))

        # With multistart, get solution with lowest decision function value
        hyp_opt[output, :]   = hyp_opt_loc[np.argmin(obj)]
        lam_x_opt[output, :] = lam_x_opt_loc[np.argmin(obj)]
        ell = hyp_opt[output, :Nx]
        sf2 = hyp_opt[output, Nx]**2
        sn2 = hyp_opt[output, Nx + 1]**2

        # Calculate the inverse covariance matrix
        K = np.zeros((N, N))
        for i in range(Nx):
            K = squaredist[:, (i  * N):(i + 1) * N] / ell[i]**2 + K
        K = sf2 * np.exp(-.5 * K)
        K = K + sn2 * np.eye(N)     # Add noise variance to diagonal
        K = (K + K.T) * 0.5         # Make sure matrix is symmentric
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            print("K matrix is not positive definit, adding jitter!")
            K = K + np.eye(N) * 1e-8
            L = np.linalg.cholesky(K)
        invL = np.linalg.solve(L, np.eye(N))
        invK[output, :, :] = np.linalg.solve(L.T, invL)
    print('----------------------------------------')

    opt = {}
    opt['hyper'] = hyp_opt
    opt['lam_x'] = lam_x_opt
    opt['invK'] = invK
    return opt


def validate(X_test, Y_test, X, Y, invK, hyper, meanFunc):
    """ Validate GP model with new test data
    """
    N, Ny = Y_test.shape
    Nx = np.size(X, 1)
    z_s = ca.MX.sym('z', Nx)
    gp_func = ca.Function('gp', [z_s],
                                gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper),
                                   z_s.T, meanFunc=meanFunc))
    loss = 0
    for i in range(N):
        y, var = gp_func(X_test[i, :])
        loss += (Y_test[i, :] - y)**2
    
    loss = loss / N
    standardized_loss = loss/ np.std(Y_test, 0)
    
    
    print('\n________________________________________')
    print('Validation of GP model ')
    print('----------------------------------------')
    print('Num training samples: ' + str(np.size(Y, 0)))
    print('Num test samples: ' + str(N))
    print('Mean squared error: ')
    for i in range(Ny):
        print('\t* State %d: %f' % (i + 1, loss[i]))
    print('Standardized mean squared error:')
    for i in range(Ny):
        print('\t* State %d: %f' % (i + 1, standardized_loss[i]))
    print('----------------------------------------')
    

# -----------------------------------------------------------------------------
# Preprocesing of training data
# -----------------------------------------------------------------------------

def standardize(X_original, Y_original, lb, ub):
    # Scale input and output variables
    X_scaled = np.zeros(X_original.shape)
    Y_scaled = np.zeros(Y_original.shape)

    # Normalize input data to [0 1]
    for i in range(np.size(X_original, 1)):
        X_scaled[:, i] = (X_original[:, i] - lb[i]) / (ub[i] - lb[i])

    # Scale output data to a Gaussian with zero mean and unit variance
    for i in range(np.size(Y_original, 1)):
        Y_scaled[:, i] = (Y_original[:, i] - np.mean(Y_original[:, i])) / np.std(Y_original[:, i])

    return X_scaled, Y_scaled


def scale_min_max(X_original, lb, ub):
    # Scale input and output variables
    #X_scaled = np.zeros(X_original.shape)

    # Normalize input data to [0 1]
    return (X_original - lb) / (ub - lb)


def scale_min_max_inverse(X_scaled, lb, ub):
    # Scale input and output variables
    # Normalize input data to [0 1]
    return X_scaled * (ub - lb) + lb


def scale_gaussian(X_original, meanX, stdX):
    # Scale input and output variables
    return (X_original - meanX) / stdX


def scale_gaussian_inverse(X_scaled, meanX, stdX):
    # Scale input and output variables
    return X_scaled * stdX + meanX