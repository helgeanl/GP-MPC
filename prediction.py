# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
path.append(r"./GP_MPC/")

#import time

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from simulation.four_tank import sim_system
from gp_casadi.gp_functions import gp, gp_exact_moment, gp_taylor_approx
from gp_casadi.optimize import train_gp
from gp_numpy.gp_functions import calc_cov_matrix

dir_data = 'data/'
dir_parameters = 'parameters/'


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


def cost_lf(mean, covar, P):
    # Cost function
    P_s = ca.SX.sym('Q', ca.SX.size(P))
    x_s = ca.SX.sym('x', ca.SX.size(mean))
    covar_x_s = ca.SX.sym('K', ca.MX.size(covar))
    
    sqnorm_x = ca.Function('sqnorm_x', [x_s, P_s],
                           [ca.mtimes(x_s.T, ca.mtimes(P_s, x_s))])
    trace_x = ca.Function('trace_x', [P_s, covar_x_s],
                           [ca.trace(ca.mtimes(P_s, covar_x_s))])
    return sqnorm_x(mean, P) + trace_x(P, covar) 


def cost_l(mean, covar, v, Q, R, K):
    Q_s = ca.SX.sym('Q', ca.MX.size(Q))
    R_s = ca.SX.sym('R', ca.MX.size(R))
    K_s = ca.SX.sym('K', ca.MX.size(K))
    x_s = ca.SX.sym('x', ca.MX.size(mean))
    v_s = ca.SX.sym('v', ca.MX.size(v))
    covar_x_s = ca.SX.sym('K', ca.MX.size(covar))
    covar_u_s = ca.SX.sym('covar_u', ca.MX.size(R))

    sqnorm_x = ca.Function('sqnorm_x', [x_s, Q_s],
                           [ca.mtimes(x_s.T, ca.mtimes(Q_s, x_s))])
    sqnorm_u = ca.Function('sqnorm_u', [v_s, R_s],
                           [ca.mtimes(v_s.T, ca.mtimes(R_s, v_s))])
    covar_u  = ca.Function('covar_u', [covar_x_s, K_s],
                           [ca.mtimes(K_s, ca.mtimes(covar_x_s, K_s.T))])
    trace_u  = ca.Function('trace_u', [R_s, covar_u_s],
                           [ca.trace(ca.mtimes(R_s, covar_u_s))])
    trace_x  = ca.Function('trace_x', [Q_s, covar_x_s],
                           [ca.trace(ca.mtimes(Q_s, covar_x_s))])  
    uFunc    = ca.Function('u', [x_s, v_s, K_s ], [ca.mtimes(K_s, x_s) + v_s])
    u = uFunc(mean, v, K)
    return sqnorm_x(mean, Q) + sqnorm_u(u, R) + trace_x(Q, covar) \
                 + trace_u(R, covar_u(covar, K))


def mpc(X, Y, invK, hyper):
    initVar = 0.005 * np.std(Y)
    horizon = 300
    dt = 30
    Nt = int(horizon / dt)
    N = X.shape[0]
    Ny = len(invK)
    Nx = X.shape[1]
    Nu = Nx - Ny
    
    P = np.eye(Ny)
    Q = np.eye(Ny)
    R = np.eye(Nu)
    K = np.ones(Nu, Ny)

    P_s = ca.MX.sym('P', Ny, Ny)
    Q_s = ca.MX.sym('Q', Ny, Ny)
    R_s = ca.MX.sym('R', Nu, Nu)
    K_s = ca.MX.sym('K', Nu, Ny)
    mean_s = ca.MX.sym('mean', Ny)
    v_s = ca.MX.sym('v', Nu)
    covar_s = ca.MX.sym('covar', Ny, Ny)
    
#    Y_s = ca.MX.sym('Y', N, Ny)
#    X_s = ca.MX.sym('X', N, Nx)
#    hyp_s = ca.MX.sym('hyp', hyper.shape)
#    z_s = ca.MX.sym('z', 1, Nx)

    
    gp = ca.Function('gp', [mean_s, covar_s],
                        gp_exact_moment(invK, X, Y, hyper, mean_s.T, covar_s))    

    # Define stage cost and terminal cost
    l_func = ca.Function('l', [mean_s, covar_s, v_s], 
                           [cost_l(mean_s, covar_s, v_s, Q, R, K)])
    lf_func = ca.Function('lf', [mean_s, covar_s], 
                           [cost_lf(mean_s, covar_s, P)])
    
    u_func = ca.Function('u', [mean_s, v_s], [ca.mtimes(K, mean_s) + v_s])
    covar_u  = ca.Function('covar_u', [covar_s],
                       [ca.mtimes(K, ca.mtimes(covar_s, K.T))])
    
    # Bounds on u
    uub = [0, 0]
    ulb = [60, 60]
    # Initial state
    mean_0 = np.array([8., 10., 8., 18.])
    covar_0 = np.eye(Ny) * initVar
    covar_temp = np.zeros(Nx, Nx)
    
    # Create variables struct
    var = ca.tools.struct_symSX([(
            ca.tools.entry('mean', shape=(Ny,), repeat=Nt + 1),
            ca.tools.entry('covar', shape=(Ny, Ny), repeat=Nt + 1),
            ca.tools.entry('u', shape=(Nu,), repeat=Nt)
    )])
    
    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    
    # Adjust the relevant constraints
    for t in range(Nt):
        varlb['u', t, :] = ulb
        varub['u', t, :] = uub

    # Now build up constraints and objective
    obj = ca.SX(0)
    con = []
    for t in range(Nt):
        z = ca.vertcat(var['mean', t], u_func(var['mean', t], var['v', t]))
        covar_temp[:Ny, :Ny] = var['covar', t]
        covar_temp[Ny:, Ny:] = covar_u(var['covar', t])
        mean_i, covar_i = gp(z, covar_temp)
        con.append(mean_i, covar_i)
        obj += l_func(var['mean', t], var['covar', t], var['v', t])
    obj += lf_func(var['mean', Nt], var['covar', Nt])

    # Build solver object
    con = ca.vertcat(*con)
    conlb = np.zeros((Ny * Nt,))
    conub = np.zeros((Ny * Nt,))
    
    nlp = dict(x=var, f=obj, g=con)
    opts = {}
    opts['ipopt.print_level'] = 0
    opts['ipopt.max_cpu_time'] = 60
    opts['print_time'] = False
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)


    # Simulate
    Nsim = 20
    times = dt * Nsim * np.linspace(0, 1, Nsim + 1)
    mean = np.zeros((Nsim + 1, Ny))
    mean[0, :] = mean_0
    covar = np.zeros((Nsim + 1, Ny, Ny))
    covar[0, :, :] = covar_0
    u = np.zeros((Nsim, Nu))
    
    for t in range(Nsim):
        # Fix initial state
        varlb['mean', 0, :] = mean[t, :]
        varub['mean', 0, :] = mean[t, :]
        varlb['covar', 0, :, :] = covar[t, :, :]
        varub['covar', 0, :, :] = covar[t, :, :]
        varguess['mean', 0, :] = mean[t, :]
        varguess['covar', 0, :, :] = covar[t, :, :]
        args = dict(x0=varguess,
                    lbx=varlb,
                    ubx=varub,
                    lbg=conlb,
                    ubg=conub)
        
        # Solve nlp
        sol = solver(**args)
        status = solver.stats()['return_status']
        optvar = var(sol['x'])
        
        # Print status
        print("%d: %s" % (t, status))
        u[t, :] = optvar['u', 0, :]
        covar[t + 1, :, :] = optvar['covar', 0, :, :]
        # Simulate
        vdpargs = dict(x0=ca.vertcat(mean[t, :], covar[t, :, :]), p=u[t, :])
        out = vdp(**vdpargs)
        mean[t + 1, :] = np.array(out['xf']).flatten()
    #####
    u_matrix = np.zeros((Nt, 2))
    x0 = np.array([8., 10., 8., 18.])
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]
    Y_sim = sim_system(x0, u_matrix, simTime, deltat)
    

def predict_casadi(X, Y, invK, hyper, x0, u):

    #X = np.log(X)
    #Y = np.log(Y)
    #x0 = np.log(x0)
    #u = np.log(u)

    # Predict future
    N = X.shape[0]
    Ny = len(invK)
    Nx = X.shape[1]

    initVar = 0.005 * np.std(Y)
    simTime = 300
    dt = 30
    Nt = int(simTime / dt)

    mu_EM = np.zeros((Nt, Ny))
    var_EM = np.zeros((Nt, Ny))
    covar_EM = np.eye(Nx) * initVar

    mu_ME = np.zeros((Nt, Ny))
    var_ME = np.zeros((Nt, Ny))

    mu_TA = np.zeros((Nt, Ny))
    var_TA = np.zeros((Nt, Ny))

    Y_s = ca.MX.sym('Y', N, Ny)
    X_s = ca.MX.sym('X', N, Nx)
    hyp_s = ca.MX.sym('hyp', hyper.shape)
    z_s = ca.MX.sym('z', 1, Nx)
    cov_s = ca.MX.sym('cov', covar_EM.shape)
    var_s = ca.MX.sym('var', Ny, 1)

    gp_EM = ca.Function('gp', [X_s, Y_s, hyp_s, z_s, cov_s],
                        gp_exact_moment(invK, X_s, Y_s, hyp_s, z_s, cov_s))
    gp_TA = ca.Function('gp_taylor_approx', [X_s, Y_s, hyp_s, z_s, var_s],
                        gp_taylor_approx(invK, X_s, Y_s, hyp_s, z_s, var_s))
    gp_simple = ca.Function('gp_simple', [X_s, Y_s, hyp_s, z_s], gp(invK, X_s, Y_s, hyp_s, z_s, meanFunc='zero'))

    z = np.concatenate([x0, u]).reshape(1, Nx)
    for dt in range(Nt):
        mu, covar = gp_EM(X, Y, hyper, z, covar_EM)
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (Ny), (Ny, Ny)
        mu_EM[dt, :], var_EM[dt, :] = mu, np.diag(covar)
        z = ca.vertcat(mu, u).T
        covar_EM[:Ny, :Ny] = covar

    z = np.concatenate([x0, u]).reshape(1, Nx)
    var = np.array([1, 1, 1, 1]) * initVar
    for dt in range(Nt):
        mu, covar = gp_TA(X, Y, hyper, z, var)
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (Ny), (Ny, Ny)
        mu_TA[dt, :], var_TA[dt, :] = mu, np.diag(covar)
        z = ca.vertcat(mu, u).T
        #covariance[:number_of_states, :number_of_states] = cov
        var = np.diag(covar)

    z = np.concatenate([x0, u]).reshape(1, Nx)
    for dt in range(Nt):
        mu, var = gp_simple(X, Y, hyper, z)
        mu, var = mu.full(), var.full()
        mu.shape, var.shape = (Ny), (Ny)
        mu_ME[dt, :], var_ME[dt, :] = mu, var
        z = ca.vertcat(mu, u).T

    t = np.linspace(0.0, simTime, Nt)
    u_matrix = np.zeros((Nt, 2))
    u = np.array([30., 30.])
    x0 = np.array([8., 10., 8., 18.])
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]
    Y_sim = sim_system(x0, u_matrix, simTime, dt)

    #mu_EM = scale_min_max_inverse(mu_EM, lby, uby)
    #mu_TA = scale_min_max_inverse(mu_TA, lby, uby)
    #mu_ME = scale_min_max_inverse(mu_ME, lby, uby)
    #mu_EM = scale_gaussian_inverse(mu_EM, meanY, stdY)
    #mu_TA = scale_gaussian_inverse(mu_TA,meanY, stdY)
    #mu_ME = scale_gaussian_inverse(mu_ME, meanY, stdY)

    #var_EM = scale_gaussian_inverse(var_EM, 0, 1)
    plt.figure()
    plt.clf()
    for i in range(Ny):
        plt.subplot(2, 2, i + 1)
        mu_EM_i = mu_EM[:, i]
        mu_TA_i = mu_TA[:, i]
        mu_ME_i = mu_ME[:, i]

        sd_EM_i = np.sqrt(var_EM[:, i])
        sd_TA_i = np.sqrt(var_TA[:, i])
        sd_ME_i = np.sqrt(var_ME[:, i])

        plt.gca().fill_between(t.flat, mu_EM_i - 2 * sd_EM_i, mu_EM_i + 2 * sd_EM_i, color="#555555")
        plt.gca().fill_between(t.flat, mu_TA_i - 2 * sd_TA_i, mu_TA_i + 2 * sd_TA_i, color="#FFFaaa")
        plt.gca().fill_between(t.flat, mu_ME_i - 2 * sd_ME_i, mu_ME_i + 2 * sd_ME_i, color="#bbbbbb")

        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_EM_i)
        #plt.errorbar(t, mu_TA_i, yerr=2 * sd_TA_i)
        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_ME_i)

        plt.plot(t, Y_sim[:, i], 'b-')
        plt.plot(t, mu_EM_i, 'rx')
        plt.plot(t, mu_TA_i, 'kx')
        plt.plot(t, mu_ME_i, 'yx')

        labels = ['Simulation', 'GP Excact moment', 'GP Mean Equivalence', 'GP Taylor Approx',
                  '95% conf interval EM', '95% conf interval TA', '95% conf interval ME']
        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(labels)
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()
    return mu_EM, var_EM


if __name__ == "__main__":
    X = np.loadtxt(dir_data + 'X_matrix_tank')

    Y = np.loadtxt(dir_data + 'Y_matrix_tank')
    optimize = True
    N, Nx = X.shape  # Number of sampling points and inputs
    Ny = Y.shape[1]  # Number of outputs

    x1 = X[:, 0].reshape(N, 1)
    x2 = X[:, 1].reshape(N, 1)
    x3 = X[:, 2].reshape(N, 1)
    x4 = X[:, 3].reshape(N, 1)
    u1 = X[:, 4].reshape(N, 1)
    u2 = X[:, 5].reshape(N, 1)
    X1 = np.hstack((x1, x3, u1, u2))
    X2 = np.hstack((x2, x4, u1, u2))
    invK = np.zeros((Ny, N, N))
    h = 2
    #K1, K2 = train_gp_casadi(X, Y[:, 0], 0)
   # X, Y = standardize(X, Y, [0, 0, 0, 0,0,0], [80, 80, 80, 80,100,100])

    #lbx = np.array([.0, .0, .0,  .0, .0,  .0])
    #ubx = np.array([40., 40., 40., 40., 100., 100.])
    #lby = np.array([0., 0., 0., 0.])
    #uby = np.array([40., 40., 40., 40.])
    #meanX = np.mean(X, 0)
    #stdX = np.std(X, 0)
    meanY = np.mean(Y, 0)
    stdY = np.std(Y, 0)
    #meanY = meanX[:E]
    #stdY = meanX[:E]
    stdU = np.std(X[:, Ny:], 0)
    meanU = np.mean(X[:, Ny:], 0)
    meanX = np.concatenate([meanY, meanU])
    stdX = np.concatenate([stdY, stdU])
    #X = scale_gaussian(X, meanX, stdX)
    #X = scale_min_max(X, lbx, ubx)
   
    #Y = scale_gaussian(Y, meanY, stdY)
    #Y = scale_min_max(Y, lby, uby)

    if optimize:
        hyper = train_gp(X, Y, meanFunc='zero')
        for i in range(Ny):
            K = calc_cov_matrix(X, hyper[i, :Nx], hyper[i, Nx]**2)
            K = K + hyper[i, Nx + 1]**2 * np.eye(N)  # Add noise variance to diagonal
            K = (K + K.T) * 0.5   # Make sure matrix is symmentric
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definit, adding jitter!")
                K = K + np.eye(N) * 1e-8
                L = np.linalg.cholesky(K)
            invL = np.linalg.solve(L, np.eye(N))
            invK[i, :, :] = np.linalg.solve(L.T, invL)    # np.linalg.inv(K)
            np.savetxt(dir_parameters + 'invK' + str(i + 1), invK[i, :, :], delimiter=',')
        np.savetxt(dir_parameters + 'hyper_opt', hyper, delimiter=',')

    else:
        hyper = np.loadtxt(dir_parameters + 'hyper_opt', delimiter=',')
        for i in range(Ny):
            invK[i, :, :] = np.loadtxt(dir_parameters + 'invK' + str(i + 1), delimiter=',')
            #hyper[i, -1] = 0  # np.mean(Y[:, i])

    u = np.array([30., 30.])
    x0 = np.array([8., 10., 8., 18.])
    z = np.concatenate([x0, u])
    #z = scaler.transform(z.reshape(1, -1))
    #z = scale_gaussian(z, meanX, stdX)
    #z = scale_min_max(z, lbx, ubx)
    mu, var = predict_casadi(X, Y, invK, hyper, z[:4], z[4:])
    #mu, var  = predict_casadi(X, Y, invK, hyper, x0, u)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
