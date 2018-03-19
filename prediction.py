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


def predict_casadi(X, Y, invK, hyper, x0, u):

    #X = np.log(X)
    #Y = np.log(Y)
    #x0 = np.log(x0)
    #u = np.log(u)

    # Predict future
    npoints = X.shape[0]
    num_states = len(invK)
    num_inputs = X.shape[1]

    initVar = 0.005 * np.std(Y)
    simTime = 300
    deltat = 30
    simPoints = int(simTime / deltat)

    #z_n = np.concatenate([x0, u]).reshape(1, number_of_inputs)
    #z_n.shape = (1, number_of_inputs)
    mu_EM = np.zeros((simPoints, num_states))
    var_EM = np.zeros((simPoints, num_states))
    #covariance = np.zeros((number_of_inputs, number_of_inputs))
    covar_EM = np.eye(num_inputs) * initVar

    #z_n2 = np.concatenate([x0, u])
    #z_n2.shape = (1, number_of_inputs)
    mu_ME = np.zeros((simPoints, num_states))
    var_ME = np.zeros((simPoints, num_states))

    #z_n3 = np.concatenate([x0, u])
    #z_n3.shape = (1, number_of_inputs)
    mu_TA = np.zeros((simPoints, num_states))
    var_TA = np.zeros((simPoints, num_states))

    F = ca.MX.sym('F', npoints, num_states)
    Xt = ca.MX.sym('X', npoints, num_inputs)
    hyp = ca.MX.sym('hyp', hyper.shape)
    z = ca.MX.sym('z', 1, num_inputs)
    cov = ca.MX.sym('cov', covar_EM.shape)
    var = ca.MX.sym('var', num_states, 1)

    gp_EM = ca.Function('gp', [Xt, F, hyp, z, cov],
                        gp_exact_moment(invK, Xt, F, hyp, num_inputs, z, cov))
    gp_TA = ca.Function('gp_taylor_approx', [Xt, F, hyp, z, var],
                        gp_taylor_approx(invK, Xt, F, hyp, z, var))
    gp_simple = ca.Function('gp_simple', [Xt, F, hyp, z], gp(invK, hyp, Xt, F, z))

    z = np.concatenate([x0, u]).reshape(1, num_inputs)
    for dt in range(simPoints):
        mu, covar = gp_EM.call([X, Y, hyper, z, covar_EM])
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (num_states), (num_states, num_states)
        mu_EM[dt, :], var_EM[dt, :] = mu, np.diag(covar)
        z = ca.vertcat(mu, u).T
        covar_EM[:num_states, :num_states] = covar

    z = np.concatenate([x0, u]).reshape(1, num_inputs)
    var = np.array([1, 1, 1, 1]) * initVar
    for dt in range(simPoints):
        mu, covar = gp_TA.call([X, Y, hyper, z, var])
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (num_states), (num_states, num_states)
        mu_TA[dt, :], var_TA[dt, :] = mu, np.diag(covar)
        z = ca.vertcat(mu, u).T
        #covariance[:number_of_states, :number_of_states] = cov
        var = np.diag(covar)

    z = np.concatenate([x0, u]).reshape(1, num_inputs)
    for dt in range(simPoints):
        mu, var = gp_simple.call([X, Y, hyper, z])
        mu, var = mu.full(), var.full()
        mu.shape, var.shape = (num_states), (num_states)
        mu_ME[dt, :], var_ME[dt, :] = mu, var
        z = ca.vertcat(mu, u).T

    t = np.linspace(0.0, simTime, simPoints)
    u_matrix = np.zeros((simPoints, 2))
    u = np.array([50., 50.])
    x0 = np.array([10., 20., 30., 40.])
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]
    Y_sim = sim_system(x0, u_matrix, simTime, deltat)

    #mu_EM = scale_min_max_inverse(mu_EM, lby, uby)
    #mu_TA = scale_min_max_inverse(mu_TA, lby, uby)
    #mu_ME = scale_min_max_inverse(mu_ME, lby, uby)
    mu_EM = scale_gaussian_inverse(mu_EM, meanY, stdY)
    mu_TA = scale_gaussian_inverse(mu_TA, meanY, stdY)
    mu_ME = scale_gaussian_inverse(mu_ME, meanY, stdY)

    #var_EM = scale_gaussian_inverse(var_EM, 0, 1)
    plt.figure()
    plt.clf()
    for i in range(num_states):
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
    n, D = X.shape  # Number of sampling points and inputs
    E = Y.shape[1]  # Number of outputs

    x1 = X[:, 0].reshape(n, 1)
    x2 = X[:, 1].reshape(n, 1)
    x3 = X[:, 2].reshape(n, 1)
    x4 = X[:, 3].reshape(n, 1)
    u1 = X[:, 4].reshape(n, 1)
    u2 = X[:, 5].reshape(n, 1)
    X1 = np.hstack((x1, x3, u1, u2))
    X2 = np.hstack((x2, x4, u1, u2))
    invK = np.zeros((E, n, n))
    h = 2
    #K1, K2 = train_gp_casadi(X, Y[:, 0], 0)
   # X, Y = standardize(X, Y, [0, 0, 0, 0,0,0], [80, 80, 80, 80,100,100])

    lbx = np.array([.0, .0, .0,  .0, .0,  .0])
    ubx = np.array([40., 40., 40., 40., 100., 100.])
    lby = np.array([0., 0., 0., 0.])
    uby = np.array([40., 40., 40., 40.])
    meanX = np.mean(X, 0)
    stdX = np.std(X, 0)
    X = scale_gaussian(X, meanX, stdX)
    #X = scale_min_max(X, lbx, ubx)
    meanY = np.mean(Y, 0)
    stdY = np.std(Y, 0)
    Y = scale_gaussian(Y, meanY, stdY)
    #Y = scale_min_max(Y, lby, uby)

    if optimize:
        hyper = train_gp(X, Y, meanFunc='zero')
        for i in range(E):
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
    z = np.concatenate([x0, u])
    #z = scaler.transform(z.reshape(1, -1))
    z = scale_gaussian(z, meanX, stdX)
    #z = scale_min_max(z, lbx, ubx)
    mu, var = predict_casadi(X, Y, invK, hyper, z[:4], z[4:])
    #mu, var  = predict_casadi(X, Y, invK, hyper, x0, u)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
