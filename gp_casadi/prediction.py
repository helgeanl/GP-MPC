# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014.01.10")
#path.append(r"../Simulation")

#import time

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from ..simulation.four_tank import sim_system
from .gp_functions import gp, gp_exact_moment, gp_taylor_approx
from noisyGP import GP_noisy_input


dir_data = '../Data/'
dir_parameters = '../Parameters/'


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
    number_of_states = len(invK)
    number_of_inputs = X.shape[1]

    simTime = 300
    deltat = 3
    simPoints = simTime / deltat

    z_n = np.concatenate([x0, u])
    z_n.shape = (1, number_of_inputs)
    mu_EM = np.zeros((simPoints, number_of_states))
    var_EM = np.zeros((simPoints, number_of_states))
    #covariance = np.zeros((number_of_inputs, number_of_inputs))
    covariance = np.eye(number_of_inputs) * 0.1

    z_n2 = np.concatenate([x0, u])
    z_n2.shape = (1, number_of_inputs)
    mu_ME = np.zeros((simPoints, number_of_states))
    var_ME = np.zeros((simPoints, number_of_states))

    z_n3 = np.concatenate([x0, u])
    z_n3.shape = (1, number_of_inputs)
    mu_TA = np.zeros((simPoints, number_of_states))
    var_TA = np.zeros((simPoints, number_of_states))

    D = number_of_inputs
    F = ca.MX.sym('F', npoints, number_of_states)
    Xt = ca.MX.sym('X', npoints, number_of_inputs)
    hyp = ca.MX.sym('hyp', hyper.shape)
    z = ca.MX.sym('z', z_n.shape)
    cov = ca.MX.sym('cov', covariance.shape)
    cov2 = ca.MX.sym('cov2', 4, 1)

    gp_EM = ca.Function('gp', [Xt, F, hyp, z, cov], GP_noisy_input(invK, Xt, F, hyp, D, z, cov))
    gp_TA = ca.Function('gp_taylor_approx', [Xt, F, hyp, z, cov2], gp_taylor_approx(invK, Xt, F, hyp, z, cov2))
    gp_simple = ca.Function('gp_simple', [Xt, F, hyp, z], gp_casadi(invK, hyp, Xt, F, z))

    for dt in range(simPoints):
        mu, cov = gp_EM.call([X, Y, hyper, z_n, covariance])
        mu, cov = mu.full(), cov.full()
        mu.shape, cov.shape = (number_of_states), (number_of_states, number_of_states)
        mu_EM[dt, :], var_EM[dt, :] = mu, np.diag(cov)
        z_n = ca.vertcat(mu, u).T
        covariance[:number_of_states, :number_of_states] = cov

    #covar2 = np.zeros((4, 1))
    covar2 = np.array([.1, .1, .1, .1])
    for dt in range(simPoints):
        mu, cov = gp_TA.call([X, Y, hyper, z_n3, covar2])
        mu, cov = mu.full(), cov.full()
        mu.shape, cov.shape = (number_of_states), (number_of_states, number_of_states)
        mu_TA[dt, :], var_TA[dt, :] = mu, np.diag(cov)
        z_n3 = ca.vertcat(mu, u).T
        #covariance[:number_of_states, :number_of_states] = cov
        covar2 = np.diag(cov)

    for dt in range(simPoints):
        mu, var = gp_simple.call([X, Y, hyper, z_n2])
        mu, var = mu.full(), var.full()
        mu.shape, var.shape = (number_of_states), (number_of_states)
        mu_ME[dt, :], var_ME[dt, :] = mu, var
        z_n2 = ca.vertcat(mu, u).T

    t = np.linspace(0.0, simTime, simPoints)
    u_matrix = np.zeros((simPoints, 2))
    u = np.array([50., 50.])
    x0 = np.array([10., 20., 30., 40.])
    u_matrix[:, 0] = u[0]
    u_matrix[:, 1] = u[1]
    Y_sim = sim_system(x0, u_matrix, simTime, deltat)

    #lby = np.array([0., 0., 0., 0.])
    #uby = np.array([80., 80., 80., 80.])

    #mu_EM = scale_min_max_inverse(mu_EM, lby, uby)
    #mu_TA = scale_min_max_inverse(mu_TA, lby, uby)
    #mu_ME = scale_min_max_inverse(mu_ME, lby, uby)
    mu_EM = scale_gaussian_inverse(mu_EM, meanY, stdY)
    mu_TA = scale_gaussian_inverse(mu_TA, meanY, stdY)
    mu_ME = scale_gaussian_inverse(mu_ME, meanY, stdY)


    #var_n = scale_gaussian_inverse(var_n)
    plt.figure()
    plt.clf()
    for i in range(number_of_states):
        plt.subplot(2, 2, i + 1)
        mu_EM_i = mu_EM[:, i]
        mu_TA_i = mu_TA[:, i]
        mu_ME_i = mu_ME[:, i]

        sd_EM_i = np.sqrt(var_EM[:, i])
        plt.gca().fill_between(t.flat, mu_EM_i - 2 * sd_EM_i, mu_EM_i + 2 * sd_EM_i, color="#555555")

        sd_TA_i = np.sqrt(var_TA[:, i])
        plt.gca().fill_between(t.flat, mu_TA_i - 2 * sd_TA_i, mu_TA_i + 2 * sd_TA_i, color="#FFFaaa")

        sd_ME_i = np.sqrt(var_ME[:, i])
        plt.gca().fill_between(t.flat, mu_ME_i - 2 * sd_ME_i, mu_ME_i + 2 * sd_ME_i, color="#bbbbbb", alpha=.9)

        #plt.errorbar(t, mu3, yerr=2 * sd3)
        #plt.errorbar(t, mu1, yerr=2 * sd1)
        #plt.errorbar(t, mu2, yerr=2 * sd2)

        plt.plot(t, Y_sim[:, i], 'b-')
        plt.plot(t, mu_EM_i, 'r--')
        plt.plot(t, mu_TA_i, 'k--')
        plt.plot(t, mu_ME_i, 'y--')

        labels = ['Simulation', 'GP Excact moment', 'GP Mean Equivalence', 'GP Taylor Approx',
                  '95% conf interval EM', '95% conf interval TA', '95% conf interval ME']
        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(labels)
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()
    return mu_EM, var_EM, covariance


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

    lbx = np.array([.0, .0, .0, .0, .0, .0])
    ubx = np.array([80., 80., 80., 80., 100., 100.])
    lby = np.array([0., 0., 0., 0.])
    uby = np.array([80., 80., 80., 80.])
    X = scale_min_max(X, lbx, ubx)
    meanY = np.mean(Y)
    stdY = np.std(Y)
    Y = scale_min_max(Y, lby, uby)
    #Y = scale_gaussian(Y, meanY, stdY)


    if optimize:
        hyper = np.zeros((E, D + h))
        n, D = X.shape
        for i in range(E):
            #hyper[i, :] = train_gp(X, Y[:, i], i)
            hyper[i, :] = train_gp_casadi(X, Y[:, i], i, meanFunc=0)
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
    z = scale_gaussian(z, meanY, stdY)
    mu, var, covariance  = predict_casadi(X, Y, invK, hyper, z[:4], z[4:])
    #mu, var  = predict_casadi(X, Y, invK, hyper, x0, u)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
