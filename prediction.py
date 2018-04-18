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

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from gp_casadi.gp_functions import gp, gp_exact_moment, gp_taylor_approx
from gp_casadi.optimize import train_gp
from gp_casadi.mpc import mpc, mpc_single
from simulation.four_tank import sim_system
dir_data = 'data/'
dir_parameters = 'parameters/'


def predict_casadi(X, Y, invK, hyper, x0, u):

    # Predict future
    N = X.shape[0]
    Ny = len(invK)
    Nx = X.shape[1]
    Nu = Nx - Ny

    initVar = 0.005 * np.std(Y)
    dt = 30
    simTime = 150.0
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

    #z = np.concatenate([x0, u[0, :]]).reshape(1, Nx)
    mu = x0
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, covar = gp_EM(X, Y, hyper, z, covar_EM)
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (Ny), (Ny, Ny)
        mu_EM[t, :], var_EM[t, :] = mu, np.diag(covar)
        covar_EM[:Ny, :Ny] = covar

    #z = np.concatenate([x0, u]).reshape(1, Nx)
    mu = x0
    var = np.array([1, 1, 1, 1]) * initVar
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, covar = gp_TA(X, Y, hyper, z, var)
        mu, covar = mu.full(), covar.full()
        mu.shape, covar.shape = (Ny), (Ny, Ny)
        mu_TA[t, :], var_TA[t, :] = mu, np.diag(covar)
        var = np.diag(covar)

    #z = np.concatenate([x0, u]).reshape(1, Nx)
    mu = x0
    for t in range(Nt):
        z = ca.vertcat(mu, u[t, :]).T
        mu, var = gp_simple(X, Y, hyper, z)
        mu, var = mu.full(), var.full()
        mu.shape, var.shape = (Ny), (Ny)
        mu_ME[t, :], var_ME[t, :] = mu, var

    t = np.linspace(0.0, simTime, Nt)
    Y_sim = sim_system(x0, u, simTime, dt)


    plt.figure()
    fontP = FontProperties()
    fontP.set_size('small')
    for i in range(Ny):
        plt.subplot(2, 2, i + 1)
        mu_EM_i = mu_EM[:, i]
        mu_TA_i = mu_TA[:, i]
        mu_ME_i = mu_ME[:, i]

        sd_EM_i = np.sqrt(var_EM[:, i])
        sd_TA_i = np.sqrt(var_TA[:, i])
        sd_ME_i = np.sqrt(var_ME[:, i])

        plt.gca().fill_between(t.flat, mu_EM_i - 2 * sd_EM_i, mu_EM_i +
               2 * sd_EM_i, color="#555555", label='95% conf interval EM')
        plt.gca().fill_between(t.flat, mu_TA_i - 2 * sd_TA_i, mu_TA_i +
               2 * sd_TA_i, color="#FFFaaa", label='95% conf interval TA')
        plt.gca().fill_between(t.flat, mu_ME_i - 2 * sd_ME_i, mu_ME_i +
               2 * sd_ME_i, color="#bbbbbb", label='95% conf interval ME')

        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_EM_i, label='95% conf interval EM')
        #plt.errorbar(t, mu_TA_i, yerr=2 * sd_TA_i, label='95% conf interval TA')
        #plt.errorbar(t, mu_EM_i, yerr=2 * sd_ME_i, label='95% conf interval ME')

        plt.plot(t, Y_sim[:, i], 'b-', label='Simulation')
        plt.plot(t, mu_EM_i, 'rx', label='GP Excact moment')
        plt.plot(t, mu_TA_i, 'kx', label='GP Taylor Approx')
        plt.plot(t, mu_ME_i, 'yx', label='GP Mean Equivalence')

        plt.ylabel('Level in tank ' + str(i + 1) + ' [cm]')
        plt.legend(prop=fontP)
        plt.suptitle('Simulation and prediction', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()

    plt.figure()
    u_temp = np.vstack((u, u[-1, :]))
    for i in range(Nu):
        plt.subplot(2, 1, i + 1)
        plt.step(t, u_temp[:, i], 'k', where='post')
        plt.ylabel('Flow  ' + str(i + 1) + ' [ml/s]')
        plt.suptitle('Control inputs', fontsize=16)
        plt.xlabel('Time [s]')
        #plt.ylim([0, 40])
    plt.show()
    return mu_EM, var_EM


if __name__ == "__main__":
    X = np.loadtxt(dir_data + 'X_matrix_tank')
    Y = np.loadtxt(dir_data + 'Y_matrix_tank')

    optimize = True
    meanFunc = 'zero'
    log = False
    N, Nx = X.shape  # Number of sampling points and inputs
    Ny = Y.shape[1]  # Number of outputs
    if optimize:
        opt = train_gp(X, Y, meanFunc=meanFunc, log=log)
        hyper = opt['hyper']
        invK  = opt['invK']
        lam_x = opt['lam_x']
        for i in range(Ny):
            np.savetxt(dir_parameters + 'invK' + str(i + 1), invK[i, :, :], delimiter=',')
        np.savetxt(dir_parameters + 'hyper_opt', hyper, delimiter=',')
    else:
        hyper = np.loadtxt(dir_parameters + 'hyper_opt', delimiter=',')
        invK = np.zeros((Ny, N, N))
        for i in range(Ny):
            invK[i, :, :] = np.loadtxt(dir_parameters + 'invK' + str(i + 1), delimiter=',')

    eps = 7
    dt = 10
    x0 = np.array([8., 10., 8., 18.])
    x_sp = np.array([14., 14., 14.2, 21.3])
    ulb = [eps, eps]
    uub = [60., 60.]
    xlb = [eps, eps, eps, eps]
    xub = [28, 28, 28, 28]

    x0_ = x0
    for i in range(0):
        x, u = mpc(X, Y, x0, x_sp, invK, hyper, horizon=120.0,
                          sim_time=60.0, dt=dt, simulator=sim_system, method='TA',
                          ulb=ulb, uub=uub, xlb=xlb, xub=xub,
                          meanFunc=meanFunc, log=log, costFunc='quad', plot=False)
        X = np.vstack((X, np.hstack((x[1:2], u[1:]))))
        Y = np.vstack((Y, x[2:]))
        x0_ = x

        # Train again
        opt = train_gp(X, Y, meanFunc=meanFunc, hyper_init=hyper, lam_x0=lam_x, log=log)
        hyper = opt['hyper']
        invK  = opt['invK']
        lam_x = opt['lam_x']

    x, u= mpc(X, Y, x0, x_sp, invK, hyper, horizon=6*dt,
          sim_time=200.0, dt=dt, simulator=sim_system, method='ME',
          ulb=ulb, uub=uub, xlb=xlb, xub=xub, plot=True,
          meanFunc=meanFunc, terminal_constraint=None, log=log,
          costFunc='quad')
    
    #mean, u_mpc = mpc(X, Y, invK, hyper, method='ME')
    #mu, var  = predict_casadi(X, Y, invK, hyper, x0, u_mpc)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
