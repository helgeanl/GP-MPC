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
from gp_casadi.optimize import train_gp, validate
from gp_casadi.mpc import mpc, plot_repeats
from gp_casadi.mpc_single_shooting import mpc_single
from gp_casadi.mpc_class import MPC
from simulation.four_tank import sim_system
dir_data = 'data/'
dir_parameters = 'parameters/'





if __name__ == "__main__":
    X = np.loadtxt(dir_data + 'X_matrix_tank')
    Y = np.loadtxt(dir_data + 'Y_matrix_tank')
    
    X_test = np.loadtxt(dir_data + 'X_matrix_test_tank')
    Y_test = np.loadtxt(dir_data + 'Y_matrix_test_tank')

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
        validate(X_test, Y_test, X, Y, invK, hyper, meanFunc)
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
    
    solver_opts = {}
    solver_opts['ipopt.linear_solver'] = 'ma27'
    #solver_opts["ipopt.hessian_approximation"] = "limited-memory"
    solver_opts['ipopt.max_cpu_time'] = 10
    #opts['ipopt.max_iter'] = 10
    solver_opts['expand'] = True
    x0_ = x0
    for i in range(0):
        x, u = mpc(X, Y, x0, x_sp, invK, hyper, horizon=1*dt,
                          sim_time=2*dt, dt=dt, simulator=sim_system, method='ME',
                          ulb=ulb, uub=uub, xlb=xlb, xub=xub,
                          meanFunc=meanFunc, log=log, costFunc='quad', plot=True)
        if False:
            X = np.vstack((X, np.hstack((x[:-1], u))))
            Y = np.vstack((Y, x[1:]))
            x0_ = x
    
            # Train again
            opt = train_gp(X, Y, meanFunc=meanFunc, hyper_init=hyper, lam_x0=lam_x, log=log)
            hyper = opt['hyper']
            invK  = opt['invK']
            lam_x = opt['lam_x']

#    x, u= mpc(X, Y, x0, x_sp, invK, hyper, horizon=4*dt,
#          sim_time=12*dt, dt=dt, simulator=sim_system, method='ME',
#          ulb=ulb, uub=uub, xlb=xlb, xub=xub, plot=True,
#          meanFunc=meanFunc, terminal_constraint=None, log=log,
#          costFunc='quad', feedback=True)
    
    if 1:
        Q = np.array([[5, 0, 0, 0],
                      [0, 5, 0, 0],
                      [0, 0, 5, 0],
                      [0, 0, 0, 5]])
        P = np.array([[10, 0, 0, 0],
                      [0, 10, 0, 0],
                      [0, 0, 10, 0],
                      [0, 0, 0, 10]])
        x_sp = np.array([14., 14., 23, 23])
        Nt = 10
        mpc = MPC(X, Y, x0, x_sp, invK, hyper, horizon=5*dt,
              sim_time=Nt*dt, dt=dt, method='ME',
              ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, P=P,
              meanFunc=meanFunc, terminal_constraint=None, log=log,
              costFunc='quad', feedback=True, solver_opts=solver_opts)
    #    mpc.plot()
        
        N_repeat = 1
        x = np.zeros((N_repeat, Nt + 1, Ny))
        for i in range(N_repeat):
            x[i, :, :], u = mpc.solve(simulator=sim_system, x0=x0)
        mpc.plot()
            
#        x_sp = np.ones((Nt + 1, Ny)) * x_sp
#        fig_x = plot_repeats(x=x, dt=dt, x_sp=x_sp,
#                   title='MPC with %d step/ %d s horizon - GP: %s' % (Nt, Nt*dt, 'ME')
#               )
        
#    x_sp = np.array([4., 18., 14.2, 30.3])
#    x, u = mpc.solve(simulator=sim_system, x0=x[-1,:], x_sp=x_sp, u0=u[-1,:])

    #mean, u_mpc = mpc(X, Y, invK, hyper, method='ME')
    #mu, var  = predict_casadi(X, Y, invK, hyper, x0, u_mpc)
    #mu2, var2  = predict(X, Y, invK, hyper, x0, u)
