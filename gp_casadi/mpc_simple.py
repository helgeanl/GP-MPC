# -*- coding: utf-8 -*-
"""
Gaussian Process
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
import time
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ctools
from simulation.four_tank import sim_system
from . gp_functions import gp



def cost_lf_simple(mean, mean_ref, P):
    # Cost function
    P_s = ca.SX.sym('Q', ca.MX.size(P))
    x_s = ca.SX.sym('x', ca.MX.size(mean))
    sqnorm_x = ca.Function('sqnorm_x', [x_s, P_s],
                           [ca.mtimes(x_s.T, ca.mtimes(P_s, x_s))])
    return sqnorm_x(mean - mean_ref, P)

                 
def cost_l_simple(mean, mean_ref, u, Q, R, K):
    Q_s = ca.SX.sym('Q', ca.MX.size(Q))
    R_s = ca.SX.sym('R', ca.MX.size(R))

    x_s = ca.SX.sym('x', ca.MX.size(mean))
    u_s = ca.SX.sym('u', ca.MX.size(u))

    sqnorm_x = ca.Function('sqnorm_x', [x_s, Q_s],
                           [ca.mtimes(x_s.T, ca.mtimes(Q_s, x_s))])
    sqnorm_u = ca.Function('sqnorm_u', [u_s, R_s],
                           [ca.mtimes(u_s.T, ca.mtimes(R_s, u_s))])

    return sqnorm_x(mean - mean_ref, Q) + sqnorm_u(u, R)

def mpc_simple(X, Y, invK, hyper, mpc_opts):
    horizon = 150
    dt = 30
    Nsim = 20
    
#    mpc_opts = {}
#    mpc_opts['horizon'] = 150
#    mpc_opts['dt'] = 30
#    mpc_opts['Nsim'] = 20
#    mpc_opts['K'] = np.array([[.5, .0, .5, .0], 
#                             [.0, .5, .0, .5]])
#    mpc_opts['P'] = np.eye(Ny) * 6
#    mpc_opts['Q'] = np.eye(Ny) * 1
#    mpc_opts['R'] = np.eye(Nu) * 0.01
    
    Nt = int(horizon / dt)
    Ny = len(invK)
    Nx = X.shape[1]
    Nu = Nx - Ny
    
    P = np.eye(Ny) * 6
    Q = np.eye(Ny) * 1
    R = np.eye(Nu) * 0.01
    #K = np.ones((Nu, Ny)) * .5
    K = np.array([[.5, .0, .5, .0], 
                  [.0, .5, .0, .5]])
    # Bounds on u
    ulb = [0., 0.]
    uub = [60., 60.] 
    xlb = [0, 0, 0, 0]
    xub = [40, 40, 40, 40]

    # Initial state
    mean_0 = np.array([8., 10., 8., 18.])
    mean_ref = ca.MX([14., 14., 14.2, 21.3])
    
    mean_s = ca.MX.sym('mean', Ny)
    v_s = ca.MX.sym('v', Nu)
    z_s = ca.vertcat(mean_s, v_s)
   

    # Simple gaussian process without uncertainty propagation
    gp_simple = ca.Function('gp_simple', [z_s], 
                            gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper), z_s.T, meanFunc='zero'))

    # Define stage cost and terminal cost
    l_func = ca.Function('l', [mean_s, v_s], 
                           [cost_l_simple(mean_s, mean_ref, v_s, ca.MX(Q), ca.MX(R), ca.MX(K))])
    lf_func = ca.Function('lf', [mean_s], 
                           [cost_lf_simple(mean_s, mean_ref,  ca.MX(P))])
    # Feedback function
    u_func = ca.Function('u', [mean_s, v_s], [ca.mtimes(ca.MX(K), mean_s) + v_s])


    # Create variables struct
    var = ctools.struct_symMX([(
            ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
            ctools.entry('v', shape=(Nu,), repeat=Nt)
    )])
    
    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    
    # Adjust the relevant constraints
    for t in range(Nt):
        varlb['v', t, :] = ulb - np.dot(K, xub)
        varub['v', t, :] = uub - np.dot(K, xlb)
        varlb['mean', t, :] = xlb
        varub['mean', t, :] = xub

    # Now build up constraints and objective
    obj = ca.MX(0)
    con = []
    for t in range(Nt):
        u_i = u_func(var['mean', t], var['v', t])
        z = ca.vertcat(var['mean', t], u_i)
        mean_i, var_i = gp_simple(z)
        con.append(var['mean', t + 1] - mean_i)
        obj += l_func(var['mean', t], u_i)
    obj += lf_func(var['mean', Nt])

    con = ca.vertcat(*con)
    conlb = np.zeros((Ny * Nt,))
    conub = np.zeros((Ny * Nt,))

    # Build solver object    
    nlp = dict(x=var, f=obj, g=con)
    opts = {}
    opts['ipopt.print_level'] = 0
    opts['ipopt.linear_solver'] = 'ma27'
    opts['ipopt.max_cpu_time'] = 1
    opts['print_time'] = False
    opts['expand'] = True
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Simulate
    mean = np.zeros((Nsim + 1, Ny))
    mean[0, :] = mean_0
    u = np.zeros((Nsim, Nu))

    for t in range(Nsim):
        solve_time = -time.time()

        # Fix initial state
        varlb['mean', 0, :] = mean[t, :]
        varub['mean', 0, :] = mean[t, :]
        varguess['mean', 0, :] = mean[t, :]
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
        solve_time += time.time()
        print("%d: %s - %f s" % (t, status, solve_time))
        v = optvar['v', 0, :]
        u[t, :] = np.array(u_func(mean[t, :], v)).flatten()
        mean[t + 1, :] = sim_system(mean[t, :], u[t, :].reshape((1, 2)), dt, dt)
        
    plt.figure()
    plt.clf()
    t = np.linspace(0.0, Nsim * dt, Nsim)
    for i in range(Nu):
        plt.subplot(Nu, 1, i + 1)
        plt.step(t, u[:, i], 'k')
        plt.ylabel('Flow  ' + str(i + 1) + ' [ml/s]')
        plt.suptitle('Control inputs', fontsize=16)
        plt.xlabel('Time [s]')
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
    plt.figure()
    plt.clf()
    t = np.linspace(0.0, Nsim * dt + dt, Nsim + 1)
    for i in range(Ny):
        plt.subplot(Ny, 1, i + 1)
        plt.plot(t, mean[:, i], 'b-')
        plt.ylabel('Tank ' + str(i + 1) + ' [cm]')
        plt.suptitle('MPC with ' + str(Nt) + ' step/' + 
                     str(horizon) + 's horizon', fontsize=16)
    plt.xlabel('Time [s]')
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=0.90)
    plt.show()

    return mean, u

