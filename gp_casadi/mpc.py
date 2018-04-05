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
from matplotlib.font_manager import FontProperties
from simulation.four_tank import sim_system
from . gp_functions import gp_taylor_approx, gp


def cost_lf(x, x_ref, covar_x, P, s=1):
    # Cost function
    P_s = ca.SX.sym('Q', ca.MX.size(P))
    x_s = ca.SX.sym('x', ca.MX.size(x))
    covar_x_s = ca.SX.sym('K', ca.MX.size(covar_x))
    
    sqnorm_x = ca.Function('sqnorm_x', [x_s, P_s],
                           [ca.mtimes(x_s.T, ca.mtimes(P_s, x_s))])
    trace_x = ca.Function('trace_x', [P_s, covar_x_s],
                           [s * ca.trace(ca.mtimes(P_s, covar_x_s))])
    return sqnorm_x(x - x_ref, P) + trace_x(P, covar_x) 

                 
def cost_l(x, x_ref, covar_x, u, Q, R, K, s=1):
    Q_s = ca.SX.sym('Q', ca.MX.size(Q))
    R_s = ca.SX.sym('R', ca.MX.size(R))
    K_s = ca.SX.sym('K', ca.MX.size(K))
    x_s = ca.SX.sym('x', ca.MX.size(x))
    u_s = ca.SX.sym('u', ca.MX.size(u))
    covar_x_s = ca.SX.sym('K', ca.MX.size(covar_x))
    covar_u_s = ca.SX.sym('covar_u', ca.MX.size(R))

    sqnorm_x = ca.Function('sqnorm_x', [x_s, Q_s],
                           [ca.mtimes(x_s.T, ca.mtimes(Q_s, x_s))])
    sqnorm_u = ca.Function('sqnorm_u', [u_s, R_s],
                           [ca.mtimes(u_s.T, ca.mtimes(R_s, u_s))])
    covar_u  = ca.Function('covar_u', [covar_x_s, K_s],
                           [ca.mtimes(K_s, ca.mtimes(covar_x_s, K_s.T))])
    trace_u  = ca.Function('trace_u', [R_s, covar_u_s],
                           [s * ca.trace(ca.mtimes(R_s, covar_u_s))])
    trace_x  = ca.Function('trace_x', [Q_s, covar_x_s],
                           [s * ca.trace(ca.mtimes(Q_s, covar_x_s))])  

    return sqnorm_x(x - x_ref, Q) + sqnorm_u(u, R) + trace_x(Q, covar_x) \
                 + trace_u(R, covar_u(covar_x, K))


def mpc(X, Y, invK, hyper, method='TA'):
    """ Model Predictive Control
    
    # Arguments:
        X: Training data matrix with inputs of size (N x Nx), where Nx is the number
            of inputs to the GP and N number of training points.
        Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.
        invK: Array with the inverse covariance matrices of size (Ny x N x N), 
            with Ny number of outputs from the GP.
        hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
        method: Method of propagating the uncertainty
            Possible options:
                'TA': Second order taylor approximation
                'ME': Mean equivalent approximation without propagation 
    # Returns:
        mean: Simulated output
        u: Control inputs 
    """
    
    horizon = 120
    dt = 30

    sim_horizon = 600.0
    Nsim = int(sim_horizon / dt)

    Nt = int(horizon / dt)
    Ny = len(invK)
    Nx = X.shape[1]
    Nu = Nx - Ny
    
    P = np.eye(Ny) * 100
#    P = np.array([[6, .0, .0, .0], 
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
#    Q = np.array([[6, .0, .0, .0], 
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
    Q = np.eye(Ny) * 10
    R = np.eye(Nu) * 0.01
    #K = np.zeros((Nu, Ny)) # * .5
    K = np.array([[1.8, .0, .5, .0], 
                  [.0, 1.8, .0, .5]]) * 0.0
    # Bounds on u
    ulb = [0., 0.]
    uub = [60., 60.] 
    xlb = [0, 0, 0, 0]
    xub = [28, 28, 28, 28]
    terminal_constraint = 1e3

    # Initial state
    mean_0 = np.array([8., 10., 8., 18.])
    mean_ref = np.array([14., 14., 14.2, 21.3])
    variance_0 = np.ones(Ny) * 1e-5 * np.std(Y)
    
    mean_s = ca.MX.sym('mean', Ny)
    variance_s = ca.MX.sym('var', Ny)
    covar_x_s = ca.MX.sym('covar', Ny, Ny)
    v_s = ca.MX.sym('v', Nu)
    z_s = ca.vertcat(mean_s, v_s)
    
    if method is 'ME':
        gp_func = ca.Function('gp_mean', [z_s, variance_s], 
                            gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper), z_s.T, meanFunc='zero'))
    elif method is 'TA':
        gp_func = ca.Function('gp_taylor_approx', [z_s, variance_s],
                            gp_taylor_approx(invK, ca.MX(X), ca.MX(Y), 
                                             ca.MX(hyper), z_s.T, variance_s, diag=True))

    # Define stage cost and terminal cost
    l_func = ca.Function('l', [mean_s, covar_x_s, v_s], 
                           [cost_l(mean_s, ca.MX(mean_ref), covar_x_s, v_s, ca.MX(Q), ca.MX(R), ca.MX(K))])
    lf_func = ca.Function('lf', [mean_s, covar_x_s], 
                           [cost_lf(mean_s, ca.MX(mean_ref), covar_x_s,  ca.MX(P))])
    # Feedback function
    u_func = ca.Function('u', [mean_s, v_s], [ca.mtimes(ca.MX(K), mean_s - ca.MX(mean_ref)) + v_s])

    # Create variables struct
    var = ctools.struct_symMX([(
            ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
            ctools.entry('variance', shape=(Ny,), repeat=Nt + 1),
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
    con_mean = []
    con_var = []
    for t in range(Nt):
        u_i = u_func(var['mean', t], var['v', t])
        z = ca.vertcat(var['mean', t], u_i)
        mean_i, var_i = gp_func(z, var['variance', t])

        con_mean.append(var['mean', t + 1] - mean_i)
        con_var.append(var['variance', t + 1] - var_i)

        obj += l_func(var['mean', t], ca.diag(var['variance', t]), u_i)
    con_mean.append(var['mean', Nt] - mean_ref)
    obj += lf_func(var['mean', Nt], ca.diag(var['variance', Nt]))

    con = ca.vertcat(*con_mean, *con_var)
    conlb = np.zeros((Ny * Nt * 2 + Ny,))
    conub = np.zeros((Ny * Nt * 2 + Ny,))
    conlb[Ny * t] = - terminal_constraint
    conub[Ny * t] = terminal_constraint

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
    variance = np.zeros((Nsim + 1, Ny))
    variance[0, :] = variance_0
    u = np.zeros((Nsim, Nu))
    
    print('\nSolving MPC with %d step horizon' % Nt)
    for t in range(Nsim):
        solve_time = -time.time()

        # Fix initial state
        varlb['mean', 0, :] = mean[t, :]
        varub['mean', 0, :] = mean[t, :]
        varguess['mean', 0, :] = mean[t, :]
        varlb['variance', 0, :] = variance[t, :]
        varub['variance', 0, :] = variance[t, :]
        varguess['variance', 0, :] = variance[t, :]
        args = dict(x0=varguess,
                    lbx=varlb,
                    ubx=varub,
                    lbg=conlb,
                    ubg=conub)

        # Solve nlp
        sol = solver(**args)
        status = solver.stats()['return_status']
        optvar = var(sol['x'])
        solve_time += time.time()

        if t == 0:
            var_prediction = np.zeros((Nt + 1, Ny))
            mean_prediction = np.zeros((Nt + 1, Ny))
            for i in range(Nt + 1):
                var_prediction[i, :] = np.array(optvar['variance', i, :]).flatten()
                mean_prediction[i, :] = np.array(optvar['mean', i, :]).flatten()

        # Print status
        print("t=%d: %s - %f s" % (t * dt, status, solve_time))
        v = optvar['v', 0, :]
        u[t, :] = np.array(u_func(mean[t, :], v)).flatten()
        variance[t + 1, :] = np.array(optvar['variance', -1, :]).flatten()
        mean[t + 1, :] = sim_system(mean[t, :], u[t, :].reshape((1, 2)), dt, dt, noise=True)

    plt.figure()
    t = np.linspace(0.0, Nsim * dt, Nsim + 1)
    u_temp = np.vstack((u, u[-1, :]))
    for i in range(Nu):
        plt.subplot(Nu, 1, i + 1)
        plt.step(t, u_temp[:, i] , 'k', where='post')
        plt.ylabel('Flow  ' + str(i + 1) + ' [ml/s]')
        plt.suptitle('Control inputs', fontsize=16)
    plt.xlabel('Time [s]')
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=0.90)
    plt.savefig("mpc_control.png", bbox_inches="tight")
    plt.show()
    
    plt.figure()
    t = np.linspace(0.0, Nsim * dt, Nsim + 1)
    t2 = np.linspace(0.0, Nt * dt, Nt + 1)
    fontP = FontProperties()
    fontP.set_size('small')
    
    numcols = 2 
    numrows = 2
    ax = []
    for i in range(Ny):
        ax.append( plt.subplot(numrows, numcols, i + 1))
        plt.plot(t, mean[:, i], 'b-', marker='.', linewidth=1.0, label='Simulation')
        plt.axhline(y=mean_ref[i], color='g', linestyle='--', label='Set point')
        plt.errorbar(t2, mean_prediction[:, i], yerr=2 * np.sqrt(var_prediction[:, i]), 
                     linestyle='None', marker='.', color='r', label='1st prediction')
        #plt.plot(t2, mean_prediction[:, i], 'r.', label='1st prediction')
        #plt.gca().fill_between(t2.flat, mean_prediction[:, i] - 
        #       2 * np.sqrt(var_prediction[:, i]), mean_prediction[:, i] + 
        #       2 * np.sqrt(var_prediction[:, i]), color="#bbbbbb", label='95% conf prediction')
        plt.ylabel('Tank ' + str(i + 1) + ' [cm]')
        plt.suptitle('MPC with ' + str(Nt) + ' step/' + 
                     str(horizon) + 's horizon', fontsize=16)

        plt.xlabel('Time [s]')
    #ax[1].legend(prop=fontP, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0 )
    ax[1].legend(prop=fontP, loc="best" )
    #plt.tight_layout(pad=1, rect=[0,0,0.75,1])
    plt.tight_layout(pad=1)
    plt.subplots_adjust(top=0.90)
    #plt.subplots_adjust(right=0.7)
    plt.savefig("mpc.png", bbox_inches="tight")
    plt.show()

    return mean, u

