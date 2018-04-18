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
import scipy
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ctools
#from matplotlib.font_manager import FontProperties
from . gp_functions import gp_taylor_approx, gp


def cost_lf(x, x_ref, covar_x, P, s=1):
    """ Terminal cost function: Expected Value of Quadratic Cost
    """
    P_s = ca.SX.sym('Q', ca.MX.size(P))
    x_s = ca.SX.sym('x', ca.MX.size(x))
    covar_x_s = ca.SX.sym('covar_x', ca.MX.size(covar_x))

    sqnorm_x = ca.Function('sqnorm_x', [x_s, P_s],
                           [ca.mtimes(x_s.T, ca.mtimes(P_s, x_s))])
    trace_x = ca.Function('trace_x', [P_s, covar_x_s],
                           [s * ca.trace(ca.mtimes(P_s, covar_x_s))])
    return sqnorm_x(x - x_ref, P) + trace_x(P, covar_x)



def cost_saturation_lf(x, x_ref, covar_x, P):
    """ Terminal Cost function: Expected Value of Saturating Cost
    """
    Nx = ca.MX.size1(P)
    
    # Create symbols
    P_s = ca.SX.sym('P', Nx, Nx)
    x_s = ca.SX.sym('x', Nx)
    covar_x_s = ca.SX.sym('covar_z', Nx, Nx)

    Z_x = ca.SX.eye(Nx) #+ 2 * covar_x_s @ P_s 
    cost_x = ca.Function('cost_x', [x_s, P_s, covar_x_s], 
                       [1 - ca.exp(-(x_s.T @ ca.solve(Z_x.T, P_s.T).T @ x_s)) 
                               / ca.sqrt(ca.det(Z_x))])
    return cost_x(x - x_ref, P, covar_x)


def cost_saturation_l(x, x_ref, covar_x, u, Q, R, K):
    """ Stage Cost function: Expected Value of Saturating Cost
    """
    Nx = ca.MX.size1(Q)
    Nu = ca.MX.size1(R)
    
    # Create symbols
    Q_s = ca.SX.sym('Q', Nx, Nx)
    R_s = ca.SX.sym('Q', Nu, Nu)
    K_s = ca.SX.sym('K', ca.MX.size(K))
    x_s = ca.SX.sym('x', Nx)
    u_s = ca.SX.sym('x', Nu)
    covar_x_s = ca.SX.sym('covar_z', Nx, Nx)
    covar_u_s = ca.SX.sym('covar_u', ca.MX.size(R))
    
    covar_u  = ca.Function('covar_u', [covar_x_s, K_s],
                           [K_s @ covar_x_s @ K_s.T])

    Z_x = ca.SX.eye(Nx) + 2 * covar_x_s @ Q_s 
    Z_u = ca.SX.eye(Nu) + 2 * covar_u_s @ R_s 
    
    cost_x = ca.Function('cost_x', [x_s, Q_s, covar_x_s], 
                       [1 - ca.exp(-(x_s.T @ ca.solve(Z_x.T, Q_s.T).T @ x_s)) 
                               / ca.sqrt(ca.det(Z_x))])
    cost_u = ca.Function('cost_u', [u_s, R_s, covar_u_s], 
                       [1 - ca.exp(-(u_s.T @ ca.solve(Z_u.T, R_s.T).T @ u_s))
                               / ca.sqrt(ca.det(Z_u))])

    return cost_x(x - x_ref, Q, covar_x)  + cost_u(u, R, covar_u(covar_x, K))


def cost_l(x, x_ref, covar_x, u, Q, R, K, s=1):
    """ Stage cost function: Expected Value of Quadratic Cost
    """
    Q_s = ca.SX.sym('Q', ca.MX.size(Q))
    R_s = ca.SX.sym('R', ca.MX.size(R))
    K_s = ca.SX.sym('K', ca.MX.size(K))
    x_s = ca.SX.sym('x', ca.MX.size(x))
    u_s = ca.SX.sym('u', ca.MX.size(u))
    covar_x_s = ca.SX.sym('covar_x', ca.MX.size(covar_x))
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


def constraint(mean, covar, H, quantile):
    r = ca.SX.sym('r')
    mean_s = ca.SX.sym('mean', ca.MX.size(mean))
    covar_s = ca.SX.sym('r', ca.MX.size(covar))
    H_s = ca.SX.sym('H', 1, ca.MX.size2(H))
    
    con_func = ca.Function('con', [mean_s, covar_s, H_s, r], 
                           [H_s @ mean_s + r * ca.sqrt(H_s @ covar_s @ H_s.T)])
    con = []
    r = quantile

    for i in range(ca.MX.size1(mean)):
        con.append(con_func(mean, covar, H[i, :], quantile[i]))
    return con


def mpc(X, Y, x0, x_sp, invK, hyper, horizon, sim_time, dt, simulator,
        ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
        feedback=True, method='TA', log=False, meanFunc='zero', 
        costFunc='quad', plot=False):
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
                'TA': Second order Taylor approximation
                'ME': Mean equivalent approximation
    # Returns:
        mean: Simulated output
        u: Control inputs
    """
    build_solver_time = -time.time()
    Nsim = int(sim_time / dt)

    Nt = int(horizon / dt)
    Ny = Y.shape[1]
    Nx = X.shape[1]
    Nu = Nx - Ny

    P = np.eye(Ny) * 1
#    P = np.array([[6, .0, .0, .0],
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
#    Q = np.array([[6, .0, .0, .0],
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
    Q = np.eye(Ny) * 1
    R = np.eye(Nu) * 0.001
    
    percentile = 0.95
    quantile_x = np.ones(Ny) * scipy.stats.norm.ppf(percentile)
    quantile_u = np.ones(Nu) * scipy.stats.norm.ppf(percentile)
    H_x = ca.MX.eye(Ny)
    H_u = ca.MX.eye(Nu)
    #K = np.ones((Nu, Ny)) * 0.001

    # Initial state
    mean_0 = x0
    mean_ref = x_sp
  
    variance_0 = np.ones(Ny) * 1e-3

    mean_s = ca.MX.sym('mean', Ny)
    variance_s = ca.MX.sym('var', Ny)
    covar_x_s = ca.MX.sym('covar', Ny, Ny)
    v_s = ca.MX.sym('v', Nu)
    u_s = ca.MX.sym('u', Nu)
    z_s = ca.vertcat(mean_s, v_s)
    K_s = ca.MX.sym('K', Nu, Ny)

    if method is 'ME':
        gp_func = ca.Function('gp_mean', [z_s, variance_s],
                            gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper),
                               z_s.T, meanFunc=meanFunc, log=log))
    elif method is 'TA':
        gp_func = ca.Function('gp_taylor_approx', [z_s, variance_s],
                            gp_taylor_approx(invK, ca.MX(X), ca.MX(Y),
                                             ca.MX(hyper), z_s.T, variance_s,
                                             meanFunc=meanFunc, diag=True, log=log))
    else:
        raise NameError('No GP method called: ' + method)

    # Define stage cost and terminal cost
    if costFunc is 'quad':
        l_func = ca.Function('l', [mean_s, covar_x_s, u_s, K_s],
                           [cost_l(mean_s, ca.MX(mean_ref), covar_x_s, u_s,
                                       ca.MX(Q), ca.MX(R), K_s)])
        lf_func = ca.Function('lf', [mean_s, covar_x_s],
                               [cost_lf(mean_s, ca.MX(mean_ref), covar_x_s,  ca.MX(P))])
    elif costFunc is 'sat':
        l_func = ca.Function('l', [mean_s, covar_x_s, u_s, K_s],
                           [cost_saturation_l(mean_s, ca.MX(mean_ref), covar_x_s, u_s,
                                       ca.MX(Q), ca.MX(R), K_s)])
        lf_func = ca.Function('lf', [mean_s, covar_x_s],
                               [cost_saturation_lf(mean_s, ca.MX(mean_ref), covar_x_s,  ca.MX(P))])
    else:
         raise NameError('No cost function called: ' + costFunc)
    
    # Feedback function
    if feedback:
        u_func = ca.Function('u', [mean_s, v_s, K_s], [ca.mtimes(K_s,
                             mean_s) + v_s])
    else:
        u_func = ca.Function('u', [mean_s, v_s, K_s], [v_s])
    
    covar_u  = ca.Function('covar_u', [covar_x_s, K_s],
                       [K_s @ covar_x_s @ K_s.T])

    # Create variables struct
    var = ctools.struct_symMX([(
            ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
            ctools.entry('variance', shape=(Ny,), repeat=Nt + 1),
            #ctools.entry('variance_u', shape=(Nu,), repeat=Nt + 1),
            ctools.entry('v', shape=(Nu,), repeat=Nt),
            ctools.entry('K', shape=(Nu*Ny,)),
    )])
    num_var = var.size

    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    
    # Adjust boundries
    for t in range(Nt):
        varlb['variance', t, :] = np.ones(Ny) * 1e-6
        #varlb['variance_u', t, :] = np.ones(Nu) * 1e-8
        varlb['K', :] = np.ones(Nu * Ny) * 1e-8

    # Now build up constraints and objective
    obj = ca.MX(0)
    con_g = []
    con_state = []
    con_state_lb    = [] #np.fill((2 * Nt,), -np.inf) #np.tile(xlb, Nt)
    con_state_ub    = [] #np.fill((2 * Nt,), np.inf) #np.tile(xub, Nt)
    con_input = []
    con_input_lb    = [] # np.tile(ulb, Nt)
    con_input_ub    =  [] #np.tile(uub, Nt)

    for t in range(Nt):
        K_t = var['K'].reshape((Nu, Ny))
        u_t = u_func(var['mean', t], var['v', t], K_t)
        z = ca.vertcat(var['mean', t], u_t)
        covar_x_t = ca.diag(var['variance', t])
        #covar_u_t = covar_u(covar_x_t, K_t)
        
        mean_next, var_next = gp_func(z, var['variance', t])
        covar_u_next = covar_u(ca.diag(var_next), K_t)
        var_u_next = ca.diag(covar_u_next)
        
        con_g.append(var['mean', t + 1] - mean_next)
        con_g.append(var['variance', t + 1] - var_next)
        #con_g.append(var['variance_u', t + 1] - var_u_next)
        
#        con_state.extend(constraint(mean_next, covar_x_next, H_x, quantile_x))
#        con_input.extend(constraint(u_t, covar_u_next, H_u, quantile_u))

        con_state.append(mean_next + quantile_x * ca.sqrt(var_next) )
        con_state_ub.append(xub)
        con_state_lb.append(np.full((Ny,), -np.inf))

        con_state.append(mean_next - quantile_x * ca.sqrt(var_next))
        con_state_ub.append(np.full((Ny,), np.inf))
        con_state_lb.append(xlb)

        con_input.append(u_t + quantile_u * ca.sqrt(var_u_next))
        con_input_ub.extend(uub)
        con_input_lb.append(np.full((Nu,), -np.inf))
        
        con_input.append(u_t - quantile_u * ca.sqrt(var_u_next))
        con_input_ub.append(np.full((Nu,), np.inf))
        con_input_lb.append(ulb)
        
        obj += l_func(var['mean', t], covar_x_t, u_t, K_t)
    obj += lf_func(var['mean', Nt], ca.diag(var['variance', Nt]))
    
    if terminal_constraint is not None:
        con_g.append(var['mean', Nt] - mean_ref)
        con_g_lb = np.zeros((Ny * Nt * 2  + Ny,))
        con_g_ub = np.zeros((Ny * Nt * 2  + Ny,))
#        con_g_lb = np.zeros((Ny * Nt * 2  + Nu * Nt + Ny,))
#        con_g_ub = np.zeros((Ny * Nt * 2  + Nu * Nt + Ny,))
        con_g_lb.append(- terminal_constraint)
        con_g_ub.append(terminal_constraint)
    else:
        con_g_lb = np.zeros((Ny * Nt * 2 ,))
        con_g_ub = np.zeros((Ny * Nt * 2 ,))
#        con_g_lb = np.zeros((Ny * Nt * 2 + Nu * Nt,))
#        con_g_ub = np.zeros((Ny * Nt * 2 + Nu * Nt,))
    con = ca.vertcat(*con_g, *con_state, *con_input)
    conlb = ca.vertcat(*con_g_lb, *con_state_lb, *con_input_lb)
    conub = ca.vertcat(*con_g_ub, *con_state_ub, *con_input_ub)
    
    # Build solver object
    nlp = dict(x=var, f=obj, g=con)
    opts = {}
    opts['ipopt.print_level'] = 0
    opts['ipopt.linear_solver'] = 'ma27'
    opts['ipopt.max_cpu_time'] = 1
    opts['ipopt.warm_start_init_point'] = 'no'
    opts['print_time'] = False
    opts['verbose'] = False
    opts['expand'] = True
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Simulate
    mean = np.zeros((Nsim + 1, Ny))
    mean[0, :] = mean_0
    variance = np.zeros((Nsim + 1, Ny))
    variance[0, :] = variance_0
    u = np.zeros((Nsim, Nu))
    u[0,:] = np.array(ulb) + 10
    #K = np.ones((Nu, Ny)) * 1e-3
    
    # Warm start each round
    lam_x0 = np.zeros(num_var)
    lam_g0 = 0

    build_solver_time += time.time()
    print('\n________________________________________')
    print('# Time to build mpc solver: %f sec' % build_solver_time)
    print('----------------------------------------')
    print('\nSolving MPC with %d step horizon' % Nt)
    for t in range(Nsim):
        solve_time = -time.time()

        # Fix initial state
        varlb['mean', 0, :] = mean[t, :]
        varub['mean', 0, :] = mean[t, :]
        varlb['variance', 0, :] = variance[t, :]
        varub['variance', 0, :] = variance[t, :]

        varguess['mean', 0, :] = mean[t, :]
        varguess['variance', 0, :] = variance[t, :]
#        varguess['variance_u', 0, :] = np.ones(Nu) * 1e-6
#        varguess['v', 0, :] = u[t, :] - np.dot(K, mean[t, :])

        args = dict(x0=varguess,
                    lbx=varlb,
                    ubx=varub,
                    lbg=conlb,
                    ubg=conub,
                    lam_x0=lam_x0,
                    lam_g0=lam_g0)

        # Solve nlp
        sol = solver(**args)
        status = solver.stats()['return_status']
        optvar = var(sol['x'])
        lam_x0 = sol['lam_x']
        lam_g0 = sol['lam_g']
        solve_time += time.time()

        if t == 0:
             var_prediction = np.zeros((Nt + 1, Ny))
             mean_prediction = np.zeros((Nt + 1, Ny))
             for i in range(Nt + 1):
                 var_prediction[i, :] = np.array(optvar['variance', i, :]).flatten()
                 mean_prediction[i, :] = np.array(optvar['mean', i, :]).flatten()

        v = optvar['v', 0, :]
        K = np.array(optvar['K']).reshape((Nu, Ny))
        print('K')
        print(K)
        u[t, :] = np.array(u_func(mean[t, :], v, K)).flatten()
        variance[t + 1, :] = np.array(optvar['variance', -1, :]).flatten()
        
        # Print status
        print("* t=%d: %s - %f sec" % (t * dt, status, solve_time))
        
        # Simulate the next step
        sim_time = -time.time()
        try:
            mean[t + 1, :] = simulator(mean[t, :], u[t, :].reshape((1, 2)),
                                dt, dt, noise=True)
        except RuntimeError:
            print('********************************')
            print('* Runtime error, adding jitter *')
            print('********************************')
            u = u.clip(min=1e-6)
            mean = mean.clip(min=1e-6)
            mean[t + 1, :] = simulator(mean[t, :], u[t, :].reshape((1, 2)),
                                dt, dt, noise=True)
        sim_time += time.time()
        print('\t Sim time: %f sec' % sim_time)
    if plot:
        x_sp = np.ones((Nsim + 1, Ny)) * x_sp
        fig_x, fig_u = plot_mpc(x=mean, u=u, dt=dt, x_pred=mean_prediction,
                   var_pred=var_prediction, x_sp=x_sp,
                   title='MPC with %d step/ %d s horizon - GP: %s' % (Nt, horizon, method)
               )
        fig_x.savefig("mpc.png", bbox_inches="tight")
    return mean, u




def mpc_single(X, Y, x0, x_sp, invK, hyper, horizon, sim_time, dt, simulator,
        ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
        feedback=True, method='TA', log=False, meanFunc='zero', 
        costFunc='quad', plot=False):
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
                'TA': Second order Taylor approximation
                'ME': Mean equivalent approximation
    # Returns:
        mean: Simulated output
        u: Control inputs
    """
    build_solver_time = -time.time()
    Nsim = int(sim_time / dt)

    Nt = int(horizon / dt)
    Ny = Y.shape[1]
    Nx = X.shape[1]
    Nu = Nx - Ny

    P = np.eye(Ny) * 1
#    P = np.array([[6, .0, .0, .0],
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
#    Q = np.array([[6, .0, .0, .0],
#                  [.0, 6, .0, .0],
#                  [.0, .0, 6, .0],
#                  [.0, .0, .0, 31]])
    Q = np.eye(Ny) * 1
    R = np.eye(Nu) * 0.001
    
    percentile = 0.95
    quantile_x = np.ones(Ny) * scipy.stats.norm.ppf(percentile)
    quantile_u = np.ones(Nu) * scipy.stats.norm.ppf(percentile)
    H_x = ca.MX.eye(Ny)
    H_u = ca.MX.eye(Nu)
    #K = np.ones((Nu, Ny)) * 0.001

    # Initial state
    mean_0 = x0
    mean_ref = x_sp
  
    variance_0 = np.ones(Ny) * 1e-3

    mean_s = ca.MX.sym('mean', Ny)
    variance_s = ca.MX.sym('var', Ny)
    covar_x_s = ca.MX.sym('covar', Ny, Ny)
    v_s = ca.MX.sym('v', Nu)
    u_s = ca.MX.sym('u', Nu)
    z_s = ca.vertcat(mean_s, v_s)
    K_s = ca.MX.sym('K', Nu, Ny)

    if method is 'ME':
        gp_func = ca.Function('gp_mean', [z_s, variance_s],
                            gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper),
                               z_s.T, meanFunc=meanFunc, log=log))
    elif method is 'TA':
        gp_func = ca.Function('gp_taylor_approx', [z_s, variance_s],
                            gp_taylor_approx(invK, ca.MX(X), ca.MX(Y),
                                             ca.MX(hyper), z_s.T, variance_s,
                                             meanFunc=meanFunc, diag=True, log=log))
    else:
        raise NameError('No GP method called: ' + method)

    # Define stage cost and terminal cost
    if costFunc is 'quad':
        l_func = ca.Function('l', [mean_s, covar_x_s, u_s, K_s],
                           [cost_l(mean_s, ca.MX(mean_ref), covar_x_s, u_s,
                                       ca.MX(Q), ca.MX(R), K_s)])
        lf_func = ca.Function('lf', [mean_s, covar_x_s],
                               [cost_lf(mean_s, ca.MX(mean_ref), covar_x_s,  ca.MX(P))])
    elif costFunc is 'sat':
        l_func = ca.Function('l', [mean_s, covar_x_s, u_s, K_s],
                           [cost_saturation_l(mean_s, ca.MX(mean_ref), covar_x_s, u_s,
                                       ca.MX(Q), ca.MX(R), K_s)])
        lf_func = ca.Function('lf', [mean_s, covar_x_s],
                               [cost_saturation_lf(mean_s, ca.MX(mean_ref), covar_x_s,  ca.MX(P))])
    else:
         raise NameError('No cost function called: ' + costFunc)
    
    # Feedback function
    if feedback:
        u_func = ca.Function('u', [mean_s, v_s, K_s], [ca.mtimes(K_s,
                             mean_s) + v_s])
    else:
        u_func = ca.Function('u', [mean_s, v_s, K_s], [v_s])
    
    covar_u  = ca.Function('covar_u', [covar_x_s, K_s],
                       [K_s @ covar_x_s @ K_s.T])

    # Create variables struct
    var = ctools.struct_symMX([(
            #ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
            #ctools.entry('variance', shape=(Ny,), repeat=Nt + 1),
            #ctools.entry('variance_u', shape=(Nu,), repeat=Nt + 1),
            ctools.entry('v', shape=(Nu,), repeat=Nt),
            ctools.entry('K', shape=(Nu*Ny,)),
    )])
    num_var = var.size

    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    
    # Adjust boundries
    for t in range(Nt):
        #varlb['variance', t, :] = np.ones(Ny) * 1e-6
        #varlb['variance_u', t, :] = np.ones(Nu) * 1e-8
        varlb['K', :] = np.ones(Nu * Ny) * 1e-8

    # Now build up constraints and objective
    obj = ca.MX(0)
    con_g = []
    con_state = []
    con_state_lb    = [] #np.fill((2 * Nt,), -np.inf) #np.tile(xlb, Nt)
    con_state_ub    = [] #np.fill((2 * Nt,), np.inf) #np.tile(xub, Nt)
    con_input = []
    con_input_lb    = [] # np.tile(ulb, Nt)
    con_input_ub    =  [] #np.tile(uub, Nt)
    
    param_s = ctools.struct_symMX([(
            ctools.entry('mean', shape=(Ny,)),
            ctools.entry('variance', shape=(Ny,)),
            #ctools.entry('variance_u', shape=(Nu,), repeat=Nt + 1),
    )])
    mean_s = ca.MX.sym('mean', Ny)
    variance_s = ca.MX.sym('variance', Ny)
    param_s = ca.vertcat(mean_s, variance_s)

    for t in range(Nt):
        
        K_t = var['K'].reshape((Nu, Ny))
        u_t = u_func(mean_s, var['v', t], K_t)
        z = ca.vertcat(mean_s, u_t)
        covar_x_t = ca.diag(variance_s)
        #covar_u_t = covar_u(covar_x_t, K_t)
        
        mean_next, var_next = gp_func(z, variance_s)
        covar_u_next = covar_u(ca.diag(var_next), K_t)
        var_u_next = ca.diag(covar_u_next)
        
        
        #con_g.append(var['mean', t + 1] - mean_next)
        #con_g.append(var['variance', t + 1] - var_next)
        #con_g.append(var['variance_u', t + 1] - var_u_next)
        
#        con_state.extend(constraint(mean_next, covar_x_next, H_x, quantile_x))
#        con_input.extend(constraint(u_t, covar_u_next, H_u, quantile_u))

        con_state.append(mean_next + quantile_x * ca.sqrt(var_next) )
        con_state_ub.append(xub)
        con_state_lb.append(np.full((Ny,), -np.inf))

        con_state.append(mean_next - quantile_x * ca.sqrt(var_next))
        con_state_ub.append(np.full((Ny,), np.inf))
        con_state_lb.append(xlb)

        con_input.append(u_t + quantile_u * ca.sqrt(var_u_next))
        con_input_ub.extend(uub)
        con_input_lb.append(np.full((Nu,), -np.inf))
        
        con_input.append(u_t - quantile_u * ca.sqrt(var_u_next))
        con_input_ub.append(np.full((Nu,), np.inf))
        con_input_lb.append(ulb)
        
        obj += l_func(mean_s, covar_x_t, u_t, K_t)
        
        mean_s = mean_next
        variance_s = var_next
    obj += lf_func(mean_s, ca.diag(variance_s))
    
    if terminal_constraint is not None:
        con_g.append(mean_s - mean_ref)
        con_g_lb = np.zeros((Ny * Nt * 2  + Ny,))
        con_g_ub = np.zeros((Ny * Nt * 2  + Ny,))
#        con_g_lb = np.zeros((Ny * Nt * 2  + Nu * Nt + Ny,))
#        con_g_ub = np.zeros((Ny * Nt * 2  + Nu * Nt + Ny,))
        con_g_lb.append(- terminal_constraint)
        con_g_ub.append(terminal_constraint)
    else:
        con_g_lb = np.zeros((Ny * Nt * 2 ,))
        con_g_ub = np.zeros((Ny * Nt * 2 ,))
#        con_g_lb = np.zeros((Ny * Nt * 2 + Nu * Nt,))
#        con_g_ub = np.zeros((Ny * Nt * 2 + Nu * Nt,))
    con = ca.vertcat(*con_state, *con_input)
    conlb = ca.vertcat(*con_state_lb, *con_input_lb)
    conub = ca.vertcat(*con_state_ub, *con_input_ub)
    
    # Build solver object
    nlp = dict(x=var, f=obj, g=con, p=param_s)
    opts = {}
    opts['ipopt.print_level'] = 0
    opts['ipopt.linear_solver'] = 'ma27'
    opts['ipopt.max_cpu_time'] = 1
    opts['ipopt.warm_start_init_point'] = 'no'
    opts['print_time'] = False
    opts['verbose'] = False
    opts['expand'] = True
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Simulate
    mean = np.zeros((Nsim + 1, Ny))
    mean[0, :] = mean_0
    variance = np.zeros((Nsim + 1, Ny))
    variance[0, :] = variance_0
    u = np.zeros((Nsim, Nu))
    u[0,:] = np.array(ulb) + 10
    #K = np.ones((Nu, Ny)) * 1e-3
    
    # Warm start each round
    lam_x0 = np.zeros(num_var)
    lam_g0 = 0

    build_solver_time += time.time()
    print('\n________________________________________')
    print('# Time to build mpc solver: %f sec' % build_solver_time)
    print('----------------------------------------')
    print('\nSolving MPC with %d step horizon' % Nt)
    for t in range(Nsim):
        solve_time = -time.time()

        # Fix initial state
        #varlb['mean', 0, :] = mean[t, :]
        #varub['mean', 0, :] = mean[t, :]
        #varlb['variance', 0, :] = variance[t, :]
        #varub['variance', 0, :] = variance[t, :]

        param  = ca.vertcat(mean[t, :], variance[t, :])
        
#        varguess['variance_u', 0, :] = np.ones(Nu) * 1e-6
#        varguess['v', 0, :] = u[t, :] - np.dot(K, mean[t, :])

        args = dict(x0=varguess,
                    lbx=varlb,
                    ubx=varub,
                    lbg=conlb,
                    ubg=conub,
                    lam_x0=lam_x0,
                    lam_g0=lam_g0,
                    p=param)

        # Solve nlp
        sol = solver(**args)
        status = solver.stats()['return_status']
        optvar = var(sol['x'])
        lam_x0 = sol['lam_x']
        lam_g0 = sol['lam_g']
        solve_time += time.time()

#        if t == 0:
#             var_prediction = np.zeros((Nt + 1, Ny))
#             mean_prediction = np.zeros((Nt + 1, Ny))
#             for i in range(Nt + 1):
#                 var_prediction[i, :] = np.array(optvar['variance', i, :]).flatten()
#                 mean_prediction[i, :] = np.array(optvar['mean', i, :]).flatten()

        v = optvar['v', 0, :]
        K = np.array(optvar['K']).reshape((Nu, Ny))
        print('K')
        print(K)
        u[t, :] = np.array(u_func(mean[t, :], v, K)).flatten()
#        variance[t + 1, :] = np.array(optvar['variance', -1, :]).flatten()
        
        # Print status
        print("* t=%d: %s - %f sec" % (t * dt, status, solve_time))
        
        # Simulate the next step
        sim_time = -time.time()
        try:
            mean[t + 1, :] = simulator(mean[t, :], u[t, :].reshape((1, 2)),
                                dt, dt, noise=True)
        except RuntimeError:
            print('********************************')
            print('* Runtime error, adding jitter *')
            print('********************************')
            u = u.clip(min=1e-6)
            mean = mean.clip(min=1e-6)
            mean[t + 1, :] = simulator(mean[t, :], u[t, :].reshape((1, 2)),
                                dt, dt, noise=True)
        sim_time += time.time()
        print('\t Sim time: %f sec' % sim_time)
    if plot:
        x_sp = np.ones((Nsim + 1, Ny)) * x_sp
        fig_x, fig_u = plot_mpc(x=mean, u=u, dt=dt,
                    x_sp=x_sp,
                   title='MPC with %d step/ %d s horizon - GP: %s' % (Nt, horizon, method)
               )
        fig_x.savefig("mpc.png", bbox_inches="tight")
    return mean, u






def plot_mpc(x, u, dt, x_pred=None, var_pred=None, x_sp=None, title=None,
             xnames=None, unames=None, time_unit = 's', numcols=2):
    Nu = np.size(u, 1)
    Nt_sim, Nx = x.shape
    if x_pred is not None:
        Nt_horizon = np.size(x_pred, 0)
        t_horizon = np.linspace(0.0, Nt_horizon * dt, Nt_horizon)
    if xnames is None:
        xnames = ['State %d' % (i + 1) for i in range(Nx)]
    if unames is None:
        unames = ['Control %d' % (i + 1) for i in range(Nu)]

    t = np.linspace(0.0, Nt_sim * dt, Nt_sim)
    u = np.vstack((u, u[-1, :]))
    numcols = 2
    numrows = int(np.ceil(Nx / numcols))

    fig_u = plt.figure()
    for i in range(Nu):
        ax = fig_u.add_subplot(Nu, 1, i + 1)
        ax.step(t, u[:, i] , 'k', where='post')
        ax.set_ylabel(unames[i])
        ax.set_xlabel('Time [' + time_unit + ']')
    fig_u.canvas.set_window_title('Control inputs')

    fig_x = plt.figure()
    for i in range(Nx):
        ax = fig_x.add_subplot(numrows, numcols, i + 1)
        ax.plot(t, x[:, i], 'b-', marker='.', linewidth=1.0, label='Simulation')
        if x_sp is not None:
            ax.plot(t, x_sp[:, i], color='g', linestyle='--', label='Setpoint')
        if x_pred is not None:
            ax.errorbar(t_horizon, x_pred[:, i], yerr=2 * np.sqrt(var_pred[:, i]),
                        linestyle='None', marker='.', color='r', label='1st prediction')
        plt.legend(loc='best')
        ax.set_ylabel(xnames[i])
        ax.set_xlabel('Time [' + time_unit + ']')
    #ax[1].legend(prop=fontP, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0 )
    #ax[1].legend(prop=fontP, loc="best" )
    #plt.tight_layout(pad=1, rect=[0,0,0.75,1])
    #plt.tight_layout(pad=.1)
    if title is not None:
        fig_x.canvas.set_window_title(title)

    return fig_x, fig_u
