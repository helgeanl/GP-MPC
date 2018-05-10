# -*- coding: utf-8 -*-
"""
Model Predictive Control with Gaussian Process
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
from scipy.stats import norm
#from matplotlib.font_manager import FontProperties
from . gp_functions import gp_taylor_approx, gp

""" Test with discrete model

"""
def ode(x, u):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                        # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-8

    dxdt = [
                1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[2]**2
                    - 2*Cf*u[2] * (x[1] + lf*x[2]) / (x[0] + eps) + 2*mu*Fzr*u[1]),
                1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[2]*u[0]
                    + 2*Cf*(x[1] + lf*x[2]) / (x[0] + eps) - 2*Cf*u[2]
                    + 2*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                1/Iz * (2*lf*mu*Fzf*u[0]*u[2] + 2*lf*Cf*(x[1] + lf*x[2]) / (x[0] + eps)
                    - 2*lf*Cf*u[2] - 2*lr*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                x[2],
                x[0]*ca.cos(x[3]) - x[1]*ca.sin(x[3]),
                x[0]*ca.sin(x[3]) + x[1]*ca.cos(x[3])
            ]
    return np.array(dxdt)


class MPC:
    def __init__(self, X, Y, x0, x_sp, invK, hyper, horizon, sim_time, dt,
                Q=None, P=None, R=None, S=None, C=None,
                u0=None, ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
                feedback=True, method='TA', log=False, meanFunc='zero',
                costFunc='quad', solver_opts=None, simulator=None,test=None):
        #TODO: Remove 'test' and 'log'
        """ Initialize and build the MPC solver
        
        # Arguments:
            X: Training data matrix with inputs of size (N x Nx), where Nx is the number
                of inputs to the GP and N number of training points.
            Y: Training data matrix with outpyts of size (N x Ny), with Ny number of outputs.
            x0: Initial state (t0)
            x_sp: State set point
            invK: Array with the inverse covariance matrices of size (Ny x N x N),
                with Ny number of outputs from the GP.
            hyper: Array with hyperparameters [ell_1 .. ell_Nx sf sn].
            horizon: Horizon of control inputs
            sim_time: Simulation time
            dt: Sampling time
        # Optional Argumants:
            Q: State penalty matrix
            P: Termial penalty matrix
            R: Input penalty matrix
            S: Input rate of change penalty matrix
            u0: Initial input (t0-1)
            ulb: Lower boundry input
            uub: Upper boundry input
            xlb: Lower boundry state
            xub: Upper boundry state
            terminal_constraint: Terminal condition on the state
                    if None: No terminal constraint is used
                    if zero: Terminal state is equal to zero
                    if nonzero: Terminal state is bounded within the constraint 
            feedback:
            method: Method of propagating the uncertainty
                    Possible options:
                        'TA': Second order Taylor approximation
                        'ME': Mean equivalent approximation
            log: 
            meanFunc: 
            costFunc: Cost function to use in the objective
                'quad': Expected valaue of Quadratic Cost
                'sat':  Expected value of Saturating cost
            solver_opts: Additional options to pass to the NLP solver
                    e.g.: solver_opts['print_time'] = False
                          solver_opts['ipopt.tol'] = 1e-8
        """
        
        
        build_solver_time = -time.time()
        self.__Nsim = int(sim_time / dt) 
        Nt = int(horizon / dt)
        Ny = Y.shape[1]
        Nx = X.shape[1]
        Nu = Nx - Ny
        self.__dt = dt
        self.__Nt = Nt
        self.__Ny = Ny
        self.__Nx = Nx
        self.__Nu = Nu
        
        if P is None:
            P = np.eye(Ny)
        if Q is None:
            Q = np.eye(Ny)
        if R is None:
            R = np.eye(Nu) * 0.001
        if S is None:
            S = np.eye(Nu) * 0.001
            
        #TODO: Add slack constraint or remove
        lam = 14000 
    
        percentile = 0.95
        quantile_x = np.ones(Ny) * norm.ppf(percentile)
        quantile_u = np.ones(Nu) * norm.ppf(percentile)
        H_x = ca.MX.eye(Ny)
        H_u = ca.MX.eye(Nu)
        #K = np.ones((Nu, Ny)) * 0.001
        
        # Initial state
        self.__x0 = x0
        if u0 is None:
            self.__u0 = np.zeros(Nu)
        else:
            self.__u0 = u0

        # Initialize state variance with the noise variance
        self.__variance_0 = hyper[:, Nx + 1]
        mean_ref = x_sp
        self.__x_sp = x_sp
        
        # Create GP and cos function symbols
        mean_s = ca.MX.sym('mean', Ny)
        mean_ref_s = ca.MX.sym('mean_ref', Ny)
        covar_x_s = ca.MX.sym('covar', Ny, Ny)
        covar_s = ca.MX.sym('covar', Nx, Nx)
        v_s = ca.MX.sym('v', Nu)
        u_s = ca.MX.sym('u', Nu)
        z_s = ca.vertcat(mean_s, u_s)
        delta_u_s = ca.MX.sym('delta_u', Nu)
        
        K_s = ca.MX.sym('K', Nu, Ny)
    
        """ Select wich GP function to use """
        if method is 'ME':
            gp_func = ca.Function('gp_mean', [z_s, covar_s],
                                gp(invK, ca.MX(X), ca.MX(Y), ca.MX(hyper),
                                   z_s.T, meanFunc=meanFunc, log=log))
        elif method is 'TA':
            gp_func = ca.Function('gp_taylor_approx', [z_s, covar_s],
                                gp_taylor_approx(invK, ca.MX(X), ca.MX(Y),
                                                 ca.MX(hyper), z_s.T, covar_s,
                                                 meanFunc=meanFunc, diag=True, log=log))
        elif method is 'EM':
            gp_func = ca.Function('gp_taylor_approx', [z_s, covar_s],
                                gp_taylor_approx(invK, ca.MX(X), ca.MX(Y),
                                                 ca.MX(hyper), z_s.T, covar_s,
                                                 diag=True))
        else:
            raise NameError('No GP method called: ' + method)
    
        
    
        """ Define stage cost and terminal cost """
        if costFunc is 'quad':
            l_func = ca.Function('l', [mean_s, covar_x_s, u_s, delta_u_s, K_s],
                               [self.__cost_l(mean_s, mean_ref_s, covar_x_s, u_s, delta_u_s,
                                           ca.MX(Q), ca.MX(R), ca.MX(S), K_s)])
            lf_func = ca.Function('lf', [mean_s, covar_x_s],
                                   [self.__cost_lf(mean_s, mean_ref_s, 
                                                   covar_x_s,  ca.MX(P))])
        elif costFunc is 'sat':
            l_func = ca.Function('l', [mean_s, covar_x_s, u_s, delta_u_s, K_s],
                               [self.__cost_saturation_l(mean_s, mean_ref_s, 
                                    covar_x_s, u_s, delta_u_s, ca.MX(Q), ca.MX(R), 
                                    ca.MX(S), K_s)])
            lf_func = ca.Function('lf', [mean_s, covar_x_s],
                                   [self.__cost_saturation_lf(mean_s, 
                                        mean_ref_s, covar_x_s,  ca.MX(P))])
        else:
             raise NameError('No cost function called: ' + costFunc)
    
        """ Feedback function """
        if feedback:
            u_func = ca.Function('u', [mean_s, v_s, K_s],
                                 [ca.mtimes(K_s, mean_s) + v_s])
        else:
            u_func = ca.Function('u', [mean_s, v_s, K_s], [v_s])        
        self.__u_func = u_func
        
        
        #TODO: Clean this up
        """ 
        ======================================================================
        Remove, slip constraint
        ======================================================================
        """
        dx_s = ca.SX.sym('dx')
        dy_s = ca.SX.sym('dy')
        dpsi_s = ca.SX.sym('dpsi')
        delta_f_s = ca.SX.sym('delta_f') 
        lf  = 2.0 
        lr  = 2.0
        slip_min = -4 * np.pi / 180
        slip_max = 4 * np.pi / 180
        slip_f = ca.Function('slip_f', [dx_s, dy_s, dpsi_s, delta_f_s],
                             [(dy_s + lf*dpsi_s)/(dx_s + 1e-6)   - delta_f_s])
        slip_r = ca.Function('slip_r', [dx_s, dy_s, dpsi_s],
                             [(dy_s - lr*dpsi_s)/(dx_s + 1e-6)])
        
        """ 
        ======================================================================
        Discrete model test
        ======================================================================
        """
        # Then get nonlinear casadi functions
        # and rk4 discretization.
        # Define symbolic variables.
        x = ca.SX.sym("x", Ny)
        u = ca.SX.sym("u", Nu)
        ode_casadi = ca.Function("ode", [x,u], [ode(x,u)])
        
        k1 = ode_casadi(x, u)
        k2 = ode_casadi(x + dt/2*k1, u)
        k3 = ode_casadi(x + dt/2*k2, u)
        k4 = ode_casadi(x + dt*k3,u)
        xrk4 = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)    
        ode_rk4 = ca.Function("ode_rk4", [x,u], [xrk4])
        
        """ 
        ======================================================================
        Integrator
        ======================================================================
        """
        # Make integrator object.
        ode_integrator = dict(x=x,p=u, ode=ode(x,u))
        intoptions = {
            "abstol" : 1e-8,
            "reltol" : 1e-8,
            "tf" : dt,
        }
        self.__vdp = ca.integrator("int_ode",
            "cvodes", ode_integrator, intoptions)
        
        # Create variables struct
        var = ctools.struct_symMX([(
                ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
                ctools.entry('covariance', shape=(Ny * Ny,), repeat=Nt + 1),
                ctools.entry('v', shape=(Nu,), repeat=Nt),
                ctools.entry('K', shape=(Nu*Ny,), repeat=Nt),
                ctools.entry('eps', repeat=Nt),
        )])
        self.__var = var
        self.__num_var = var.size
        
        # Create parameter symbols
        mean_0_s = ca.MX.sym('mean_0', Ny)
        u_0_s = ca.MX.sym('u_0', Nu)
        covariance_0_s = ca.MX.sym('covariance_0', Ny * Ny)
        param_s = ca.vertcat(mean_0_s, mean_ref_s, covariance_0_s, u_0_s)
    
        # Decision variables boundries
        self.__varlb = var(-np.inf)
        self.__varub = var(np.inf)
        self.__var_init = var(0)
    
        # Adjust boundries
        for t in range(Nt):
            self.__varlb['covariance', t] = np.full((Ny * Ny,), 0) 
            self.__varlb['eps', t] = 0
            self.__varub['mean', t] = xub
            self.__varlb['mean', t] = xlb
            self.__varub['v', t] = uub
            self.__varlb['v', t] = ulb
            if not feedback:
                self.__varlb['K', t] = np.full((Nu * Ny,), 0)
                self.__varub['K', t] = np.full((Nu * Ny,), 0)
    
        # Build up constraints and objective
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
    
        # Set initial value
        con_eq.append(var['mean', 0] - mean_0_s)
        con_eq.append(var['covariance', 0] - covariance_0_s)
        u_past = u_0_s
        
        

        covar_t = ca.MX(Nx, Nx)

        for t in range(Nt):
            # Input to GP
            K_t = var['K', t].reshape((Nu, Ny))
            u_t = u_func(var['mean', t], var['v', t], K_t)
            z = ca.vertcat(var['mean', t], u_t)
            covar_x_t = var['covariance', t].reshape((Ny, Ny))
            covar_t[:Ny, :Ny] = covar_x_t
            
            #TODO: Fix this
            # Calculate next step
#            mean_next, covar_x_next = gp_func(z, covar_t)
            
            """
            ======================================================================
            """
            #TODO: Remove after test

            mean_next = ode_rk4(var["mean",t], var["v",t])
            covar_x_next = ca.MX(Ny, Ny)
            """
            ======================================================================
            """
            
            # Continuity constraints
            con_eq.append(ode_rk4(var["mean",t], var["v",t]) - var['mean', t + 1] )
            con_eq.append(covar_x_next.reshape((Ny * Ny, 1)) - var['covariance', t + 1])
            
            # Chance state constraints
#            con_ineq.append(mean_next + quantile_x * ca.sqrt(ca.diag(covar_x_next) ))
#            con_ineq_ub.append(xub)
#            con_ineq_lb.append(np.full((Ny,), -ca.inf))
#            con_ineq.append(mean_next - quantile_x * ca.sqrt(ca.diag(covar_x_next)))
#            con_ineq_ub.append(np.full((Ny,), ca.inf))
#            con_ineq_lb.append(xlb)
#            
#            con_ineq.append(var['mean', t ])
#            con_ineq_ub.append(xub)
#            con_ineq_lb.append(xlb)
    
            # Input constraints
#            con_ineq.append(u_t)
#            con_ineq_ub.extend(uub)
#            con_ineq_lb.append(ulb)
            
            # Slip angle constraint
            dx = var['mean', t, 0]
            dy = var['mean', t, 1]
            dpsi = var['mean', t, 2]
            delta_f = u_t[2]

            con_ineq.append(slip_f(dx, dy, dpsi, delta_f) - slip_max - var['eps', t])
            con_ineq_ub.append(0)
            con_ineq_lb.append(-np.inf)
            
            con_ineq.append(slip_min - slip_f(dx, dy, dpsi, delta_f) - var['eps', t])
            con_ineq_ub.append(0)
            con_ineq_lb.append(-np.inf)
            
            con_ineq.append(slip_r(dx, dy, dpsi) - slip_max - var['eps', t])
            con_ineq_ub.append(0)
            con_ineq_lb.append(-np.inf)
            
            con_ineq.append(slip_min - slip_r(dx, dy, dpsi) - var['eps', t])
            con_ineq_ub.append(0)
            con_ineq_lb.append(-np.inf)
            
            # Objective function
            u_delta = u_t - u_past
            obj += l_func(var['mean', t], covar_x_t, u_t, u_delta, K_t) + lam*var['eps', t]
            u_t = u_past
        obj += lf_func(var['mean', Nt], var['covariance', Nt].reshape((Ny, Ny)))
    
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))
    
        if terminal_constraint is not None:
            con_ineq.append(var['mean', Nt] - mean_ref)
            num_ineq_con += 1
            con_ineq_lb.append(np.full((Ny,), - terminal_constraint))
            con_ineq_ub.append(np.full((Ny,), terminal_constraint))
    
        con = ca.vertcat(*con_eq, *con_ineq)
        self.__conlb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.__conub = ca.vertcat(con_eq_ub, *con_ineq_ub)
    
        # Build solver object
        nlp = dict(x=var, f=obj, g=con, p=param_s)
        opts = {}
        opts['ipopt.print_level'] = 0
        opts['ipopt.mu_init'] = 0.01
        opts['ipopt.tol'] = 1e-8
        opts['ipopt.warm_start_init_point'] = 'yes'
        opts['ipopt.warm_start_bound_push'] = 1e-9
        opts['ipopt.warm_start_bound_frac'] = 1e-9
        opts['ipopt.warm_start_slack_bound_frac'] = 1e-9
        opts['ipopt.warm_start_slack_bound_push'] = 1e-9
        opts['ipopt.warm_start_mult_bound_push'] =  1e-9
        #opts['ipopt.fixed_variable_treatment'] = 'make_constraint'
        opts['print_time'] = False
        opts['verbose'] = False
        opts['expand'] = True
        if solver_opts is not None:
            opts.update(solver_opts)
        self.__solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
    
        # Simulate
        self.__mean = np.zeros((self.__Nsim + 1, Ny))
        self.__mean[0, :] = self.__x0
        self.__covariance = np.zeros((self.__Nsim + 1, Ny, Ny))
        self.__covariance[0] = np.diag(self.__variance_0)
        self.__u = np.zeros((self.__Nsim, Nu))

    
        # Initial guess of the warm start variables
        self.__lam_x0 = np.zeros(self.__num_var)
        self.__lam_g0 = 0
        
        # First prediction used in the NLP, used in plot later
        self.__var_prediction = np.zeros((Nt + 1, Ny))
        self.__mean_prediction = np.zeros((Nt + 1, Ny))
        
        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.__num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        
        
    def solve(self, simulator, sim_time=None, x0=None, u0=None, x_sp=None):
        """ Solve the optimal control problem
        
        # Arguments:
            simulator: Casadi integrator with the system DAE/ODE
            sim_time: Simulation length
            x0: Initial state
            u0: Initial input
            x_sp: State set point
        
        # Returns:
            mean: Simulated output using the optimal control inputs
            u: Optimal control inputs
        """

        Nt = self.__Nt
        Ny = self.__Ny
        Nu = self.__Nu
        dt = self.__dt

        # Initial state
        if x0 is not None:
            self.__x0 = x0
        if u0 is not None:
            self.__u0 = u0
        if x_sp is not None:
            self.__x_sp = x_sp

        if sim_time is not None:
            self.__Nsim = int(sim_time / dt)

        self.__mean = np.zeros((self.__Nsim + 1, Ny))
        self.__mean[0, :] = self.__x0
        self.__mean[0, :] = self.__x0
        self.__covariance = np.zeros((self.__Nsim + 1, Ny, Ny))
        self.__covariance[0] = np.diag(self.__variance_0)
        self.__u = np.zeros((self.__Nsim, Nu))
        self.__u[0, :] = self.__u0
        
        # Initial guess of the warm start variables
        self.__lam_x0 = np.zeros(self.__num_var)
        self.__lam_g0 = 0
        
        print('\nSolving MPC with %d step horizon' % Nt)
        for t in range(self.__Nsim):
            solve_time = -time.time()

            # Initial values
            param  = ca.vertcat(self.__mean[t, :], self.__x_sp, 
                                self.__covariance[t, :].flatten(), self.__u0)
    
            args = dict(x0=self.__var_init,
                        lbx=self.__varlb,
                        ubx=self.__varub,
                        lbg=self.__conlb,
                        ubg=self.__conub,
                        lam_x0=self.__lam_x0,
                        lam_g0=self.__lam_g0,
                        p=param)
    
            # Solve nlp
            sol = self.__solver(**args)
            status = self.__solver.stats()['return_status']
            optvar = self.__var(sol['x'])
            self.optvar=self.__var(sol['x'])
            self.__var_init = optvar
            self.__lam_x0 = sol['lam_x']
            self.__lam_g0 = sol['lam_g']
            solve_time += time.time()
            # Print status
            print("* t=%f: %s - %f sec" % (t * self.__dt, status, solve_time))

            if t == 0:
                 for i in range(Nt + 1):
                     cov = optvar['covariance', i, :].reshape((Ny, Ny))
                     
                     self.__var_prediction[i, :] = np.array(ca.diag(cov)).flatten()
                     self.__mean_prediction[i, :] = np.array(optvar['mean', i]).flatten()
    
            v = optvar['v', 0, :]
            K = np.array(optvar['K', 0]).reshape((Nu, Ny))
            self.__u[t, :] = np.array(self.__u_func(self.__mean[t, :], v, K)).flatten()
            self.__covariance[t + 1, :] = np.array(optvar['covariance', -1, :].reshape((Ny, Ny)))

            # Simulate the next step
            try:
                vdpargs = dict(x0=self.__mean[t , :] , p=self.__u[t,:])
                out = self.__vdp(**vdpargs)
                self.__mean[t + 1, :] = np.array(out["xf"]).flatten()
#                self.__mean[t + 1, :] = simulator(self.__mean[t, :], self.__u[t, :].reshape((1, Nu)),
#                                    dt, dt, noise=True)
            except RuntimeError:
                print('********************************')
                print('* Runtime error, adding jitter *')
                print('********************************')
#                self.__u = self.__u.clip(min=1e-6)
#                self.__mean = self.__mean.clip(min=1e-6)
#                self.__mean[t + 1, :] = simulator(self.__mean[t, :], self.__u[t, :].reshape((1, Nu)),
#                                    dt, dt, noise=True)
                
        return self.__mean, self.__u
        
        
    def __cost_lf(self, x, x_ref, covar_x, P, s=1):
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



    def __cost_saturation_lf(self, x, x_ref, covar_x, P):
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
    
    
    def __cost_saturation_l(self, x, x_ref, covar_x, u, delta_u, Q, R, S, K):
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
    
    
    def __cost_l(self, x, x_ref, covar_x, u, delta_u, Q, R, S, K, s=1):
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
    
        return sqnorm_x(x - x_ref, Q) + sqnorm_u(u, R) + sqnorm_u(delta_u, S) \
                + trace_x(Q, covar_x)  + trace_u(R, covar_u(covar_x, K))
    
    
    def __constraint(self, mean, covar, H, quantile):
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
    
    
    def plot(self, title=None,
             xnames=None, unames=None, time_unit = 's', numcols=2):
        x = self.__mean
        u = self.__u
        dt = self.__dt
        x_pred = self.__mean_prediction
        var_pred = self.__var_prediction
        
        Nu = self.__Nu
        Nt_sim, Nx = x.shape
        x_sp = self.__x_sp * np.ones((Nt_sim, Nx)) 

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

        if title is not None:
            fig_x.canvas.set_window_title(title)
        else:
            fig_x.canvas.set_window_title('Simulation')
    
        return fig_x, fig_u