# -*- coding: utf-8 -*-
"""
Model Predictive Control with Gaussian Process
@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import casadi.tools as ctools
from scipy.stats import norm
import scipy.linalg


class MPC:
    def __init__(self, horizon, model, gp=None,
                 Q=None, P=None, R=None, S=None, lam=None,
                 ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
                 feedback=True, gp_method='TA', costFunc='quad', solver_opts=None,
                 discrete_method='gp', inequality_constraints=None, num_con_par=0
                 ):

        """ Initialize and build the MPC solver

        # Arguments:
            horizon: Prediction horizon with control inputs
            model: System model

        # Optional Argumants:
            gp: GP model
            Q: State penalty matrix
            P: Termial penalty matrix
            R: Input penalty matrix
            S: Input rate of change penalty matrix
            lam: Slack variable penalty
            ulb: Lower boundry input
            uub: Upper boundry input
            xlb: Lower boundry state
            xub: Upper boundry state
            terminal_constraint: Terminal condition on the state
                    * if None: No terminal constraint is used
                    * if zero: Terminal state is equal to zero
                    * if nonzero: Terminal state is bounded within +/- the constraint
            feedback: If true, use a feedback function u= Kx + v
            gp_method: Method of propagating the uncertainty
                    Possible options:
                        'TA': Second order Taylor approximation
                        'ME': Mean equivalent approximation

            costFunc: Cost function to use in the objective
                    'quad': Expected valaue of Quadratic Cost
                    'sat':  Expected value of Saturating cost
            solver_opts: Additional options to pass to the NLP solver
                    e.g.: solver_opts['print_time'] = False
                          solver_opts['ipopt.tol'] = 1e-8
            discrete_method: 'gp', 'rk4', or 'exact'
            num_con_par: Number of parameters to pass to the inequality function
            inequality_constraints: Additional inequality constraints
                    Use a function with inputs (x, covar, u, eps) and
                    that returns a dictionary with inequality constraints and limits.
                        e.g. cons = dict(con_ineq=con_ineq_array,
                                         con_ineq_lb=con_ineq_lb_array,
                                         con_ineq_ub=con_ineq_ub_array
                                    )
        """


        build_solver_time = -time.time()
        dt = model.sampling_time()
        Ny, Nu, Np = model.size()
        Nx = Nu + Ny
        Nt = int(horizon / dt)

        self.__dt = dt
        self.__Nt = Nt
        self.__Ny = Ny
        self.__Nx = Nx
        self.__Nu = Nu
        self.__num_con_par = num_con_par
        self.__model = model
        self.__gp = gp
        self.__feedback = feedback
        self.__discrete_method = discrete_method
        

        """ Default penalty values """
        if P is None:
            P = np.eye(Ny)
        if Q is None:
            Q = np.eye(Ny)
        if R is None:
            R = np.eye(Nu) * 0.01
        if S is None:
            S = np.eye(Nu) * 0.1
        if lam is None:
            lam = 1000
            
        self.__Q = Q
        self.__P = P
        self.__R = R
        self.__S = S
        
        if xub is None:
            xub = np.full((Ny), np.inf)
        if xlb is None:
            xlb = np.full((Ny), -np.inf)
        if uub is None:
            uub = np.full((Nu), np.inf)
        if ulb is None:
            ulb = np.full((Nu), -np.inf)
            
      
        eps_sqrt =0
        #TODO: Clean this up
        percentile = 0.95
        quantile_x = np.ones(Ny) * norm.ppf(percentile)
        quantile_u = np.ones(Nu) * norm.ppf(percentile)
        Hx = ca.MX.eye(Ny)
        Hu = ca.MX.eye(Nu)


        # Create parameter symbols
        mean_0_s       = ca.MX.sym('mean_0', Ny)
        mean_ref_s     = ca.MX.sym('mean_ref', Ny)
        u_0_s          = ca.MX.sym('u_0', Nu)
        covariance_0_s = ca.MX.sym('covariance_0', Ny * Ny)
        K_s            = ca.MX.sym('K', Nu * Ny)
        con_par        = ca.MX.sym('con_par', num_con_par)
        param_s        = ca.vertcat(mean_0_s, mean_ref_s, covariance_0_s, 
                                    u_0_s, K_s, con_par)

        

        """ Select wich GP function to use """
        if discrete_method is 'gp':
            self.__gp.set_method(gp_method)
        
        if solver_opts['expand'] is not False and discrete_method is 'exact':
            raise NameError("Can't use exact discrete system with expanded graph")

        # Initialize state variance with the GP noise variance
        if gp is not None:
            self.__variance_0 = gp.noise_variance()
        else:
            self.__variance_0 = np.full((Ny), 0)
            print('var_0')

        # Define which cost function to use
        self.__set_cost_function(costFunc, mean_ref_s)
        

        """ Feedback function """
        mean_s = ca.MX.sym('mean', Ny)
        v_s = ca.MX.sym('v', Nu)
        if feedback:
            u_func = ca.Function('u', [mean_s, v_s, K_s],
                                 [v_s + ca.mtimes(K_s.reshape((Nu, Ny)), mean_s)])
        else:
            u_func = ca.Function('u', [mean_s, v_s, K_s], [v_s])
        self.__u_func = u_func


        """ Create variables struct """
        var = ctools.struct_symMX([(
                ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
                ctools.entry('covariance', shape=(Ny * Ny,), repeat=Nt + 1),
                ctools.entry('v', shape=(Nu,), repeat=Nt),
#                ctools.entry('sparse', shape=(Ny * Ny,), repeat=Nt + 1),
                ctools.entry('eps', repeat=Nt),
        )])
        self.__var = var
        self.__num_var = var.size

        # Decision variables boundries
        self.__varlb = var(-np.inf)
        self.__varub = var(np.inf)

        """ Adjust hard boundries """
        for t in range(Nt):
            for i in range(Ny):
                # Lower boundry of diagonal
                self.__varlb['covariance', t, i + i*Ny] = 0
            self.__varlb['eps', t] = 0
            if xub is not None:
                self.__varub['mean', t] = xub
            if xlb is not None:
                self.__varlb['mean', t] = xlb
         
        
        """ Input covariance matrix """
        covar_x_s = ca.MX.sym('covar_x', Ny, Ny)
        covar_x_sx = ca.SX.sym('cov_x', Ny, Ny)
        K_sx = ca.SX.sym('K', Nu, Ny)

        covar_u_func = ca.Function('cov_u', [covar_x_sx, K_sx], 
                                   [K_sx @ covar_x_sx @ K_sx.T])

        cov_xu_func = ca.Function('cov_xu', [covar_x_sx, K_sx], 
                                  [covar_x_sx @ K_sx.T])

        covar_s = ca.MX(Nx, Nx)
        cov_xu = cov_xu_func(covar_x_s, K_s.reshape((Nu, Ny)))
        covar_s[:Ny, :Ny] = covar_x_s
        covar_s[Ny:, Ny:] = covar_u_func(covar_x_s, K_s.reshape((Nu, Ny)))
        covar_s[Ny:, :Ny] = cov_xu.T
        covar_s[:Ny, Ny:] = cov_xu
        covar_func = ca.Function('covar', [covar_x_s], [covar_s])
        
        
        """ Set initial values """
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(var['mean', 0] - mean_0_s)
        con_eq.append(var['covariance', 0] - covariance_0_s)
        u_past = u_0_s
        
        cholesky = ca.Function('cholesky', [covar_x_sx], [ca.chol(covar_x_sx).T])
       

        """ Build constraints """
        for t in range(Nt):
            # Input to GP
            mean_t = var['mean', t]
            u_t = u_func(mean_t, var['v', t], K_s)

            covar_x_t = var['covariance', t].reshape((Ny, Ny))
            covar_t = covar_func(covar_x_t)

            # Choose between GP and RK4 for next step
            if discrete_method is 'rk4':
                mean_next_pred = model.rk4(mean_t, u_t,[])
                covar_x_next_pred = ca.MX(Ny, Ny)
            elif discrete_method is 'exact':
                mean_next_pred = model.Integrator(x0=mean_t, p=u_t)['xf']
                covar_x_next_pred = ca.MX(Ny, Ny)
            else:
                mean_next_pred, covar_x_next_pred = self.__gp.predict(mean_t, u_t, covar_t)
                
            S = cholesky(covar_x_next_pred)

            # Continuity constraints
            mean_next = var['mean', t + 1]
            covar_x_next = var['covariance', t + 1]
            con_eq.append(mean_next_pred - mean_next )
            con_eq.append(covar_x_next_pred.reshape((Ny * Ny, 1)) - covar_x_next)

            # Chance state constraints
            cons = self.__constraint(mean_next_pred, covar_x_next_pred, Hx, quantile_x, xub, xlb)
            con_ineq.extend(cons['con'])
            con_ineq_lb.extend(cons['con_lb'])
            con_ineq_ub.extend(cons['con_ub'])

            # Input constraints
            cov_u = covar_u_func(covar_x_t, K_s.reshape((Nu, Ny)))
#            cons = self.__constraint(u_t, cov_u, Hu, quantile_u, uub, ulb)
#            con_ineq.extend(cons['con_ineq'])
#            con_ineq_lb.extend(cons['con_ineq_lb'])
#            con_ineq_ub.extend(cons['con_ineq_ub'])
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.extend(uub)
                con_ineq_lb.append(np.full((Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((Nu,), ca.inf))
                con_ineq_lb.append(ulb)
           
            # Add extra constraints
            if inequality_constraints is not None:
                cons = inequality_constraints(var['mean', t],
                                              var['covariance', t],
                                              u_t, var['eps', t], con_par)
                con_ineq.extend(cons['con_ineq'])
                con_ineq_lb.extend(cons['con_ineq_lb'])
                con_ineq_ub.extend(cons['con_ineq_ub'])

            # Objective function
            u_delta = u_t - u_past
            obj += self.__l_func(var['mean', t], covar_x_t, u_t, cov_u, u_delta) + lam * var['eps', t]
            u_t = u_past
        obj += self.__lf_func(var['mean', Nt], var['covariance', Nt].reshape((Ny, Ny)))

        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        if terminal_constraint is not None:
            con_ineq.append(var['mean', Nt] - mean_ref_s)
            num_ineq_con += 1
            con_ineq_lb.append(np.full((Ny,), - terminal_constraint))
            con_ineq_ub.append(np.full((Ny,), terminal_constraint))

        con = ca.vertcat(*con_eq, *con_ineq)
        self.__conlb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.__conub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build solver object
        nlp = dict(x=var, f=obj, g=con, p=param_s)
        options = {
            'ipopt.print_level' : 0,
            'ipopt.mu_init' : 0.01,
            'ipopt.tol' : 1e-8,
            'ipopt.warm_start_init_point' : 'yes',
            'ipopt.warm_start_bound_push' : 1e-9,
            'ipopt.warm_start_bound_frac' : 1e-9,
            'ipopt.warm_start_slack_bound_frac' : 1e-9,
            'ipopt.warm_start_slack_bound_push' : 1e-9,
            'ipopt.warm_start_mult_bound_push' : 1e-9,
#            'ipopt.mu_strategy' : 'adaptive',
            'print_time' : False,
            'verbose' : False,
            'expand' : True
        }
        if solver_opts is not None:
            options.update(solver_opts)
        self.__solver = ca.nlpsol('mpc_solver', 'ipopt', nlp, options)
        


        # First prediction used in the NLP, used in plot later
        self.__var_prediction = np.zeros((Nt + 1, Ny))
        self.__mean_prediction = np.zeros((Nt + 1, Ny))
        self.__mean = None

        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.__num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')


    def solve(self, x0, sim_time, x_sp=None, u0=None, debug=False, noise=False,
              con_par_func=None):
        """ Solve the optimal control problem

        # Arguments:
            sim_time: Simulation length
            x0: Initial state

        # Optional Arguments:
            x_sp: State set point, default is zero
            u0: Initial input
            noise: If True, add gaussian noise to the simulation
            con_par_func: Function to calculate the parameters to pass to the 
                          inequality function, inputs the current state.
                          
        # Returns:
            mean: Simulated output using the optimal control inputs
            u: Optimal control inputs
        """

        Nt = self.__Nt
        Ny = self.__Ny
        Nu = self.__Nu
        dt = self.__dt

        # Initial state
        if u0 is None:
            u0 = np.zeros(Nu)
        if x_sp is None:
            self.__x_sp = np.zeros(Ny)
        else:
            self.__x_sp = x_sp

        self.__Nsim = int(sim_time / dt)

        # Initialize variables
        self.__mean          = np.full((self.__Nsim + 1, Ny), x0)
        self.__mean_pred     = np.full((self.__Nsim + 1, Ny), x0)
        self.__covariance    = np.full((self.__Nsim + 1, Ny, Ny), np.eye(Ny)* 0)
        self.__u             = np.full((self.__Nsim, Nu), u0)

        self.__mean[0]       = x0
        self.__mean_pred[0]  = x0
        self.__covariance[0] = np.diag(self.__variance_0)
        self.__u[0]          = u0

        # Initial guess of the warm start variables
        #TODO: Add option to restart with previous state
        self.__var_init = self.__var(0)
        self.__var_init['covariance', 0] = self.__covariance[0].flatten()
        self.__lam_x0 = np.zeros(self.__num_var)
        self.__lam_g0 = 0
        
        # Linearize around operating point and calculate LQR gain matrix
        if self.__feedback:
            if self.__discrete_method is 'exact':
                Ad, Bd = self.__model.discrete_linearize(x0, u0)
            elif self.__discrete_method is 'rk4':
                Ad, Bd = self.__model.discrete_rk4_linearize(x0, u0)
            elif self.__discrete_method is 'gp':
                Ad, Bd = self.__gp.discrete_linearize(x0, u0, np.eye(self.__Nx)*1e-2)
            K, S, E = lqr(Ad, Bd, self.__Q, self.__R)
        else:
            K = np.zeros((Nu, Ny))
        self.__K = K

        print('\nSolving MPC with %d step horizon' % Nt)
        for t in range(self.__Nsim):
            solve_time = -time.time()

            """ Initial values """
            self.__var_init['mean', 0]  = self.__mean[t]
            
            # Get constraint parameters
            if con_par_func is not None:
                con_par = con_par_func(self.__mean[t, :])
            else:
                con_par = []
                if self.__num_con_par > 0:
                    raise TypeError(('Number of constraint parameters ({x}) is greater than zero, '
                                    'but no parameter function is provided.'
                                        ).format(x=self.__num_con_par))

            param  = ca.vertcat(self.__mean[t, :], self.__x_sp,
                                self.__covariance[t, :].flatten(), u0, K.flatten(),
                                con_par)
            args = dict(x0=self.__var_init,
                        lbx=self.__varlb,
                        ubx=self.__varub,
                        lbg=self.__conlb,
                        ubg=self.__conub,
                        lam_x0=self.__lam_x0,
                        lam_g0=self.__lam_g0,
                        p=param)

            """ Solve nlp"""
            sol             = self.__solver(**args)
            status          = self.__solver.stats()['return_status']
            optvar          = self.__var(sol['x'])
            self.__var_init = optvar
            self.__lam_x0   = sol['lam_x']
            self.__lam_g0   = sol['lam_g']

            """ Print status """
            solve_time     += time.time()
            print("* t=%f: %s - %f sec" % (t * self.__dt, status, solve_time))

            if t == 0:
                 for i in range(Nt + 1):
                     cov = optvar['covariance', i, :].reshape((Ny, Ny))
                     self.__var_prediction[i, :] = np.array(ca.diag(cov)).flatten()
                     self.__mean_prediction[i, :] = np.array(optvar['mean', i]).flatten()

            v = optvar['v', 0, :]

            self.__u[t, :] = np.array(self.__u_func(self.__mean[t, :], v, K.flatten())).flatten()
            self.__mean_pred[t + 1] = np.array(optvar['mean', 1]).flatten()
            self.__covariance[t + 1] = np.array(optvar['covariance', 1].reshape((Ny, Ny)))

            if debug:
                self.__debug(t)

            """ Simulate the next step """
            try:
                self.__mean[t + 1] = self.__model.sim(self.__mean[t],
                                       self.__u[t].reshape((1, Nu)), noise=noise)
            except RuntimeError:
                print('----------------------------------------')
                print('** System unstable, simulator crashed **')
                print('----------------------------------------')
                return self.__mean, self.__u
        return self.__mean, self.__u

    
    def __set_cost_function(self, costFunc, mean_ref_s):
        """ Define stage cost and terminal cost 
        """
        # Create GP and cos function symbols
        mean_s = ca.MX.sym('mean', self.__Ny)
        covar_x_s = ca.MX.sym('covar_x', self.__Ny, self.__Ny)
        covar_u_s = ca.MX.sym('covar_u', self.__Nu, self.__Nu)
        u_s = ca.MX.sym('u', self.__Nu)
        delta_u_s = ca.MX.sym('delta_u', self.__Nu)
        Q = ca.MX(self.__Q)
        P = ca.MX(self.__P)
        R = ca.MX(self.__R)
        S = ca.MX(self.__S)

        if costFunc is 'quad':
            self.__l_func = ca.Function('l', [mean_s, covar_x_s, u_s, covar_u_s, delta_u_s],
                               [self.__cost_l(mean_s, mean_ref_s, covar_x_s, u_s,
                                covar_u_s, delta_u_s, Q, R, S)])
            self.__lf_func = ca.Function('lf', [mean_s, covar_x_s],
                                   [self.__cost_lf(mean_s, mean_ref_s, covar_x_s, P)])
        elif costFunc is 'sat':
            self.__l_func = ca.Function('l', [mean_s, covar_x_s, u_s, covar_u_s, delta_u_s],
                               [self.__cost_saturation_l(mean_s, mean_ref_s,
                                    covar_x_s, u_s, covar_u_s, delta_u_s, Q, R, S)])
            self.__lf_func = ca.Function('lf', [mean_s, covar_x_s],
                                   [self.__cost_saturation_lf(mean_s,
                                        mean_ref_s, covar_x_s,  P)])
        else:
             raise NameError('No cost function called: ' + costFunc)
        
    
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

        Z_x = ca.SX.eye(Nx) + 2 * covar_x_s @ P_s
        cost_x = ca.Function('cost_x', [x_s, P_s, covar_x_s],
                           [1 - ca.exp(-(x_s.T @ ca.solve(Z_x.T, P_s.T).T @ x_s))
                                   / ca.sqrt(ca.det(Z_x))])
        return cost_x(x - x_ref, P, covar_x)


    def __cost_saturation_l(self, x, x_ref, covar_x, u, covar_u, delta_u, Q, R, S):
        """ Stage Cost function: Expected Value of Saturating Cost
        """
        Nx = ca.MX.size1(Q)
        Nu = ca.MX.size1(R)

        # Create symbols
        Q_s = ca.SX.sym('Q', Nx, Nx)
        R_s = ca.SX.sym('Q', Nu, Nu)
        x_s = ca.SX.sym('x', Nx)
        u_s = ca.SX.sym('x', Nu)
        covar_x_s = ca.SX.sym('covar_z', Nx, Nx)
        covar_u_s = ca.SX.sym('covar_u', ca.MX.size(R))

        Z_x = ca.SX.eye(Nx) + 2 * covar_x_s @ Q_s
        Z_u = ca.SX.eye(Nu) + 2 * covar_u_s @ R_s

        cost_x = ca.Function('cost_x', [x_s, Q_s, covar_x_s],
                           [1 - ca.exp(-(x_s.T @ ca.solve(Z_x.T, Q_s.T).T @ x_s))
                                   / ca.sqrt(ca.det(Z_x))])
        cost_u = ca.Function('cost_u', [u_s, R_s, covar_u_s],
                           [1 - ca.exp(-(u_s.T @ ca.solve(Z_u.T, R_s.T).T @ u_s))
                                   / ca.sqrt(ca.det(Z_u))])

        return cost_x(x - x_ref, Q, covar_x)  + cost_u(u, R, covar_u)


    def __cost_l(self, x, x_ref, covar_x, u, covar_u, delta_u, Q, R, S, s=1):
        """ Stage cost function: Expected Value of Quadratic Cost
        """
        Q_s = ca.SX.sym('Q', ca.MX.size(Q))
        R_s = ca.SX.sym('R', ca.MX.size(R))
        x_s = ca.SX.sym('x', ca.MX.size(x))
        u_s = ca.SX.sym('u', ca.MX.size(u))
        covar_x_s = ca.SX.sym('covar_x', ca.MX.size(covar_x))
        covar_u_s = ca.SX.sym('covar_u', ca.MX.size(R))

        sqnorm_x = ca.Function('sqnorm_x', [x_s, Q_s],
                               [ca.mtimes(x_s.T, ca.mtimes(Q_s, x_s))])
        sqnorm_u = ca.Function('sqnorm_u', [u_s, R_s],
                               [ca.mtimes(u_s.T, ca.mtimes(R_s, u_s))])
        trace_u  = ca.Function('trace_u', [R_s, covar_u_s],
                               [s * ca.trace(ca.mtimes(R_s, covar_u_s))])
        trace_x  = ca.Function('trace_x', [Q_s, covar_x_s],
                               [s * ca.trace(ca.mtimes(Q_s, covar_x_s))])

        return sqnorm_x(x - x_ref, Q) + sqnorm_u(u, R) + sqnorm_u(delta_u, S) \
                + trace_x(Q, covar_x)  + trace_u(R, covar_u)
                


    def __constraint(self, mean, covar, H, quantile, ub, lb):
        """ Build up chance constraint """
        r = ca.SX.sym('r')
        mean_s = ca.SX.sym('mean', ca.MX.size(mean))
        covar_s = ca.SX.sym('cov', ca.MX.size(covar))
        S_s = ca.SX.sym('S', ca.MX.size(covar))
        H_s = ca.SX.sym('H', 1, ca.MX.size2(H))
        
        cholesky = ca.Function('cholesky', [covar_s], [ca.chol(covar_s).T])
        S = cholesky(covar)
        
#        con_func = ca.Function('con', [mean_s, covar_s, H_s, r],
#                               [H_s @ mean_s + r * ca.sqrt(H_s @ covar_s @ H_s.T)])
        con_func = ca.Function('con', [mean_s, S_s, H_s, r],
                               [H_s @ mean_s + r * ca.norm_2(H_s @ S_s)])
        
        con = []
        con_lb = []
        con_ub = []        
        for i in range(ca.MX.size1(mean)):
            con.append(con_func(mean, S, H[i, :], quantile[i]))
            con_ub.append(ub[i])
            con_lb.append(-np.inf)
            con.append(con_func(mean, S, H[i, :], -quantile[i]))
            con_ub.append(np.inf)
            con_lb.append(lb[i])
        cons = dict(con=con, con_lb=con_lb, con_ub=con_ub)
        return cons


    def __debug(self, t):
        print('_______________  Debug  ________________')
        print('* Mean_%d:' %t)
        print(self.__mean[t])
        print('* u_%d:' % t)
        print(self.__u[t])
        print('* covar_%d:' % t)
        print(self.__covariance[t, :])
        print('----------------------------------------')


    def plot(self, title=None,
             xnames=None, unames=None, time_unit = 's', numcols=2):
        if self.__mean is None:
            print('Please solve the MPC before plotting')
            return

        x = self.__mean
        u = self.__u
        dt = self.__dt
        Nu = self.__Nu
        Nt_sim, Nx = x.shape
        
        # First predictin horizon
        x_pred = self.__mean_prediction
        var_pred = self.__var_prediction
        
        # One step prediction
        var = np.zeros((Nt_sim, Nx))
        mean = self.__mean_pred
        for t in range(Nt_sim):
            var[t] = np.diag(self.__covariance[t])

        
        x_sp = self.__x_sp * np.ones((Nt_sim, Nx))

        if x_pred is not None:
            Nt_horizon = np.size(x_pred, 0)
            t_horizon = np.linspace(0.0, Nt_horizon * dt -dt, Nt_horizon)
        if xnames is None:
            xnames = ['State %d' % (i + 1) for i in range(Nx)]
        if unames is None:
            unames = ['Control %d' % (i + 1) for i in range(Nu)]

        t = np.linspace(0.0, Nt_sim * dt -dt, Nt_sim)
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
            ax.plot(t, x[:, i], 'k-', marker='.', linewidth=1.0, label='Simulation')
            ax.errorbar(t, mean[:, i], yerr=2 * np.sqrt(var[:, i]), marker='.',
                            linestyle='None', color='b', label='One step prediction')
            if x_sp is not None:
                ax.plot(t, x_sp[:, i], color='g', linestyle='--', label='Setpoint')
            if x_pred is not None:
                ax.errorbar(t_horizon, x_pred[:, i], yerr=2 * np.sqrt(var_pred[:, i]),
                            linestyle='None', marker='.', color='r', label='1st prediction horizon')
            plt.legend(loc='best')
            ax.set_ylabel(xnames[i])
            ax.set_xlabel('Time [' + time_unit + ']')

        if title is not None:
            fig_x.canvas.set_window_title(title)
        else:
            fig_x.canvas.set_window_title(('MPC Horizon: {x}, Feedback: {y}, '
                                           'Discretization: {z}'
                                           ).format( x=self.__Nt,
                                                     y=self.__feedback,
                                                     z=self.__discrete_method
                                           ))
        plt.show()
        return fig_x, fig_u


def lqr(A, B, Q, R):
    """Solve the infinite-horizon, discrete-time LQR controller
        x[k+1] = A x[k] + B u[k]
        u[k] = -K*x[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    # Arguments:
        A, B: Linear system matrices
        Q, R: State and input penalty matrices, both positive definite
    
    # Returns:
        K: LQR gain matrix
        S: Solution to the Riccati equation
        E: Eigenvalues of the closed loop system
    """

    S = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = np.array(scipy.linalg.inv(B.T @ S @ B + R) @ (B.T @ S @ A))
    eigenvalues, eigenvec = scipy.linalg.eig(A - B @ K)

    return K, S, eigenvalues


def plot_eig(A, discrete=True):
    eigenvalues, eigenvec = scipy.linalg.eig(A)
    fig,ax = plt.subplots()
    ax.axhline(y=0, color='k', linestyle='--')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.scatter(eigenvalues.real, eigenvalues.imag)
    if discrete:
        ax.add_artist(plt.Circle((0,0), 1, color='g', alpha=.1))
    plt.ylim([min(-1, min(eigenvalues.imag)), max(1, max(eigenvalues.imag))])
    plt.xlim([min(-1, min(eigenvalues.real)), max(1, max(eigenvalues.real))])
    plt.gca().set_aspect('equal', adjustable='box')

    fig.canvas.set_window_title('Eigenvalues of linearized system')
    return eigenvalues