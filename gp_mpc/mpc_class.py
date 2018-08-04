# -*- coding: utf-8 -*-
"""
Model Predictive Control with Gaussian Process
Copyright (c) 2018, Helge-André Langåker
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
                 Q=None, P=None, R=None, S=None, lam=None, lam_state=None,
                 ulb=None, uub=None, xlb=None, xub=None, terminal_constraint=None,
                 feedback=True, percentile=None, gp_method='TA', costFunc='quad', solver_opts=None,
                 discrete_method='gp', inequality_constraints=None, num_con_par=0,
                 hybrid=None, Bd=None, Bf=None
                 ):

        """ Initialize and build the MPC solver

        # Arguments:
            horizon: Prediction horizon with control inputs
            model: System model

        # Optional Argumants:
            gp: GP model
            Q: State penalty matrix, default=diag(1,...,1)
            P: Termial penalty matrix, default=diag(1,...,1)
                if feedback is True, then P is the solution of the DARE,
                discarding this option.
            R: Input penalty matrix, default=diag(1,...,1)*0.01
            S: Input rate of change penalty matrix, default=diag(1,...,1)*0.1
            lam: Slack variable penalty for constraints, defalt=1000
            lam_state: Slack variable penalty for violation of upper/lower
                        state boundy, defalt=None
            ulb: Lower boundry input
            uub: Upper boundry input
            xlb: Lower boundry state
            xub: Upper boundry state
            terminal_constraint: Terminal condition on the state
                    * if None: No terminal constraint is used
                    * if zero: Terminal state is equal to zero
                    * if nonzero: Terminal state is bounded within +/- the constraint
                    * if not None and feedback is True, then the expected value of
                        the Lyapunov function E{x^TPx} < terminal_constraint
                        is used as a terminal constraint.
            feedback: If true, use an LQR feedback function u= Kx + v
            percentile: Measure how far from the contrain that is allowed,
                        P(X in constrained set) > percentile,
                        percentile= 1 - probability of violation,
                        default=0.95
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
            discrete_method: 'gp' -  Gaussian process model
                             'rk4' - Runga-Kutta 4 Integrator
                             'exact' - CVODES or IDEAS (for ODEs or DEAs)
                             'hybrid' - GP model for dynamic equations, and RK4
                                        for kinematic equations
                             'd_hybrid' - Same as above, without uncertainty
                             'f_hybrid' - GP estimating modelling errors, with
                                          RK4 computing the the actual model
            num_con_par: Number of parameters to pass to the inequality function
            inequality_constraints: Additional inequality constraints
                    Use a function with inputs (x, covar, u, eps) and
                    that returns a dictionary with inequality constraints and limits.
                        e.g. cons = dict(con_ineq=con_ineq_array,
                                         con_ineq_lb=con_ineq_lb_array,
                                         con_ineq_ub=con_ineq_ub_array
                                    )

        # NOTES:
            * Differentiation of Sundails integrators is not supported with SX graph,
                meaning that the solver option 'extend_graph' must be set to False
                to use MX graph instead when using the 'exact' discrete method.
            * At the moment the f_hybrid option is not finished implemented...
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
        self.__hybrid = hybrid
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
        self.__Bd = Bd
        self.__Bf = Bf

        if xub is None:
            xub = np.full((Ny), np.inf)
        if xlb is None:
            xlb = np.full((Ny), -np.inf)
        if uub is None:
            uub = np.full((Nu), np.inf)
        if ulb is None:
            ulb = np.full((Nu), -np.inf)

        """ Default percentile probability """
        if percentile is None:
            percentile = 0.95
        quantile_x = np.ones(Ny) * norm.ppf(percentile)
        quantile_u = np.ones(Nu) * norm.ppf(percentile)
        Hx = ca.MX.eye(Ny)
        Hu = ca.MX.eye(Nu)


        """ Create parameter symbols """
        mean_0_s       = ca.MX.sym('mean_0', Ny)
        mean_ref_s     = ca.MX.sym('mean_ref', Ny)
        u_0_s          = ca.MX.sym('u_0', Nu)
        covariance_0_s = ca.MX.sym('covariance_0', Ny * Ny)
        K_s            = ca.MX.sym('K', Nu * Ny)
        P_s            = ca.MX.sym('P', Ny * Ny)
        con_par        = ca.MX.sym('con_par', num_con_par)
        param_s        = ca.vertcat(mean_0_s, mean_ref_s, covariance_0_s,
                                    u_0_s, K_s, P_s, con_par)


        """ Select wich GP function to use """
        if discrete_method is 'gp':
            self.__gp.set_method(gp_method)
#TODO:Fix
        if solver_opts['expand'] is not False and discrete_method is 'exact':
            raise TypeError("Can't use exact discrete system with expanded graph")

        """ Initialize state variance with the GP noise variance """
        if gp is not None:
            #TODO: Cannot use gp variance with hybrid model
            self.__variance_0 = np.full((Ny), 1e-10) #gp.noise_variance()
        else:
            self.__variance_0 = np.full((Ny), 1e-10)


        """ Define which cost function to use """
        self.__set_cost_function(costFunc, mean_ref_s, P_s.reshape((Ny, Ny)))


        """ Feedback function """
        mean_s = ca.MX.sym('mean', Ny)
        v_s = ca.MX.sym('v', Nu)
        if feedback:
            u_func = ca.Function('u', [mean_s, mean_ref_s, v_s, K_s],
                                 [v_s + ca.mtimes(K_s.reshape((Nu, Ny)),
                                 mean_s-mean_ref_s)])
        else:
            u_func = ca.Function('u', [mean_s, mean_ref_s, v_s, K_s], [v_s])
        self.__u_func = u_func


        """ Create variables struct """
        var = ctools.struct_symMX([(
                ctools.entry('mean', shape=(Ny,), repeat=Nt + 1),
                ctools.entry('L', shape=(int((Ny**2 - Ny)/2 + Ny),), repeat=Nt + 1),
                ctools.entry('v', shape=(Nu,), repeat=Nt),
                ctools.entry('eps', shape=(3,), repeat=Nt + 1),
                ctools.entry('eps_state', shape=(Ny,), repeat=Nt + 1),
        )])
        num_slack = 3 #TODO: Make this a little more dynamic...
        num_state_slack = Ny
        self.__var = var
        self.__num_var = var.size

        # Decision variable boundries
        self.__varlb = var(-np.inf)
        self.__varub = var(np.inf)

        """ Adjust hard boundries """
        for t in range(Nt + 1):
            j = Ny
            k = 0
            for i in range(Ny):
                # Lower boundry of diagonal
                self.__varlb['L', t, k] = 0
                k += j
                j -= 1
            self.__varlb['eps', t] = 0
            self.__varlb['eps_state', t] = 0
            if xub is not None:
                self.__varub['mean', t] = xub
            if xlb is not None:
                self.__varlb['mean', t] = xlb
            if lam_state is None:
                self.__varub['eps_state'] = 0


        """ Input covariance matrix """
        if discrete_method is 'hybrid':
            N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
            Nz_gp = Ny_gp + Nu_gp
            covar_d_sx = ca.SX.sym('cov_d', Ny_gp, Ny_gp)
            K_sx = ca.SX.sym('K', Nu, Ny)
            covar_u_func = ca.Function('cov_u', [covar_d_sx, K_sx],
#                                       [K_sx @ covar_d_sx @ K_sx.T])
                                        [ca.SX(Nu, Nu)])
            covar_s = ca.SX(Nz_gp, Nz_gp)
            covar_s[:Ny_gp, :Ny_gp] = covar_d_sx
#            covar_s = ca.blockcat(covar_x_s, cov_xu, cov_xu.T, cov_u)
            covar_func = ca.Function('covar', [covar_d_sx], [covar_s])
        elif discrete_method is 'f_hybrid':
            #TODO: Fix this...
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                Nz_gp = Ny_gp + Nu_gp
    #            covar_x_s = ca.MX.sym('covar_x', Ny_gp, Ny_gp)
                covar_d_sx = ca.SX.sym('cov_d', Ny_gp, Ny_gp)
                K_sx = ca.SX.sym('K', Nu, Ny)
    #
                covar_u_func = ca.Function('cov_u', [covar_d_sx, K_sx],
    #                                       [K_sx @ covar_d_sx @ K_sx.T])
                                            [ca.SX(Nu, Nu)])
    #            cov_xu_func = ca.Function('cov_xu', [covar_x_sx, K_sx],
    #                                      [covar_x_sx @ K_sx.T])
    #            cov_xu = cov_xu_func(covar_x_s, K_s.reshape((Nu, Ny)))
    #            cov_u = covar_u_func(covar_x_s, K_s.reshape((Nu, Ny)))
                covar_s = ca.SX(Nz_gp, Nz_gp)
                covar_s[:Ny_gp, :Ny_gp] = covar_d_sx
    #            covar_s = ca.blockcat(covar_x_s, cov_xu, cov_xu.T, cov_u)
                covar_func = ca.Function('covar', [covar_d_sx], [covar_s])
        else:
            covar_x_s = ca.MX.sym('covar_x', Ny, Ny)
            covar_x_sx = ca.SX.sym('cov_x', Ny, Ny)
            K_sx = ca.SX.sym('K', Nu, Ny)
            covar_u_func = ca.Function('cov_u', [covar_x_sx, K_sx],
                                       [K_sx @ covar_x_sx @ K_sx.T])
            cov_xu_func = ca.Function('cov_xu', [covar_x_sx, K_sx],
                                      [covar_x_sx @ K_sx.T])
            cov_xu = cov_xu_func(covar_x_s, K_s.reshape((Nu, Ny)))
            cov_u = covar_u_func(covar_x_s, K_s.reshape((Nu, Ny)))
            covar_s = ca.blockcat(covar_x_s, cov_xu, cov_xu.T, cov_u)
            covar_func = ca.Function('covar', [covar_x_s], [covar_s])

        """ Hybrid output covariance matrix """
        if discrete_method is 'hybrid':
            N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
            covar_d_sx = ca.SX.sym('covar_d', Ny_gp, Ny_gp)
            covar_x_sx = ca.SX.sym('covar_x', Ny, Ny)
            u_s       = ca.SX.sym('u', Nu)

            cov_x_next_s = ca.SX(Ny, Ny)
            cov_x_next_s[:Ny_gp, :Ny_gp] = covar_d_sx
            #TODO: Missing kinematic states
            covar_x_next_func = ca.Function( 'cov',
                                #[mean_s, u_s, covar_d_sx, covar_x_sx],
                                [covar_d_sx],
                                [cov_x_next_s])

            """ f_hybrid output covariance matrix """
        elif discrete_method is 'f_hybrid':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
    #            Nz_gp = Ny_gp + Nu_gp
                covar_d_sx = ca.SX.sym('covar_d', Ny_gp, Ny_gp)
                covar_x_sx = ca.SX.sym('covar_x', Ny, Ny)
    #            L_x       = ca.SX.sym('L', ca.Sparsity.lower(Ny))
    #            L_d       = ca.SX.sym('L', ca.Sparsity.lower(3))
                mean_s    = ca.SX.sym('mean', Ny)
                u_s       = ca.SX.sym('u', Nu)

    #            A_f     = hybrid.rk4_jacobian_x(mean_s[Ny_gp:], mean_s[:Ny_gp])
    #            B_f     = hybrid.rk4_jacobian_u(mean_s[Ny_gp:], mean_s[:Ny_gp])
    #            C       = ca.horzcat(A_f, B_f)
    #            cov = ca.blocksplit(covar_x_s, Ny_gp, Ny_gp)
    #            cov[-1][-1] = covar_d_sx
    #            cov_i = ca.blockcat(cov)
    #            cov_f   =  C @ cov_i @ C.T
    #            cov[0][0] = cov_f

                cov_x_next_s = ca.SX(Ny, Ny)
                cov_x_next_s[:Ny_gp, :Ny_gp] = covar_d_sx
    #            cov_x_next_s[Ny_gp:, Ny_gp:] =
    #TODO: Pre-solve the GP jacobian using the initial condition in the solve iteration
    #            jac_mean  = ca.SX(Ny_gp, Ny)
    #            jac_mean = self.__gp.jacobian(mean_s[:Ny_gp], u_s, 0)
    #            A = ca.horzcat(jac_f, Bd)
    #            jac = Bf @ jac_f @ Bf.T + Bd @ jac_mean @ Bd.T

    #            temp = jac_mean @ covar_x_s
    #            temp = jac_mean @ L_s
    #            cov_i = ca.SX(Ny + 3, Ny + 3)
    #            cov_i[:Ny,:Ny] = covar_x_s
    #            cov_i[Ny:, Ny:] = covar_d_s
    #            cov_i[Ny:, :Ny] = temp
    #            cov_i[:Ny, Ny:] = temp.T
                #TODO: This is just a new TA implementation... CLEAN UP...
                covar_x_next_func = ca.Function( 'cov',
                                    [mean_s, u_s, covar_d_sx, covar_x_sx],
                                                #TODO: Clean up
    #                                            [A @ cov_i @ A.T])
    #                                            [Bd @ covar_d_s @ Bd.T + jac @ covar_x_s @ jac.T])
    #                                             [ca.blockcat(cov)])
                                    [cov_x_next_s])
                # Cholesky factorization of covariance function
    #            S_x_next_func = ca.Function( 'S_x', [mean_s, u_s, covar_d_s, covar_x_s],
    #                                            [Bd @ covar_d_s + jac @ covar_x_s])



        L_s = ca.SX.sym('L', ca.Sparsity.lower(Ny))
        L_to_cov_func = ca.Function('cov', [L_s], [L_s @ L_s.T])
        covar_x_sx = ca.SX.sym('cov_x', Ny, Ny)
        cholesky = ca.Function('cholesky', [covar_x_sx], [ca.chol(covar_x_sx).T])

        """ Set initial values """
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(var['mean', 0] - mean_0_s)
        L_0_s = ca.MX(ca.Sparsity.lower(Ny), var['L', 0])
        L_init = cholesky(covariance_0_s.reshape((Ny,Ny)))
        con_eq.append(L_0_s.nz[:]- L_init.nz[:])
        u_past = u_0_s


        """ Build constraints """
        for t in range(Nt):
            # Input to GP
            mean_t = var['mean', t]
            u_t = u_func(mean_t, mean_ref_s, var['v', t], K_s)
            L_x = ca.MX(ca.Sparsity.lower(Ny), var['L', t])
            covar_x_t = L_to_cov_func(L_x)

            if discrete_method is 'hybrid':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                covar_t = covar_func(covar_x_t[:Ny_gp, :Ny_gp])
            elif discrete_method is 'd_hybrid':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                covar_t = ca.MX(Ny_gp + Nu_gp, Ny_gp + Nu_gp)
            elif discrete_method is 'gp':
                covar_t = covar_func(covar_x_t)
            else:
                covar_t = ca.MX(Nx, Nx)


            """ Select the chosen integrator """
            if discrete_method is 'rk4':
                mean_next_pred = model.rk4(mean_t, u_t,[])
                covar_x_next_pred = ca.MX(Ny, Ny)
            elif discrete_method is 'exact':
                mean_next_pred = model.Integrator(x0=mean_t, p=u_t)['xf']
                covar_x_next_pred = ca.MX(Ny, Ny)
            elif discrete_method is 'd_hybrid':
                # Deterministic hybrid GP model
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                mean_d, covar_d = self.__gp.predict(mean_t[:Ny_gp], u_t, covar_t)
                mean_next_pred = ca.vertcat(mean_d, hybrid.rk4(mean_t[Ny_gp:],
                                            mean_t[:Ny_gp], []))
                covar_x_next_pred = ca.MX(Ny, Ny)
            elif discrete_method is 'hybrid':
                # Hybrid GP model
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                mean_d, covar_d = self.__gp.predict(mean_t[:Ny_gp], u_t, covar_t)
                mean_next_pred = ca.vertcat(mean_d, hybrid.rk4(mean_t[Ny_gp:],
                                                    mean_t[:Ny_gp], []))
                #covar_x_next_pred = covar_x_next_func(mean_t, u_t, covar_d,
                #                                        covar_x_t)
                covar_x_next_pred = covar_x_next_func(covar_d )
            elif discrete_method is 'f_hybrid':
                #TODO: Hybrid GP model estimating model error
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                mean_d, covar_d = self.__gp.predict(mean_t[:Ny_gp], u_t, covar_t)
                mean_next_pred = ca.vertcat(mean_d, hybrid.rk4(mean_t[Ny_gp:],
                                                    mean_t[:Ny_gp], []))
                covar_x_next_pred = covar_x_next_func(mean_t, u_t, covar_d,
                                                        covar_x_t)
            else: # Use GP as default
                mean_next_pred, covar_x_next_pred = self.__gp.predict(mean_t,
                                                                u_t, covar_t)


            """ Continuity constraints """
            mean_next = var['mean', t + 1]
            con_eq.append(mean_next_pred - mean_next )

            L_x_next = ca.MX(ca.Sparsity.lower(Ny), var['L', t + 1])
            covar_x_next = L_to_cov_func(L_x_next).reshape((Ny*Ny,1))
            L_x_next_pred = cholesky(covar_x_next_pred)
            con_eq.append(L_x_next_pred.nz[:] - L_x_next.nz[:])


            """ Chance state constraints """
            cons = self.__constraint(mean_next, L_x_next, Hx, quantile_x, xub,
                                    xlb, var['eps_state',t])
            con_ineq.extend(cons['con'])
            con_ineq_lb.extend(cons['con_lb'])
            con_ineq_ub.extend(cons['con_ub'])

            """ Input constraints """
#            cov_u = covar_u_func(covar_x_t, K_s.reshape((Nu, Ny)))
            cov_u = ca.MX(Nu, Nu)
#            cons = self.__constraint(u_t, cov_u, Hu, quantile_u, uub, ulb)
#            con_ineq.extend(cons['con'])
#            con_ineq_lb.extend(cons['con_lb'])
#            con_ineq_ub.extend(cons['con_ub'])
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.extend(uub)
                con_ineq_lb.append(np.full((Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((Nu,), ca.inf))
                con_ineq_lb.append(ulb)

            """ Add extra constraints """
            if inequality_constraints is not None:
                cons = inequality_constraints(var['mean', t + 1],
                                              covar_x_next,
                                              u_t, var['eps', t], con_par)
                con_ineq.extend(cons['con_ineq'])
                con_ineq_lb.extend(cons['con_ineq_lb'])
                con_ineq_ub.extend(cons['con_ineq_ub'])

            """ Objective function """
            u_delta = u_t - u_past
            obj += self.__l_func(var['mean', t], covar_x_t, u_t, cov_u, u_delta) \
                    + np.full((1, num_slack),lam) @ var['eps', t]
            if lam_state is not None:
                obj += np.full((1,num_state_slack),lam_state) @ var['eps_state', t]
            u_t = u_past
        L_x = ca.MX(ca.Sparsity.lower(Ny), var['L', Nt])
        covar_x_t = L_to_cov_func(L_x)
        obj += self.__lf_func(var['mean', Nt], covar_x_t, P_s.reshape((Ny, Ny))) \
            + np.full((1, num_slack),lam) @ var['eps', Nt]
        if lam_state is not None:
            obj += np.full((1,num_state_slack),lam_state) @ var['eps_state', Nt]


        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        """ Terminal contraint """
        if terminal_constraint is not None and not feedback:
            con_ineq.append(var['mean', Nt] - mean_ref_s)
            num_ineq_con += Ny
            con_ineq_lb.append(np.full((Ny,), - terminal_constraint))
            con_ineq_ub.append(np.full((Ny,), terminal_constraint))
        elif terminal_constraint is not None and feedback:
            con_ineq.append(self.__lf_func(var['mean', Nt],
                            covar_x_t, P_s.reshape((Ny, Ny))))
            num_ineq_con += 1
            con_ineq_lb.append(0)
            con_ineq_ub.append(terminal_constraint)
        con = ca.vertcat(*con_eq, *con_ineq)
        self.__conlb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.__conub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        """ Build solver object """
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
            'ipopt.mu_strategy' : 'adaptive',
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
            x0: Initial state vector.
            sim_time: Simulation length.

        # Optional Arguments:
            x_sp: State set point, default is zero.
            u0: Initial input vector.
            debug: If True, print debug information at each solve iteration.
            noise: If True, add gaussian noise to the simulation.
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
        self.__covariance    = np.full((self.__Nsim + 1, Ny, Ny), np.eye(Ny) * 1e-8)
        self.__u             = np.full((self.__Nsim, Nu), u0)

        self.__mean[0]       = x0
        self.__mean_pred[0]  = x0
        #TODO: cannot use variance_0 with a hybrid model
        self.__covariance[0] = np.eye(Ny)*1e-10 #np.diag(self.__variance_0)
        self.__u[0]          = u0

        # Initial guess of the warm start variables
        #TODO: Add option to restart with previous state
        self.__var_init = self.__var(0)

        #TOTO: Add initialization of cov cholesky
        cov0 = self.__covariance[0]
        self.__var_init['L', 0] = cov0[np.tril_indices(Ny)]
        self.__lam_x0 = np.zeros(self.__num_var)
        self.__lam_g0 = 0

        """ Linearize around operating point and calculate LQR gain matrix """
        if self.__feedback:
            if self.__discrete_method is 'exact':
                A, B = self.__model.discrete_linearize(x0, u0)
            elif self.__discrete_method is 'rk4':
                A, B = self.__model.discrete_rk4_linearize(x0, u0)
            elif self.__discrete_method is 'hybrid':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                A_f, B_f = self.__hybrid.discrete_rk4_linearize(x0[Ny_gp:], x0[:Ny_gp])
                A_gp, B_gp = self.__gp.discrete_linearize(x0[:Ny_gp],
                                                u0, np.eye(Ny_gp+Nu_gp)*1e-8)
                A = np.zeros((Ny, Ny))
                B = np.zeros((Ny, Nu))
                A[:Ny_gp, :Ny_gp] = A_gp
#                A[Ny_gp:, Ny_gp:] = A_f
#                A[Ny_gp:, :Ny_gp] = B_f
                B[:Ny_gp, :] = B_gp

            elif self.__discrete_method is 'd_hybrid':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                A_f, B_f = self.__hybrid.discrete_rk4_linearize(x0[Ny_gp:], x0[:Ny_gp])
                A_gp, B_gp = self.__gp.discrete_linearize(x0[:Ny_gp],
                                                u0, np.eye(Ny_gp+Nu_gp)*1e-8)
                A = np.zeros((Ny, Ny))
                B = np.zeros((Ny, Nu))
                A[:Ny_gp, :Ny_gp] = A_gp
#                A[Ny_gp:, Ny_gp:] = A_f
#                A[Ny_gp:, :Ny_gp] = B_f
                B[:Ny_gp, :] = B_gp

#                A = self.__Bf @ A_f @ self.__Bf.T + self.__Bd @ A_gp @ self.__Bd.T
#                B = self.__Bf @ B_f + self.__Bd @ B_gp

            elif self.__discrete_method is 'gp':
                N_gp, Ny_gp, Nu_gp = self.__gp.get_size()
                A, B = self.__gp.discrete_linearize(x0,
                                                u0, np.eye(Ny_gp+Nu_gp)*1e-8)

            K, P, E = lqr(A, B, self.__Q, self.__R)
        else:
            K = np.zeros((Nu, Ny))
            P = self.__P
        self.__K = K

        print('\nSolving MPC with %d step horizon' % Nt)
        for t in range(self.__Nsim):
            solve_time = -time.time()

            # Test if RK4 is stable for given initial state
            if self.__discrete_method is 'rk4':
                if not self.__model.check_rk4_stability(x0,u0):
                    print('-- WARNING: RK4 is not stable! --')

            """ Update Initial values with measurment"""
            self.__var_init['mean', 0]  = self.__mean[t]

            # Get constraint parameters
            if con_par_func is not None:
                con_par = con_par_func(self.__mean[t, :])
            else:
                con_par = []
                if self.__num_con_par > 0:
                    raise TypeError(('Number of constraint parameters ({x}) is '
                                     'greater than zero, but no parameter '
                                     'function is provided.'
                                        ).format(x=self.__num_con_par))

            param  = ca.vertcat(self.__mean[t, :], self.__x_sp,
                                cov0.flatten(), u0, K.flatten(),
                                P.flatten(), con_par)
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
                     Li = ca.DM(ca.Sparsity.lower(self.__Ny), optvar['L', i])
                     cov = Li @ Li.T
                     self.__var_prediction[i, :] = np.array(ca.diag(cov)).flatten()
                     self.__mean_prediction[i, :] = np.array(optvar['mean', i]).flatten()

            v = optvar['v', 0, :]

            self.__u[t, :] = np.array(self.__u_func(self.__mean[t, :], self.__x_sp,
                                v, K.flatten())).flatten()
            self.__mean_pred[t + 1] = np.array(optvar['mean', 1]).flatten()
            L = ca.DM(ca.Sparsity.lower(self.__Ny), optvar['L', 1])
            self.__covariance[t + 1] = L @ L.T

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

            """Initial values for next iteration"""
            x0 = self.__mean[t + 1]
            u0 = self.__u[t]
        return self.__mean, self.__u


    def __set_cost_function(self, costFunc, mean_ref_s, P_s):
        """ Define stage cost and terminal cost
        """

        mean_s = ca.MX.sym('mean', self.__Ny)
        covar_x_s = ca.MX.sym('covar_x', self.__Ny, self.__Ny)
        covar_u_s = ca.MX.sym('covar_u', self.__Nu, self.__Nu)
        u_s = ca.MX.sym('u', self.__Nu)
        delta_u_s = ca.MX.sym('delta_u', self.__Nu)
        Q = ca.MX(self.__Q)
        R = ca.MX(self.__R)
        S = ca.MX(self.__S)

        if costFunc is 'quad':
            self.__l_func = ca.Function('l', [mean_s, covar_x_s, u_s,
                                                covar_u_s, delta_u_s],
                               [self.__cost_l(mean_s, mean_ref_s, covar_x_s, u_s,
                                covar_u_s, delta_u_s, Q, R, S)])
            self.__lf_func = ca.Function('lf', [mean_s, covar_x_s, P_s],
                                   [self.__cost_lf(mean_s, mean_ref_s, covar_x_s, P_s)])
        elif costFunc is 'sat':
            self.__l_func = ca.Function('l', [mean_s, covar_x_s, u_s,
                                                covar_u_s, delta_u_s],
                               [self.__cost_saturation_l(mean_s, mean_ref_s,
                                    covar_x_s, u_s, covar_u_s, delta_u_s, Q, R, S)])
            self.__lf_func = ca.Function('lf', [mean_s, covar_x_s, P_s],
                                   [self.__cost_saturation_lf(mean_s,
                                        mean_ref_s, covar_x_s,  P_s)])
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


    def __constraint(self, mean, covar, H, quantile, ub, lb, eps):
        """ Build up chance constraint vectors
        """

        r = ca.SX.sym('r')
        mean_s = ca.SX.sym('mean', ca.MX.size(mean))
        S_s = ca.SX.sym('S', ca.MX.size(covar))
        H_s = ca.SX.sym('H', 1, ca.MX.size2(H))
        S = covar
        con_func = ca.Function('con', [mean_s, S_s, H_s, r],
                                [H_s @ mean_s + r * H_s @ ca.diag(S_s)])

        con = []
        con_lb = []
        con_ub = []
        for i in range(ca.MX.size1(mean)):
            con.append(con_func(mean, S, H[i, :], quantile[i]) - eps[i])
            con_ub.append(ub[i])
            con_lb.append(-np.inf)
            con.append(con_func(mean, S, H[i, :], -quantile[i]) + eps[i])
            con_ub.append(np.inf)
            con_lb.append(lb[i])
        cons = dict(con=con, con_lb=con_lb, con_ub=con_ub)
        return cons


    def __debug(self, t):
        """ Print debug messages during each solve iteration
        """

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
        """ Plot MPC

        # Optional Arguments:
            title: Text displayed in figure window, defaults to MPC setting.
            xnames: List with labels for the states, defaults to 'State i'.
            unames: List with labels for the inputs, default to 'Control i'.
            time_unit: Label for the time axis, default to seconds.
            numcols: Number of columns in the figure.

        # Return:
            fig_x: Figure with states
            fig_u: Figure with control inputs
        """

        if self.__mean is None:
            print('Please solve the MPC before plotting')
            return

        x = self.__mean
        u = self.__u
        dt = self.__dt
        Nu = self.__Nu
        Nt_sim, Nx = x.shape

        # First prediction horizon
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

        fig_u = plt.figure(figsize=(9.0, 6.0))
        for i in range(Nu):
            ax = fig_u.add_subplot(Nu, 1, i + 1)
            ax.step(t, u[:, i] , 'k', where='post')
            ax.set_ylabel(unames[i])
            ax.set_xlabel('Time [' + time_unit + ']')
        fig_u.canvas.set_window_title('Control inputs')
        plt.tight_layout()

        fig_x = plt.figure(figsize=(9, 6.0))
        for i in range(Nx):
            ax = fig_x.add_subplot(numrows, numcols, i + 1)
            ax.plot(t, x[:, i], 'k-', marker='.', linewidth=1.0, label='Simulation')
            ax.errorbar(t, mean[:, i], yerr=2 * np.sqrt(var[:, i]), marker='.',
                            linestyle='None', color='b', label='One step prediction')
            if x_sp is not None:
                ax.plot(t, x_sp[:, i], color='g', linestyle='--', label='Setpoint')
            if x_pred is not None:
                ax.errorbar(t_horizon, x_pred[:, i], yerr=2 * np.sqrt(var_pred[:, i]),
                            linestyle='None', marker='.', color='r',
                            label='1st prediction horizon')
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
        plt.tight_layout()
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
        P: Solution to the Riccati equation
        E: Eigenvalues of the closed loop system
    """

    P = np.array(scipy.linalg.solve_discrete_are(A, B, Q, R))
    K = -np.array(scipy.linalg.solve(R + B.T @ P @ B, B.T @ P @ A))
    eigenvalues, eigenvec = scipy.linalg.eig(A + B @ K)

    return K, P, eigenvalues


def plot_eig(A, discrete=True):
    """ Plot eigenvelues

    # Arguments:
        A: System matrix (N x N).

    # Optional Arguments:
        discrete: If true the unit circle is added to the plot.

    # Returns:
        eigenvalues: Eigenvelues of the matrix A.
    """
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
    plt.show()
    return eigenvalues
