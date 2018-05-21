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

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .gp_functions import gp_exact_moment, gp_taylor_approx, gp, build_gp, build_TA_cov
from .optimize import train_gp, validate
from .mpc_class import lqr

class GP:
    def __init__(self, X, Y, mean_func='zero', gp_method='TA',
                 optimizer_opts=None):
        """ Initialize and optimize GP model

        """

        self.__Ny = Y.shape[1]
        self.__Nx = X.shape[1]
        self.__N = X.shape[0]
        self.__Nu = self.__Nx - self.__Ny
        self.__X = X
        self.__Y = Y
        self.__mean_func = mean_func
        self.__gp_method = gp_method

        """ Optimize hyperparameters """
        opt = train_gp(self.__X, self.__Y, meanFunc=self.__mean_func,
                       optimizer_opts=optimizer_opts)
        self.__hyper = opt['hyper']
        self.__invK  = opt['invK']
        self.__alpha = opt['alpha']
        self.__chol  = opt['chol']
        self.__hyper_length_scales   = self.__hyper[:, :self.__Nx]
        self.__hyper_signal_variance = self.__hyper[:, self.__Nx]**2
        self.__hyper_noise_variance = self.__hyper[:, self.__Nx + 1]**2
        self.__hyper_mean           = self.__hyper[:, (self.__Nx + 1):]

        # Build GP
        self.__mean, self.__covar, self.__mean_jac = build_gp(self.__invK, self.__X,
                                                     self.__hyper, self.__alpha, self.__chol)
        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)

        self.set_method(gp_method)


    def gp_test(self, z):
        return self.__mean(z), self.__covar(z)


    def gp_TA_test(self, z, covar):
        return self.__mean(z), self.__TA_covar(z, covar)




    def validate(self, X_test, Y_test):
        """ Validate GP model with test data """
        SMSE = validate(X_test, Y_test, self.__X, self.__Y, self.__invK,
                 self.__hyper, self.__mean_func)
        self.__SMSE = np.max(SMSE)


    def set_method(self, gp_method='TA'):
        """ Select wich GP function to use """

        x = ca.MX.sym('x', self.__Ny)
        covar_s = ca.MX.sym('covar', self.__Nx, self.__Nx)
        u = ca.MX.sym('u', self.__Nu)
        self.__gp_method = gp_method

        if gp_method is 'ME':
            self.predict = ca.Function('gp_mean', [x, u, covar_s],
                                [self.__mean(ca.vertcat(x,u)),
                                 self.__covar(ca.vertcat(x,u))])
        elif gp_method is 'TA':
            self.predict = ca.Function('gp_taylor', [x, u, covar_s],
                                [self.__mean(ca.vertcat(x,u)),
                                 self.__TA_covar(ca.vertcat(x,u), covar_s)])
        elif gp_method is 'EM':
            self.predict = ca.Function('gp_exact_moment', [x, u, covar_s],
                                gp_exact_moment(self.__invK, ca.MX(self.__X),
                                        ca.MX(self.__Y), ca.MX(self.__hyper),
                                        ca.vertcat(x, u).T, covar_s))
        elif gp_method is 'old_ME':
            self.predict = ca.Function('gp_mean', [x, u, covar_s],
                                gp(self.__invK, ca.MX(self.__X), ca.MX(self.__Y),
                                   ca.MX(self.__hyper),
                                   ca.vertcat(x, u).T, meanFunc=self.__mean_func))
        elif gp_method is 'old_TA':
            self.predict = ca.Function('gp_taylor_approx', [x, u, covar_s],
                                gp_taylor_approx(self.__invK, ca.MX(self.__X),
                                        ca.MX(self.__Y), ca.MX(self.__hyper),
                                        ca.vertcat(x, u).T, covar_s,
                                        meanFunc=self.__mean_func, diag=True))
        else:
            raise NameError('No GP method called: ' + gp_method)

        self.__discrete_jac_x = ca.Function('jac_x', [x, u, covar_s],
                                      [ca.jacobian(self.predict(x,u, covar_s)[0], x)])
        self.__discrete_jac_u = ca.Function('jac_x', [x, u, covar_s],
                                      [ca.jacobian(self.predict(x,u,covar_s)[0], u)])


    def discrete_linearize(self, x0, u0, cov0):
        """ Linearize the GP around the operating point
            x[k+1] = Ax[k] + Bu[k]
        # Arguments:
            x0: State vector
            u0: Input vector
            cov0: Covariance
        """
        Ad = np.array(self.__discrete_jac_x(x0, u0, cov0))
        Bd = np.array(self.__discrete_jac_u(x0, u0, cov0))
        return Ad, Bd


    def noise_variance(self):
        """ Get the noise variance
        """
        return self.__hyper_noise_variance

#TODO: Fix this
    def sparse(self, M):
        """ Sparse Gaussian Process
            Use Fully Independent Training Conditional (FITC) to approximate
            the GP distribution and reduce computational complexity.

        # Arguments:
            M: Reduce the model size from N to M.
        """


    def predict_compare(self, x0, u, model, num_cols=2, xnames=None,
                        title=None, feedback=False):
        """ Predict and compare all GP methods
        """
        # Predict future
        Nx = self.__Nx
        Ny = self.__Ny

        dt = model.sampling_time()
        Nt = np.size(u, 0)
        sim_time = Nt * dt
        initVar = self.__hyper[:,Nx + 1]**2
        methods = ['EM', 'TA', 'ME']
        color = ['k', 'y', 'r']
        mean = np.zeros((len(methods), Nt + 1 , Ny))
        var = np.zeros((len(methods), Nt + 1, Ny))
        covar = np.eye(Nx) * 1e-5 # Initial covar input matrix
        Q = np.eye(Ny)
        R= np.eye(Nx - Ny)


        for i in range(len(methods)):
            self.set_method(methods[i])
            mean_t = x0
            covar[:Ny, :Ny] = ca.diag(initVar)
            mean[i, 0, :] = x0
            if feedback:
                A, B = self.discrete_linearize(x0, u[0], covar)
                K, S, E = lqr(A, B, Q, R)
            for t in range(1, Nt + 1):
                if feedback:
                    u_t = -K @ mean_t
                else:
                    u_t = u[t-1, :]
                mean_t, covar_x = self.predict(mean_t, u_t, covar)
                mean[i, t, :] = np.array(mean_t).reshape((Ny,))
                var[i, t, :] = np.diag(covar_x)
                covar[:Ny, :Ny] = covar_x


        t = np.linspace(0.0, sim_time, Nt + 1)
        Y_sim = model.sim(x0, u)
        Y_sim = np.vstack([x0, Y_sim])
        if np.any(var < 0):
            var = var.clip(min=0)

        num_rows = int(np.ceil(Ny / num_cols))
        if xnames is None:
            xnames = ['State %d' % (i + 1) for i in range(Ny)]

        fontP = FontProperties()
        fontP.set_size('small')
        fig = plt.figure()
        for i in range(Ny):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.plot(t, Y_sim[:, i], 'b-', label='Simulation')

            for k in range(len(methods)):
                mean_i = mean[k, :, i]
                sd_i = np.sqrt(var[k, :, i])
                ax.errorbar(t, mean_i, yerr=2 * sd_i, color = color[k],
                             label='GP ' + methods[k])
            ax.set_ylabel(xnames[i])
            ax.legend(prop=fontP, loc='best')
            ax.set_xlabel('Time')
        if title is not None:
            fig.canvas.set_window_title(title)
        #else:
            # fig.canvas.set_window_title(('Training data: {x},  Mean Function: {y},  '
            #                              'Max Standarized Mean Squared Error: {z:.3g}'
            #                             ).format(x=self.__N, y=self.__mean_func,
            #                                     z=self.__SMSE))
        plt.show()
