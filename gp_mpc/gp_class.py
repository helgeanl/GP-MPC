# -*- coding: utf-8 -*-
"""
Gaussian Process Model
Copyright (c) 2018, Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .gp_functions import gp_exact_moment, gp_taylor_approx, gp, build_gp, build_TA_cov
from .optimize import train_gp, validate
from .mpc_class import lqr

class GP:
    def __init__(self, X, Y, mean_func="zero", gp_method="TA",
                 optimizer_opts=None, hyper=None, normalize=True,
                 xlb=None, xub=None, ulb=None, uub=None, meta=None):
        """ Initialize and optimize GP model

        """

        X = np.array(X).copy()
        Y = np.array(Y).copy()
        self.__X = X
        self.__Y = Y
        self.__Ny = Y.shape[1]
        self.__Nx = X.shape[1]
        self.__N = X.shape[0]
        self.__Nu = self.__Nx - self.__Ny

        self.__gp_method = gp_method
        self.__mean_func = mean_func
        self.__normalize = normalize

        if meta is not None:
            self.__meanY = np.array(meta['meanY'])
            self.__stdY  = np.array(meta['stdY'])
            self.__meanZ= np.array(meta['meanZ'])
            self.__stdZ = np.array(meta['stdZ'])
            self.__meanX = np.array(meta['meanX'])
            self.__stdX = np.array(meta['stdX'])
            self.__meanU = np.array(meta['meanU'])
            self.__stdU  = np.array(meta['stdU'])

        """ Optimize hyperparameters """
        if hyper is None:
            self.optimize(X=X, Y=Y, opts=optimizer_opts, mean_func=mean_func, normalize=normalize,
                          xlb=xlb, xub=xub, ulb=ulb, uub=uub)
        else:
            self.__hyper  = np.array(hyper['hyper'])
            self.__invK   = np.array(hyper['invK'])
            self.__alpha  = np.array(hyper['alpha'])
            self.__chol   = np.array(hyper['chol'])
            self.__hyper_length_scales   = np.array(hyper['length_scale'])
            self.__hyper_signal_variance = np.array(hyper['signal_var'])
            self.__hyper_noise_variance  = np.array(hyper['noise_var'])
            self.__hyper_mean = np.array(hyper['mean'])

        # Build GP
        self.__mean, self.__covar, self.__mean_jac = build_gp(self.__invK, self.__X,
                                                     self.__hyper, self.__alpha, self.__chol)
        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)

        self.set_method(gp_method)


    def optimize(self, X=None, Y=None, opts=None, mean_func='zero', normalize=True,
                 xlb=None, xub=None, ulb=None, uub=None):
        self.__mean_func = mean_func
        self.__normalize = normalize

        if normalize:
            self.__xlb = np.array(xlb)
            self.__xub = np.array(xub)
            self.__ulb = np.array(ulb)
            self.__uub = np.array(uub)
            self.__lb = np.hstack([xlb, ulb])
            self.__ub = np.hstack([xub, uub])
            self.__meanY = np.mean(Y, 0)
            self.__stdY  = np.std(Y, 0)
            self.__meanZ = np.mean(X, 0)
            self.__stdZ  = np.std(X, 0)
            self.__meanX = np.mean(X[:, :self.__Ny], 0)
            self.__stdX  = np.std(X[:, :self.__Ny], 0)
            self.__meanU = np.mean(X[:, self.__Ny:], 0)
            self.__stdU  = np.std(X[:, self.__Ny:], 0)


        if X is not None:
            X = np.array(X).copy()
            self.__X = X.copy()
        else:
            X = self.__X.copy()
        if Y is not None:
            Y = np.array(Y).copy()
            self.__Y = Y.copy()
        else:
            Y = self.__Y.copy()

        if normalize:
            self.__Y = self.standardize(Y, self.__meanY, self.__stdY)
#            self.__X = self.normalize(self.__X, self.__lb, self.__ub)
            self.__X = self.standardize(X, self.__meanZ, self.__stdZ)

        opt = train_gp(self.__X, self.__Y, meanFunc=self.__mean_func,
                           optimizer_opts=opts)
        self.__hyper = opt['hyper']
        self.__invK  = opt['invK']
        self.__alpha = opt['alpha']
        self.__chol  = opt['chol']
        self.__hyper_length_scales   = self.__hyper[:, :self.__Nx]
        self.__hyper_signal_variance = self.__hyper[:, self.__Nx]**2
        self.__hyper_noise_variance  = self.__hyper[:, self.__Nx + 1]**2
        self.__hyper_mean            = self.__hyper[:, (self.__Nx + 1):]


    def validate(self, X_test, Y_test):
        """ Validate GP model with test data """
        Y_test = Y_test.copy()
        X_test = X_test.copy()
        if self.__normalize:
            Y_test = self.standardize(Y_test, self.__meanY, self.__stdY)
#            X_test = self.normalize(X_test, self.__lb, self.__ub)
            X_test = self.standardize(X_test, self.__meanZ, self.__stdZ)
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
            self.__predict = ca.Function('gp_mean', [x, u, covar_s],
                                [self.__mean(ca.vertcat(x,u)),
                                 self.__covar(ca.vertcat(x,u))])
        elif gp_method is 'TA':
            self.__predict = ca.Function('gp_taylor', [x, u, covar_s],
                                [self.__mean(ca.vertcat(x,u)),
                                 self.__TA_covar(ca.vertcat(x,u), covar_s)])
        elif gp_method is 'EM':
            self.__predict = ca.Function('gp_exact_moment', [x, u, covar_s],
                                gp_exact_moment(self.__invK, ca.MX(self.__X),
                                        ca.MX(self.__Y), ca.MX(self.__hyper),
                                        ca.vertcat(x, u).T, covar_s))
        elif gp_method is 'old_ME':
            self.__predict = ca.Function('gp_mean', [x, u, covar_s],
                                gp(self.__invK, ca.MX(self.__X), ca.MX(self.__Y),
                                   ca.MX(self.__hyper),
                                   ca.vertcat(x, u).T, meanFunc=self.__mean_func))
        elif gp_method is 'old_TA':
            self.__predict = ca.Function('gp_taylor_approx', [x, u, covar_s],
                                gp_taylor_approx(self.__invK, ca.MX(self.__X),
                                        ca.MX(self.__Y), ca.MX(self.__hyper),
                                        ca.vertcat(x, u).T, covar_s,
                                        meanFunc=self.__mean_func, diag=True))
        else:
            raise NameError('No GP method called: ' + gp_method)

        self.__discrete_jac_x = ca.Function('jac_x', [x, u, covar_s],
                                      [ca.jacobian(self.__predict(x,u, covar_s)[0], x)])
        self.__discrete_jac_u = ca.Function('jac_x', [x, u, covar_s],
                                      [ca.jacobian(self.__predict(x,u,covar_s)[0], u)])


    def predict(self, x, u, cov):
        if self.__normalize:
            x_s = self.standardize(x, self.__meanX, self.__stdX)
            u_s = self.standardize(u, self.__meanU, self.__stdU)
#            x_s = self.normalize(x, self.__xlb, self.__xub)
#            u_s = self.normalize(u, self.__ulb, self.__uub)
        else:
            x_s = x
            u_s = u
        mean, cov = self.__predict(x_s, u_s, cov)
        if self.__normalize:
            mean = self.inverse_mean(mean, self.__meanY, self.__stdY)
#            cov = self.inverse_covariance(cov, self.__stdY)
        return mean, cov


    def standardize(self, Y, mean, std):
        return (Y - mean) / std

    def normalize(self, u, lb, ub):
        return (u - lb) / (ub - lb)

    def inverse_mean(self, x, mean, std):
        return (x * std) + mean

    def inverse_variance(self, variance):
#        return (covariance[..., np.newaxis] * self.__stdY**2)
        return variance * self.__stdY**2


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

    def __to_dict(self):
        """ Store model data in a struct """
        gp_dict = {}
        gp_dict['X'] = self.__X.tolist()
        gp_dict['Y'] = self.__Y.tolist()
        gp_dict['hyper'] = dict(
                    hyper = self.__hyper.tolist(),
                    invK = self.__invK.tolist(),
                    alpha = self.__alpha.tolist(),
                    chol = self.__chol.tolist(),
                    length_scale = self.__hyper_length_scales.tolist(),
                    signal_var = self.__hyper_signal_variance.tolist(),
                    noise_var = self.__hyper_noise_variance.tolist(),
                    mean = self.__hyper_mean.tolist()
                )
        gp_dict['mean_func'] = self.__mean_func
        gp_dict['normalize'] = self.__normalize
        if self.__normalize:
            gp_dict['xlb'] = self.__xlb.tolist()
            gp_dict['xub'] = self.__xub.tolist()
            gp_dict['ulb'] = self.__ulb.tolist()
            gp_dict['uub'] = self.__uub.tolist()
            gp_dict['meta'] = dict(
                        meanY = self.__meanY.tolist(),
                        stdY = self.__stdY.tolist(),
                        meanZ = self.__meanZ.tolist(),
                        stdZ = self.__stdZ.tolist(),
                        meanX = self.__meanX.tolist(),
                        stdX = self.__stdX.tolist(),
                        meanU = self.__meanU.tolist(),
                        stdU = self.__stdU.tolist()
                )
        return gp_dict


    def save_model(self, output_filename):
        """ Save model to file"""
        import json
        output_dict = self.__to_dict()
        with open(output_filename + ".json", "w") as outfile:
            json.dump(output_dict, outfile)


    @classmethod
    def load_model(cls, filename):
        """ Create a new model from file"""
        import json
        with open(filename + ".json") as json_data:
            input_dict = json.load(json_data)
        return cls(**input_dict)


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
        else:
            fig.canvas.set_window_title(('Training data: {x},  Mean Function: {y},  '
                                         'Normalize: {q}, '
                                         ).format(x=self.__N, y=self.__mean_func, q=self.__normalize))
        plt.show()
