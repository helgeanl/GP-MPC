# -*- coding: utf-8 -*-
"""
Gaussian Process Model
Copyright (c) 2018, Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .gp_functions import gp_exact_moment, gp_taylor_approx, gp
from .gp_functions import build_gp, build_TA_cov, get_mean_function
from .optimize import train_gp, train_gp_numpy
from .mpc_class import lqr

class GP:
    def __init__(self, X, Y, mean_func="zero", gp_method="TA",
                 optimizer_opts=None, hyper=None, normalize=True, multistart=1,
                 xlb=None, xub=None, ulb=None, uub=None, meta=None,
                 optimize_nummeric=True):
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
            self.optimize(X=X, Y=Y, opts=optimizer_opts, mean_func=mean_func,
                          xlb=xlb, xub=xub, ulb=ulb, uub=uub,
                          multistart=multistart, normalize=normalize,
                          optimize_nummeric=optimize_nummeric)
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
        self.__mean, self.__var, self.__covar, self.__mean_jac = \
                                build_gp(self.__invK, self.__X, self.__hyper,
                                         self.__alpha, self.__chol)
        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)

        self.set_method(gp_method)


    def optimize(self, X=None, Y=None, opts=None, mean_func='zero',
                 xlb=None, xub=None, ulb=None, uub=None,
                 multistart=1, normalize=True, warm_start=False,
                 optimize_nummeric=True):
        self.__mean_func = mean_func
        self.__normalize = normalize

        if normalize and X is not None:
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
            if normalize and X is not None:
                self.__X = self.standardize(X, self.__meanZ, self.__stdZ)
            else:
                self.__X = X.copy()
        else:
            X = self.__X.copy()
        if Y is not None:
            Y = np.array(Y).copy()
            if normalize and X is not None:
                self.__Y = self.standardize(Y, self.__meanY, self.__stdY)
            else:
                self.__Y = Y.copy()
        else:
            Y = self.__Y.copy()

        if warm_start:
            hyp_init = self.__hyper
            lam_x = self.__lam_x
        else:
            hyp_init = None
            lam_x = None
        if optimize_nummeric:
            opt = train_gp_numpy(self.__X, self.__Y, meanFunc=self.__mean_func,
                               optimizer_opts=opts, multistart=multistart,
                               hyper_init=hyp_init)
        else:
            opt = train_gp(self.__X, self.__Y, meanFunc=self.__mean_func,
                               optimizer_opts=opts, multistart=multistart,
                               hyper_init=hyp_init, lam_x0=lam_x)

        self.__hyper = opt['hyper']
        self.__lam_x = opt['lam_x']
        self.__invK  = opt['invK']
        self.__alpha = opt['alpha']
        self.__chol  = opt['chol']
        self.__hyper_length_scales   = self.__hyper[:, :self.__Nx]
        self.__hyper_signal_variance = self.__hyper[:, self.__Nx]**2
        self.__hyper_noise_variance  = self.__hyper[:, self.__Nx + 1]**2
        self.__hyper_mean            = self.__hyper[:, (self.__Nx + 1):]


    def validate(self, X_test, Y_test):
        """ Validate GP model with test data
        """

        Y_test = Y_test.copy()
        X_test = X_test.copy()
        if self.__normalize:
            Y_test = self.standardize(Y_test, self.__meanY, self.__stdY)
            X_test = self.standardize(X_test, self.__meanZ, self.__stdZ)

        N, Ny = Y_test.shape
        loss = 0
        NLP = 0

        for i in range(N):
            mean = self.__mean(X_test[i, :])
            var = self.__var(X_test[i, :]) + self.noise_variance()
            loss += (Y_test[i, :] - mean)**2
            NLP += 0.5*np.log(2*np.pi * (var)) + ((Y_test[i, :] - mean)**2)/(2*var)

        loss = loss / N
        SMSE = loss/ np.std(Y_test, 0)
        MNLP = NLP / N

        print('\n________________________________________')
        print('# Validation of GP model ')
        print('----------------------------------------')
        print('* Num training samples: ' + str(self.__N))
        print('* Num test samples: ' + str(N))
        print('----------------------------------------')
        print('* Mean squared error: ')
        for i in range(Ny):
            print('\t- State %d: %f' % (i + 1, loss[i]))
        print('----------------------------------------')
        print('* Standardized mean squared error:')
        for i in range(Ny):
            print('\t* State %d: %f' % (i + 1, SMSE[i]))
        print('----------------------------------------')
        print('* Mean Negative log Probability:')
        for i in range(Ny):
            print('\t* State %d: %f' % (i + 1, MNLP[i]))
        print('----------------------------------------\n')

        self.__SMSE = np.max(SMSE)

        return np.array(SMSE).flatten(), np.array(MNLP).flatten()


    def set_method(self, gp_method='TA'):
        """ Select wich GP function to use

        # Arguments:
            gp_method: Method for propagating uncertainty.
                        'ME': Mean Equivalence (normal GP),
                        'TA': 1st order Tayolor Approximation,
                        'EM': 1st and 2nd Expected Moments,
                        'old_ME': non-optimized ME function,
                        'old_TA': non-optimzed TA function. Use 2nd order
                                    TA, but don't take into account covariance
                                    between states.
        """

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
        """ Predict future state

        # Arguments:
            x: State vector (Nx x 1)
            u: Input vector (Nu x 1)
            cov: Covariance matrix of input z=[x, u] (Nx+nu x Nx+Nu)
        """
        if self.__normalize:
            x_s = self.standardize(x, self.__meanX, self.__stdX)
            u_s = self.standardize(u, self.__meanU, self.__stdU)
        else:
            x_s = x
            u_s = u
        mean, cov = self.__predict(x_s, u_s, cov)
        if self.__normalize:
            mean = self.inverse_mean(mean, self.__meanY, self.__stdY)
#            cov = self.inverse_covariance(cov, self.__stdY)
        return mean, cov


    def get_size(self):
        """ Get the size of the GP model

        # Returns:
                N:  Number of training data
                Ny: Number of outputs
                Nu: Number of control inputs
        """
        return self.__N, self.__Ny, self.__Nu


    def get_hyper_parameters(self):
        """ Get hyper-parameters

        # Return:
            hyper: Dictionary containing the hyper-parameters,
                    'length_scale', 'signal_var', 'noise_var', 'mean'
        """
        hyper = dict(
                    length_scale = self.__hyper_length_scales,
                    signal_var = self.__hyper_signal_variance,
                    noise_var = self.__hyper_noise_variance,
                    mean = self.__hyper_mean
                )
        return hyper


    def print_hyper_parameters(self):
        """ Print out all hyperparameters
        """
        print('\n________________________________________')
        print('# Hyper-parameters')
        print('----------------------------------------')
        print('* Num samples:', self.__N)
        print('* Ny:', self.__Ny)
        print('* Nu:', self.__Nu)
        print('* Normalization:', self.__normalize)
        for state in range(self.__Ny):
            print('----------------------------------------')
            print('* Lengthscale: ', state)
            for i in range(self.__Ny + self.__Nu):
                print(('-- l{a}: {l}').format(a=i,l=self.__hyper_length_scales[state, i]))
            print('* Signal variance: ', state)
            print('-- sf2:', self.__hyper_signal_variance[state])
            print('* Noise variance: ', state)
            print('-- sn2:', self.__hyper_noise_variance[state])
        print('----------------------------------------')

    def covSEard(self, X, Z, ell, sf2):
        """ GP Squared Exponential Kernel

        # Arguments:
            X: Input matrix or vector (n_x x D)
            Z: Input matrix or vector (n_z x D)
            ell: Lengthscale vector (D x 1)
            sf2: Signal variance (scalar)

        # Returns:
            k(X,Z): Covariance between X and Z.
        """
        dist = 0

        if X.ndim > 1:
            n1, D = X.shape
        else:
            D = X.shape[0]
            n1 = 1
            X.shape = (n1, D)

        if Z.ndim > 1:
            n2, D2 = Z.shape
        else:
            D2 = Z.shape[0]
            n2 = 1
            Z.shape = (n2, D2)

        if D != D2:
            raise ValueError('Input dimensions are not the same! D_x=' + str(D)
                            + ', D_z=' + str(D2))
        for i in range(D):
            x1 = X[:, i].reshape(n1, 1)
            x2 = Z[:, i].reshape(n2, 1)
            dist = (np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) -
                2 * np.dot(x1, x2.T)) / ell[i]**2 + dist
        return sf2 * np.exp(-.5 * dist)


    def covar(self, X_new):
        """ Compute covariance of input data

        # Arguments:
            X_new: Input matrix or vector of size (n x D), with n samples,
                    and D inputs.
        # Returns:
            covar: Covariance between the samples for all the inputs
                    (D x (n x n)).
        """

        if X_new.ndim > 1:
            n, D = X_new.shape
            covar = np.zeros((D,n,n))
        else:
            D = X_new.shape[0]
            n = 1
            X_new.shape = (n, D)
            covar = np.zeros((D,n))

        for output in range(self.__Ny):
            ell = self.__hyper_length_scales[output]
            sf2 = self.__hyper_signal_variance[output]
            L = self.__chol[output]
            ks = self.covSEard(self.__X, X_new, ell, sf2)
            kss = sf2
            v = np.linalg.solve(L, ks)
            covar[output] = kss - v.T @ v
        return covar


    def update_data(self, X_new, Y_new, N_new=None):
        """ Update training data with new observations

        Will update training data with N_new samples, updating the
        cholesky covariance matrix, alpha, inverse covariance, and re-build
        the GP functions with the updated matrices.

        # Arguments:
            X_new: Input matrix with (n x Nx) new observations.
            Y_new: Corresponding measurments (n x Ny) from input X_new.
            N_new: Number of new samples to pick, default to N_new=n.


        # NOTE: NOT working as intended...
        """

        X_new = np.array(X_new).copy()
        Y_new = np.array(Y_new).copy()
        n, D = X_new.shape
        if N_new is None:
            N_new = n

        if self.__normalize:
            Y_new = self.standardize(Y_new, self.__meanY, self.__stdY)
            X_new = self.standardize(X_new, self.__meanZ, self.__stdZ)

        print('\n________________________________________')
        print('# Updating training data with ' + str(N_new) + ' new samples')
        print('----------------------------------------')
        for k in range(N_new):
            """ Explore point with highest combined variance """
            n, D = X_new.shape

            covar = self.covar(X_new)
            covar = np.sum(covar, 0) # Sum covariance of all states
            var = np.diag(covar)

            max_var_index = np.argmin(var)
            x_new = X_new[max_var_index]
            y_new = Y_new[max_var_index]
            X_new = np.delete(X_new, max_var_index, 0)
            Y_new = np.delete(Y_new, max_var_index, 0)

            """ Update matrices """
            N, D = self.__X.shape
            hyper = self.__hyper
            invK  = np.zeros((self.__Ny, N + 1, N + 1))
            alpha = np.zeros((self.__Ny, N + 1))
            chol  = np.zeros((self.__Ny, N + 1, N + 1))
            chol[:, :N, :N] = self.__chol

            for output in range(self.__Ny):
                ell = self.__hyper_length_scales[output]
                sf2 = self.__hyper_signal_variance[output]
                sn2 = self.__hyper_noise_variance[output]
                K_new   = self.covSEard(self.__X, x_new, ell, sf2)
                k_new__ = self.covSEard(x_new, x_new, ell, sf2) + sn2
                L = self.__chol[output]
                l_new = np.linalg.solve(L, K_new)
                l_new__ = np.sqrt(k_new__ - np.linalg.norm(l_new))
                chol[output, N:, :N] = l_new.T
                chol[output, N, N] = l_new__
                invL = np.linalg.solve(chol[output], np.eye(N + 1))
                invK[output, :, :] = np.linalg.solve(chol[output].T, invL)

            self.__X = np.vstack([self.__X, x_new])
            self.__Y = np.vstack([self.__Y, y_new])
            self.__N = self.__X.shape[0]

            for output in range(self.__Ny):
                m = get_mean_function(ca.MX(hyper[output, :]),
                                      self.__X.T, func=self.__mean_func)
                mean = np.array(m(self.__X.T)).reshape((self.__N + 1,))
                alpha[output] = np.linalg.solve(chol[output].T,
                                     np.linalg.solve(chol[output],
                                     self.__Y[:, output] - mean))
            self.__alpha = alpha
            self.__chol = chol
            self.__invK = invK

            # Rebuild GP with the new data
            self.__mean, self.__var, self.__covar, self.__mean_jac = \
                                build_gp(self.__invK, self.__X, self.__hyper,
                                         self.__alpha, self.__chol)
            print('* Update ' + str(k) + ' with new data point ' +str(max_var_index))
        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)
        self.set_method(self.__gp_method)


    def update_data_all(self, X_new, Y_new):
        """ Update training data with all new observations

        Will update training data with all samples, updating the
        cholesky covariance matrix, alpha, inverse covariance, and re-build
        the GP functions with the updated matrices.

        # Arguments:
            X_new: Input matrix with (n x Nx) new observations.
            Y_new: Corresponding measurments (n x Ny) from input X_new.
        """

        X_new = np.array(X_new).copy()
        Y_new = np.array(Y_new).copy()
        n, D = X_new.shape

        N_new = n

        if self.__normalize:
            Y_new = self.standardize(Y_new, self.__meanY, self.__stdY)
            X_new = self.standardize(X_new, self.__meanZ, self.__stdZ)

        print('\n________________________________________')
        print('# Updating training data with ' + str(N_new) + ' new samples')
        print('----------------------------------------')

        """ Explore point with highest combined variance """
        n, D = X_new.shape

        """ Update matrices """
        self.__X = np.vstack([self.__X, X_new])
        self.__Y = np.vstack([self.__Y, Y_new])
        self.__N = self.__X.shape[0]

        N, D = self.__X.shape
        hyper = self.__hyper

        invK  = np.zeros((self.__Ny, N , N ))
        alpha = np.zeros((self.__Ny, N ))
        chol  = np.zeros((self.__Ny, N , N ))


        for output in range(self.__Ny):
            ell = self.__hyper_length_scales[output]
            sf2 = self.__hyper_signal_variance[output]
            sn2 = self.__hyper_noise_variance[output]
            K_new   = self.covSEard(self.__X, self.__X, ell, sf2)

            K = K_new + sn2 * np.eye(self.__N)
            K = (K + K.T) * 0.5   # Make sure matrix is symmentric
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definit, adding jitter!")
                K = K + np.eye(N) * 1e-8
                L = np.linalg.cholesky(K)
            invL = np.linalg.solve(L, np.eye(self.__N))
            invK[output, :, :] = np.linalg.solve(L.T, invL)
            chol[output] = L
            m = get_mean_function(ca.MX(hyper[output, :]), self.__X.T,
                                    func=self.__mean_func)
            mean = np.array(m(self.__X.T)).reshape((self.__N,))
            alpha[output] = np.linalg.solve(L.T,
                                    np.linalg.solve(L, self.__Y[:, output] - mean))

        self.__alpha = alpha
        self.__chol = chol
        self.__invK = invK

        # Rebuild GP with the new data
        self.__mean, self.__var, self.__covar, self.__mean_jac = \
                            build_gp(self.__invK, self.__X, self.__hyper,
                                     self.__alpha, self.__chol)

        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)
        self.set_method(self.__gp_method)


    def replace_data_all(self, X_new, Y_new):
        """ Replace training data with new observations

        Will replace training data with new samples, replacing the
        cholesky covariance matrix, alpha, inverse covariance, and re-build
        the GP functions with the updated matrices.

        # Arguments:
            X_new: Input matrix with (n x Nx) new observations.
            Y_new: Corresponding measurments (n x Ny) from input X_new.
        """

        X_new = np.array(X_new).copy()
        Y_new = np.array(Y_new).copy()
        n, D = X_new.shape

        N_new = n

        if self.__normalize:
            Y_new = self.standardize(Y_new, self.__meanY, self.__stdY)
            X_new = self.standardize(X_new, self.__meanZ, self.__stdZ)

        print('\n________________________________________')
        print('# Replacing training data with ' + str(N_new) + ' new samples')
        print('----------------------------------------')

        """ Update matrices """
        self.__X = X_new
        self.__Y = Y_new
        self.__N = self.__X.shape[0]

        N, D = self.__X.shape
        hyper = self.__hyper

        invK  = np.zeros((self.__Ny, N , N ))
        alpha = np.zeros((self.__Ny, N ))
        chol  = np.zeros((self.__Ny, N , N ))


        for output in range(self.__Ny):
            ell = self.__hyper_length_scales[output]
            sf2 = self.__hyper_signal_variance[output]
            sn2 = self.__hyper_noise_variance[output]
            K_new   = self.covSEard(self.__X, self.__X, ell, sf2)

            K = K_new + sn2 * np.eye(self.__N)
            K = (K + K.T) * 0.5   # Make sure matrix is symmentric
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                print("K matrix is not positive definit, adding jitter!")
                K = K + np.eye(N) * 1e-8
                L = np.linalg.cholesky(K)
            invL = np.linalg.solve(L, np.eye(self.__N))
            invK[output, :, :] = np.linalg.solve(L.T, invL)
            chol[output] = L
            m = get_mean_function(ca.MX(hyper[output, :]), self.__X.T,
                                    func=self.__mean_func)
            mean = np.array(m(self.__X.T)).reshape((self.__N,))
            alpha[output] = np.linalg.solve(L.T,
                                    np.linalg.solve(L, self.__Y[:, output] - mean))

        self.__alpha = alpha
        self.__chol = chol
        self.__invK = invK

        # Rebuild GP with the new data
        self.__mean, self.__var, self.__covar, self.__mean_jac = \
                            build_gp(self.__invK, self.__X, self.__hyper,
                                     self.__alpha, self.__chol)

        self.__TA_covar = build_TA_cov(self.__mean, self.__covar,
                                       self.__mean_jac, self.__Nx, self.__Ny)
        self.set_method(self.__gp_method)


    def standardize(self, Y, mean, std):
        return (Y - mean) / std

    def normalize(self, u, lb, ub):
        return (u - lb) / (ub - lb)

    def inverse_mean(self, x, mean, std):
        """ Inverse standardization of the mean
        """
        return (x * std) + mean

    def inverse_variance(self, variance):
        """ Inverse standardization of the variance
        """
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
        if self.__normalize:
            x0 = self.standardize(x0, self.__meanX, self.__stdX)
            u0 = self.standardize(u0, self.__meanU, self.__stdU)
        Ad = np.array(self.__discrete_jac_x(x0, u0, cov0))
        Bd = np.array(self.__discrete_jac_u(x0, u0, cov0))

        return Ad, Bd


    def jacobian(self, x0, u0, cov0):
        """ Jacobian of posterior mean
            J = dmu/dx
        # Arguments:
            x0: State vector
            u0: Input vector
            cov0: Covariance
        """
        return self.__discrete_jac_x(x0, u0, cov0)


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
        """ Store model data in a dictionary """

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


    def save_model(self, filename):
        """ Save model to a json file"""
        import json
        output_dict = self.__to_dict()
        with open(filename + ".json", "w") as outfile:
            json.dump(output_dict, outfile)


    @classmethod
    def load_model(cls, filename):
        """ Create a new model from file"""
        import json
        with open(filename + ".json") as json_data:
            input_dict = json.load(json_data)
        return cls(**input_dict)


    def predict_compare(self, x0, u, model, num_cols=2, xnames=None,
                        title=None, feedback=False, x_ref = None,
                        Q=None, R=None, methods=None):
        """ Predict and compare all GP methods
        """
        # Predict future
        Nx = self.__Nx
        Ny = self.__Ny

        dt = model.sampling_time()
        Nt = np.size(u, 0)
        sim_time = Nt * dt
        initVar = self.__hyper[:,Nx + 1]**2
        if methods is None:
            methods = ['EM', 'TA', 'ME']
        color = ['k', 'y', 'r']
        mean = np.zeros((len(methods), Nt + 1 , Ny))
        var = np.zeros((len(methods), Nt + 1, Ny))
        covar = np.eye(Nx) * 1e-6 # Initial covar input matrix
        if Q is None:
            Q = np.eye(Ny)
        if R is None:
            R= np.eye(Nx - Ny)

        if x_ref is None and feedback:
            x_ref = np.zeros((Ny))

        if feedback:
            A, B = self.discrete_linearize(x0, u[0], covar)
            K, S, E = lqr(A, B, Q, R)

        for i in range(len(methods)):
            self.set_method(methods[i])
            mean_t = x0
            covar[:Ny, :Ny] = ca.diag(initVar)
            mean[i, 0, :] = x0
            u_t = u[0]

            A, B = self.discrete_linearize(mean_t, u_t, covar)
            K, P, E = lqr(A, B, Q, R)

            for t in range(1, Nt + 1):
                if feedback:
                    u_t = K @ (mean_t - x_ref)
                else:
                    u_t = u[t-1, :]
                mean_t, covar_x = self.predict(mean_t, u_t, covar)
                mean[i, t, :] = np.array(mean_t).reshape((Ny,))
                var[i, t, :] = np.diag(covar_x)
                if self.__normalize:
                    var[i, t, :] = self.inverse_variance(var[i, t, :])

                if feedback:
                    covar_u = K @ covar_x @ K.T
                    cov_xu = covar_x @ K.T
                    covar[Ny:, Ny:] = covar_u
                    covar[Ny:, :Ny] = cov_xu.T
                    covar[:Ny, Ny:] = cov_xu
                covar[:Ny, :Ny] = covar_x

        #TODO: Fix feedback
        if feedback:
            A, B = model.discrete_linearize(x0, u[0])
            K, P, E = lqr(A, B, Q, R)
            y_sim = np.zeros((Nt + 1 , Ny))
            y_sim[0] = x0
            y_t = x0
            for t in range(1, Nt + 1):
                if 0: #feedback:
                    u_t = K @ (y_t - x_ref)
                else:
                    u_t = u[t-1, :]
                y_t = model.integrate(x0, u_t, []).flatten()
                y_sim[t] = y_t
        else:
            y_sim = model.sim(x0, u)
            y_sim = np.vstack([x0, y_sim])

        t = np.linspace(0.0, sim_time, Nt + 1)

        if np.any(var < 0):
            var = var.clip(min=0)

        num_rows = int(np.ceil(Ny / num_cols))
        if xnames is None:
            xnames = ['State %d' % (i + 1) for i in range(Ny)]
        if x_ref is not None:
            x_sp = x_ref * np.ones((Nt+1, Ny))

        fontP = FontProperties()
        fontP.set_size('small')
        fig = plt.figure(figsize=(9.0, 6.0))
        for i in range(Ny):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.plot(t, y_sim[:, i], 'b-', label='Simulation')
            if x_ref is not None:
                ax.plot(t, x_sp[:, i], color='g', linestyle='--', label='Setpoint')

            for k in range(len(methods)):
                mean_i = mean[k, :, i]
                sd_i = np.sqrt(var[k, :, i])
                ax.errorbar(t, mean_i, yerr=2 * sd_i, color = color[k],
                             label='GP ' + methods[k])
            ax.set_ylabel(xnames[i])
            ax.legend(prop=fontP, loc='best')
            ax.set_xlabel('Time [s]')
    #        ax.set_ylim([-20,20])
        if title is not None:
            fig.canvas.set_window_title(title)
        else:
            fig.canvas.set_window_title(('Training data: {x},  Mean Function: {y},  '
                                         'Normalize: {q}, Feedback: {f}'
                                         ).format(x=self.__N, y=self.__mean_func,
                                         q=self.__normalize, f=feedback))
        plt.tight_layout()
        plt.show()
