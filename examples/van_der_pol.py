# -*- coding: utf-8 -*-
"""
Example of predicting the Van der Pol equation with a Gaussian Process

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../") # Add gp_mpc pagkage to path

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr


def plot_van_der_pol():
    """ Plot comparison of GP prediction with exact simulation
        on a 2000 step prediction horizon
    """
    Nt = 2000
    x0 = np.array([2., .201])

    cov = np.zeros((2,2))
    x = np.zeros((Nt,2))
    x_sim = np.zeros((Nt,2))

    x[0] = x0
    x_sim[0] = x0

    gp.set_method('ME')         # Use Mean Equivalence as GP method
    for i in range(Nt-1):
        x_t, cov = gp.predict(x[i], [], cov)
        x[i + 1] = np.array(x_t).flatten()
        x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[])

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_sim[:,0], x_sim[:,1], 'k-', linewidth=1.0, label='Exact')
    ax.plot(x[:,0], x[:,1], 'b-', linewidth=1.0, label='GP')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.legend(loc='best')
    plt.show()


def ode(x, u, z, p):
    """ Van der Pol equation
    """
    mu = 2
    dxdt = [
            x[1],
            -x[0] + mu * (1 - x[0]**2) * x[1]
    ]
    return  ca.vertcat(*dxdt)


""" System Parameters """
dt = .01                    # Sampling time
Nx = 2                      # Number of states
Nu = 0                      # Number of inputs
R_n = np.eye(Nx) * 1e-6     # Covariance matrix of added noise

# Limits in the training data
ulb = []    # No inputs are used
uub = []    # No inputs are used
xlb = [-4., -6.]
xub = [4., 6.]

N = 40          # Number of training data
N_test = 100    # Number of test data

""" Create simulation model and generate training/test data"""
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N_test, uub, ulb, xub, xlb, noise=True)


""" Create GP model and optimize hyper-parameters"""
gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
        uub=uub, optimizer_opts=None)
gp.validate(X_test, Y_test)

""" Predict and plot the result """
plot_van_der_pol()
