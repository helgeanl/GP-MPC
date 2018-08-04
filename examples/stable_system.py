# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"./../")

import numpy as np
import casadi as ca
import pyDOE
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr


def plot_system():
    Nt = 1000
    x0 = np.array([-4., 1.])
    cov0 = np.zeros((2,2))
    x = np.zeros((Nt,2))
    x_sim = np.zeros((Nt,2))
    #x_rk4 = np.zeros((Nt,2))

    x[0] = x0
    x_sim[0] = x0
    #x_rk4[0] = x0
    for i in range(Nt-1):
        x_t, c = gp.predict(x[i], [], cov0)
        x[i + 1] = np.array(x_t).flatten()
        x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[])
    #    x_rk4[i+1] = np.array(model.rk4(x_rk4[i], [],[])).flatten()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_sim[:,0], x_sim[:,1], 'k-', linewidth=1.0, label='Exact')
    ax.plot(x[:,0], x[:,1], 'b-', linewidth=1.0, label='GP')
    #ax.plot(x_rk4[:,0], x_rk4[:,1], 'g--', linewidth=1.0, label='RK4')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.legend(loc='best')
    plt.show()




def ode(x, u, z, p):
    # Model Parameters (Jordan, 2007)
    lam = 5
    a11 = -lam
    a12 = 1.
    a21 = -lam
    a22 = -3.
    dxdt = [
            a11 * x[0] + a12 * x[1] ,
            a21 * x[0] + a22 * x[1]
    ]

    return  ca.vertcat(*dxdt)


solver_opts = {
                'ipopt.linear_solver' : 'ma27',
                'ipopt.max_cpu_time' : 10,
                'expand' : False,
}

meanFunc = 'zero'
dt = .01
Nx = 2
Nu = 0
R = np.eye(Nx) * 1e-6

# Limits in the training data
ulb = []
uub = []
xlb = [-5., -3.]
xub = [5., 3.]

N = 30 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)

# Create GP model
gp = GP(X, Y, mean_func=meanFunc, normalize=False, xlb=xlb, xub=xub, ulb=ulb,
        uub=uub, optimizer_opts=solver_opts, multistart=1)
print(gp._GP__hyper)
#gp.save_model('gp_tank')
#gp = GP.load_model('gp_tank')
gp.validate(X_test, Y_test)
#plot_system()
