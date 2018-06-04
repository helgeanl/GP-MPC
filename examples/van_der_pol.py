# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:12:26 2018

@author: Helge-André Langåker
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
#path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.1-64bit")
path.append(r"./../")

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from gp_mpc import Model, GP, MPC, plot_eig, lqr

def plot_van_der_pol():
    Nt = 2000
#    x0 = np.array([-1, 1.5])
    x0 = np.array([2., .201])
#    x0 = np.array([1.9, .27])
    #x0 = np.array([-1., 1.5])
    cov = np.zeros((2,2))
    x = np.zeros((Nt,2))
    x_sim = np.zeros((Nt,2))
    #x_rk4 = np.zeros((Nt,2))

    x[0] = x0
    x_sim[0] = x0
    #model.check_rk4_stability(x0,[])
    #x_rk4[0] = x0
    gp.set_method('ME')
    for i in range(Nt-1):
        x_t, cov = gp.predict(x[i], [], cov) 
        x[i + 1] = np.array(x_t).flatten() #- gp.noise_variance()
        x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[]) #+ np.random.multivariate_normal(
                                   # np.zeros((Nx)), R_n)
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
    # Model Parameters (Raff, Tobias et al., 2006)
    mu = -2
    dxdt = [
            x[1],
            -x[0] + mu * (1 - x[0]**2) * x[1]
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
R_n = np.eye(Nx) * 1e-6

# Limits in the training data
ulb = []
uub = []
xlb = [-4., -6.]
xub = [4., 6.]

N = 40 # Number of training data
N_new = 100

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R_n, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N_new, uub, ulb, xub, xlb, noise=True)

# Create GP model
gp = GP(X, Y, mean_func=meanFunc, normalize=True, xlb=xlb, xub=xub, ulb=ulb, 
        uub=uub, optimizer_opts=solver_opts, multistart=1)
print(gp._GP__hyper)
#gp.save_model('gp_tank')
#gp = GP.load_model('gp_tank')
gp.validate(X_test, Y_test)
plot_van_der_pol()
gp.update_data(X_test, Y_test, int(50))
gp.validate(X_test, Y_test)
plot_van_der_pol()


