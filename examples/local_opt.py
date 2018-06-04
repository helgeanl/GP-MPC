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


def covSEard(x, z, ell, sf2):
    """ GP squared exponential kernel """
    #dist = 0
    #for i in range(len(x)):
    #    dist = dist + (x[i] - z[i])**2 / (ell[i]**2)
    dist = np.sum((x - z)**2 / ell**2)
    return sf2 * np.exp(-.5 * dist)

def calc_cov_matrix(X, ell, sf2):
    """ GP squared exponential kernel """
    dist = 0
    n, D = X.shape
    for i in range(D):
        x = X[:, i].reshape(n, 1)
        
        dist = (np.sum(x**2, 1).reshape(-1, 1) + np.sum(x**2, 1) -
                2 * np.dot(x, x.T)) / ell[i]**2 + dist
    return sf2 * np.exp(-.5 * dist)


def calc_NLL_numpy(ell, sf2, X, Y):
    """ Objective function """
    # Calculate NLL
    n, D = X.shape
#    ell = hyper[:D]
#    sf2 = hyper[D]**2
    lik = 1e-5 #hyper[D + 1]**2

#    K = calc_cov_matrix(X, ell, sf2)
    K   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = covSEard(X[i,:], X[j,:], ell, sf2)

    K = K + lik * np.eye(n)
    K = (K + K.T) * 0.5   # Make sure matrix is symmentric
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        print("K is not positive definit, adding jitter!")
        K = K + np.eye(3) * 1e-8
        L = np.linalg.cholesky(K)

    logK = 2 * np.sum(np.log(np.abs(np.diag(L))))

    invLy = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, invLy)
    NLL = -0.5 * np.dot(Y.T, alpha) - 0.5 * logK
    return NLL


def ode(x, u, z, p):
    # Model Parameters (Jordan, 2007)

    dxdt = [
            ca.sin(0.9*x)
    ]
    
    return  ca.vertcat(*dxdt)



solver_opts = {
                'ipopt.linear_solver' : 'ma27',
                'ipopt.max_cpu_time' : 10,
                'expand' : False,
}

meanFunc = 'zero'
dt = .1
Nx = 1
Nu = 0
R = np.eye(Nx) * 1e-6 

# Limits in the training data
ulb = []
uub = []
xlb = [-5.]
xub = [5.]

N = 10 # Number of training data

# Create simulation model
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=True)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
X_test, Y_test = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
gp = GP(X,Y)

#hlb        = -np.inf * np.ones(3)
#hub        = np.inf * np.ones(3)
#hlb[:Nx]    = 1e-2
#hub[:Nx]    = 1e2
#hlb[Nx]     = 1e-8
#hub[Nx]     = 1e2
#hlb[Nx + 1] = 10**-6
#hub[Nx + 1] = 10**-4
#
#ell = np.linspace(hlb[0], hub[0], 50)
#sf = np.linspace(hlb[1], hub[1], 50)**2
##sn = np.linspace(hlb[2], hub[2], 50)
#X_ = gp._GP__X
#Y_ = gp._GP__Y
#nll = []
#for e in ell:
#    for s in sf:
#        nll.append(calc_NLL_numpy(e, s, X_, Y_))
##hyp = np.vstack([ell, sf])
##ELL, SF= np.meshgrid(ell,sf)
#
#plt.figure()
#nll=np.array(nll).reshape(50,50)
#plt.contourf(ell, sf, nll.T)
#plt.colorbar()


#x_rk4 = np.zeros((Nt,2))

x[0] = x0
cov
#x_sim[0] = x0
#x_rk4[0] = x0
for i in range(Nt-1):
    x_t, c = gp.predict(x[i], [], cov0)
    x[i + 1] = np.array(x_t).flatten()
#    x_sim[i+1] = model.integrate(x0=x_sim[i], u=[], p=[])
#    x_rk4[i+1] = np.array(model.rk4(x_rk4[i], [],[])).flatten()

#plt.figure()
#ax = plt.subplot(111)
#ax.plot(x_sim[:,0], x_sim[:,1], 'k-', linewidth=1.0, label='Exact')
#ax.plot(x[:,0], x[:,1], 'b-', linewidth=1.0, label='GP')
##ax.plot(x_rk4[:,0], x_rk4[:,1], 'g--', linewidth=1.0, label='RK4')
#ax.set_ylabel('y')
#ax.set_xlabel('x')
#plt.legend(loc='best')
#plt.show()
