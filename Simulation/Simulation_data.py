# Generate simulation data for regression model
from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
#path.append(r"C:\Users\helganl\Documents\coinhsl-win32-openblas-2014.01.10")
from scipy.stats import norm as norms
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from casadi import *
from pyDOE import *
from scipy.stats.distributions import lognorm

# Specifications
## Discrete Time Model
ndstate = 5                  # Number of states
nastate = 2                  # Number of algebraic equations
ninput  = 2                  # Number of inputs
deltat  = 0.1                # Time of one control interval

## Regression data
npoints = 10                              # Number of data points generated
u_min   = np.array([0.,200.])             # lower bound of control inputs
u_max   = np.array([250.,500.])           # upper bound of control inputs
x_min   = np.array([0.,0.,0.,290.,100.])  # lower bound of expected minimum state
x_max   = np.array([1.,1.,1.,350.,1000.]) # upper bound of expected minimum state
R       = np.diag([1e-3,1e-3,1e-3,4.,4.]) # noise covariance matrix

def integrate_system(ndstate,nastate,u,t0,tf,x0,xu_true):

    # Model Parameters
    N0H2S04 = 100.
    CpA = 30.
    CpB = 60.
    CpC = 20.
    CpH2SO4 = 35.
    T0 = 305.
    HRA = -6500.
    HRB = 8000.
    E1A = 9500./1.987
    E2A = 7000./1.987
    A1 = 1.25
    A2 = 0.08
    Tr1 = 420.
    Tr2 = 400.
    UA = 35000.

    # Declare variables (use scalar graph)
    p =     SX.sym("p",0,1)      # parameters

    # Differential states
    xd   =  SX.sym("xd",ndstate) # vector of states [x,s,p,V,theta]

    # Algebraic states
    xa =    SX.sym("xa",nastate) # vector of algebraic states [mu,q_p]

    # Initial conditions
    xDi = x0
    xAi = [(A1*exp(E1A*(1./Tr1-1./xDi[3]))),\
           (xu_true[1]*exp(E2A*(1./Tr2-1./xDi[3])))]

    # Construct the DAE system
    ode = vertcat(
            -xa[0]*xd[0] + (xu_true[0]-xd[0])*(u[0]/xd[4]) ,\
            xa[0]*xd[0]/2 - xa[1]*xd[1] - xd[1]*(u[0]/xd[4]) ,\
            3*xa[1]*xd[1] - xd[2]*(u[0]/xd[4]) , \
            (xu_true[2]*10.**4*(u[1]-xd[3]) - xu_true[0]*u[0]*CpA*(xd[3]-T0) + (HRA*(-xa[0]*xd[0])+HRB*(-xa[1]*xd[1]\
            ))*xd[4])/((xd[0]*CpA+CpB*xd[1]+CpC*xd[2])*xd[4] + xu_true[3]*CpH2SO4), \
            u[0])

    alg = vertcat((A1*exp(E1A*(1./Tr1-1./xd[3]))) - xa[0]  ,\
            (xu_true[1]*exp(E2A*(1./Tr2-1./xd[3]))) - xa[1])

    dae = {'x':xd, 'z':xa, 'ode':ode, 'alg':alg}

    # Create a DAE system solver
    I = integrator('I', 'idas', dae, {'t0':t0, 'tf':tf, 'abstol':1e-10, \
    'reltol':1e-10})
    res = I(x0=xDi,z0=xAi)
    x_current = array(res['xf'])
    return x_current

# -----------------------------------------------------------------------------
## Generation of simulation data
# -----------------------------------------------------------------------------
u_matrix  = np.zeros((npoints,ninput))  # Predefine matrix to collect control inputs
X_matrix  = np.zeros((npoints,ndstate)) # Predefine matrix to collect state inputs
Y_matrix  = np.zeros((npoints,ndstate)) # Predefine matrix to collect noisy state outputs
j = 0

# Create control input design using a latin hypecube
u_matrix = lhs(ninput, samples=npoints,criterion='maximin') # Latin hypercube design for unit cube [0,1]^ndstate
for k in range(npoints):
    u_matrix[k,:] = u_matrix[k,:]*(u_max-u_min)+u_min # Scale control inputs to correct range

# Create state input design using a latin hypecube
X_matrix = lhs(ndstate, samples=npoints,criterion='maximin') # Latin hypercube design for unit cube [0,1]^ndstate
for k in range(npoints):
    X_matrix[k,:] = X_matrix[k,:]*(x_max-x_min)+x_min # Scale state inputs to correct range

for un in range(npoints):

    t0i = 0.             # start time of integrator
    tfi = deltat         # end time of integrator

    u_s = u_matrix[un,:] # control input for simulation
    x_s = X_matrix[un,:] # state input for simulation

    x_output = integrate_system(ndstate,nastate,u_s,t0i,tfi,x_s)[:,0]                  # simulate system with x_s and u_s inputs for deltat time
    Y_matrix[un,:] = x_output + np.random.multivariate_normal(np.zeros((ndstate)),R)   # save simulated state with normal white noise added

X_matrix = np.hstack([X_matrix,u_matrix]) # Concatenate inputs to obtain overall input to GP model
np.savetxt('../Data/' + 'X_matrix', X_matrix)          # Save input matrix  as text file
np.savetxt('../Data/' + 'Y_matrix', Y_matrix)          # Save output matrix as text file
