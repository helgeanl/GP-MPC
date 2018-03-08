# Generate simulation data for regression model

# path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\Software\coinhsl-win32-openblas-2014.01.10")
# from scipy.stats import norm as norms
import numpy as np
# import matplotlib.pyplot as plt
#import time
#import math
import matplotlib.pyplot as plt
import pylab
from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py27-v3.3.0")
import casadi as ca
import pyDOE

# Specifications
# Discrete Time Model
ndstate = 3                  # Number of states
nastate = 2                  # Number of algebraic equations
ninput  = 2                  # Number of inputs
deltat  = 0.1                 # Time of one control interval

training = False                         # True: generate training data
optimize = False                         # Optimize hyperperameters
sim = True                              # True: simualate and plot system
simTime = 50                         # Simulation time in seconds

# Regression data
npoints = 200                                # Number of data points generated
u_min   = np.array([0., 0.])                 # lower bound of control inputs
u_max   = np.array([1., 0.1])            # upper bound of control inputs
x_min   = np.array([0., 0., 0.])              # lower bound of expected minimum state
x_max   = np.array([1., 1., 1.])            # upper bound of expected minimum state
R       = np.diag([1e-3, 1e-3, 1e-3])   # noise covariance matrix


def integrate_system(ndstate, nastate, u, t0, tf, x0):

    # Model Parameters
    mu1_M = 0.4
    mu2_M = 0.5
    K = 0.05
    K_I = 0.02
    y1 = 0.2
    y2 = 0.15
    Sf = 2.0
    p = 0.5

    # Declare variables (use scalar graph)
    #u =     SX.sym("u")      # parameters

    # Differential states
    xd   = ca.SX.sym("xd", ndstate)  # vector of states [c1,c2,I]

    # Algebraic states
    xa = ca.SX.sym("xa", nastate)  # vector of algebraic states [mu1,mu2]

    # Initial conditions
    xDi = x0

    xAi = [mu1_M * (Sf - x0[0] / y1 - x0[1] / y2) / (K + (Sf - x0[0] / y1 - x0[1] / y2)),
           mu2_M * (Sf - x0[0] / y1 - x0[1] / y2) * K / ((K + (Sf - x0[0] / y1 - x0[1] / y2)) * (K_I + x0[2]))]

    ode = ca.vertcat(xa[0] * xd[0] - xd[0] * u[0],
                     xa[1] * xd[1] - xd[1] * u[0],
                     -p * xd[0] * xd[2] + u[1] - xd[2] * u[0])

    alg = ca.vertcat(mu1_M * (Sf - xd[0] / y1 - xd[1] / y2) / (K + (Sf - xd[0] / y1 - xd[1] / y2)) - xa[0],
                     mu2_M * (Sf - xd[0] / y1 - xd[1] / y2) * K / ((K + (Sf - xd[0] / y1 - xd[1] / y2)) * (K_I + xd[2])) - xa[1])

    dae = {'x': xd, 'z': xa, 'ode': ode, 'alg': alg}

    # Create a DAE system solver
    opts = {}
    opts['abstol'] = 1e-10  # abs. tolerance
    opts['reltol'] = 1e-10  # rel. tolerance
    opts['t0'] = t0
    opts['tf'] = tf
    sim = ca.integrator('sim', 'idas', dae, opts)
    res = sim(x0=xDi, z0=xAi)
    x_current = pylab.array(res['xf'])
    return x_current


# -----------------------------------------------------------------------------
# Generation of simulation data
# -----------------------------------------------------------------------------
if training is True:
    u_matrix  = np.zeros((npoints, ninput))   # Predefine matrix to collect control inputs
    X_matrix  = np.zeros((npoints, ndstate))  # Predefine matrix to collect state inputs
    Y_matrix  = np.zeros((npoints, ndstate))  # Predefine matrix to collect noisy state outputs
    j = 0

    # Create control input design using a latin hypecube
    u_matrix = pyDOE.lhs(ninput, samples=npoints, criterion='maximin')  # Latin hypercube design for unit cube [0,1]^ndstate
    for k in range(npoints):
        u_matrix[k, :] = u_matrix[k, :] * (u_max - u_min) + u_min  # Scale control inputs to correct range

    # Create state input design using a latin hypecube
    X_matrix = pyDOE.lhs(ndstate, samples=npoints, criterion='maximin')  # Latin hypercube design for unit cube [0,1]^ndstate
    for k in range(npoints):
        X_matrix[k, :] = X_matrix[k, :] * (x_max - x_min) + x_min  # Scale state inputs to correct range

    for un in range(npoints):

        t0i = 0.             # start time of integrator
        tfi = deltat         # end time of integrator

        u_s = u_matrix[un, :]  # control input for simulation
        x_s = X_matrix[un, :]  # state input for simulation

        x_output = integrate_system(ndstate, nastate, u_s, t0i, tfi, x_s)[:, 0]                  # simulate system with x_s and u_s inputs for deltat time
        Y_matrix[un, :] = x_output + np.random.multivariate_normal(np.zeros((ndstate)), R)   # save simulated state with normal white noise added

    X_matrix = np.hstack([X_matrix, u_matrix])  # Concatenate inputs to obtain overall input to GP model
    np.savetxt('../Data/' + 'X_matrix_reactor', X_matrix)          # Save input matrix  as text file
    np.savetxt('../Data/' + 'Y_matrix_reactor', Y_matrix)          # Save output matrix as text file


# -----------------------------------------------------------------------------
# Simulate and plot system
# -----------------------------------------------------------------------------
def simSystem(x0, u, simTime, deltat, noise=False):
    simPoints = int(simTime / deltat)
    # Predefine matrix to collect control inputs
    u_matrix = u  # np.zeros((simPoints, ninput))
    #u_matrix[:, 0] = 50
    #u_matrix[:, 1] = 50

    # Initial state of the system
    x = x0  # np.array([10, 20, 30, 40])

    # Predefine matrix to collect noisy state outputs
    Y_sim = np.zeros((simPoints, ndstate))

    for dt in range(simPoints):
        t0i = 0.                 # start time of integrator
        tfi = deltat             # end time of integrator
        u_s = u_matrix[dt, :]    # control input for simulation
        # x_s = X_matrix[un, :]    # state input for simulation

        # simulate system
        x = integrate_system(ndstate, nastate, u_s, t0i, tfi, x)[:, 0]
        # save simulated state with normal white noise added
        if noise:
            Y_sim[dt, :] = x + np.random.multivariate_normal(np.zeros((ndstate)), R)
        else:
            Y_sim[dt, :] = x

    return Y_sim


if sim is True:
    # Plot simulation
    simPoints = int(simTime / deltat)
    u_matrix = np.zeros((simPoints, 2))
    u_matrix[:, 0] = 0.3
    u_matrix[:, 1] = 0.002
    x0 = np.array([0.16, 0.06, 0.005])

    t = np.linspace(0.0, simTime, simPoints)
    Y_sim = simSystem(x0, u_matrix, simTime, deltat)
    plt.figure()
    plt.clf()
    labels = ['c_1', 'c_2', 'I']
    plt.suptitle('Bioreactor', fontsize=16)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        
        plt.plot(t, Y_sim[:, i], 'b-')
        plt.ylabel(labels[i])
        plt.xlabel('Time [hr]')
    plt.show()
