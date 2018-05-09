# Generate simulation data for regression model
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")

import pyDOE
import pylab
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt


class Model:
    def __init__(self, x, u, ode, dt, R, opt=None):
        """ Initialize dynamic model
        
        # Arguments:
            x:   SX symbol vector with the states (Ny,1)
            u:   SX symbol vector with the inputs (Nu,1)
            ode: Casadi vector with the ODE, with u as a paramater
            dt:  Sample time
    
        # Arguments (optional):    
            R:   Noise covariance matrix (Ny, Ny)
            opt: Options dict to pass to the IDEAS integrator
        """
        dae = {'x': x, 'ode': ode, 'p':u}
        
        # Create a default noise covariance matrix
        if R is None:
            self.__R = np.eye(self.__Ny) * 1e-3 
            
        # Integrator options
        opts = {}
        opts['abstol'] = 1e-10  # abs. tolerance
        opts['reltol'] = 1e-10  # rel. tolerance
        opts['t0'] = 0
        opts['tf'] = dt
        if opt is not None:
            opts.update(opt)

        self.__dt
        self.__Ny = ca.SX.size1(x)
        self.__Nu = ca.SX.size1(u)
        self.__Nx = self.__Ny + self.__Nu
        self.__Sim = ca.integrator('Sim', 'idas', dae, opts)
        

    def sim(self, x0, u, T, noise=False, clip_negative=False):
        
        Nt = int(T / self.__dt)
        
        # Initial state of the system
        x = x0

        # Predefine matrix to collect noisy state outputs
        Y_sim = np.zeros((Nt, self.__Ny))
    
        for t in range(Nt):
            u_t = u[t, :]    # control input for simulation
    
            # simulate system
            res = self.__Sim(x0=x0, p=u_t)
            x = pylab.array(res['xf'])[:, 0]
    
            # Add normal white noise to state outputs
            if noise:
                Y_sim[t, :] = x + np.random.multivariate_normal(np.zeros((self.__Ny)), self.__R)
            else:
                Y_sim[t, :] = x

            # Limit values to above 1e-8 to avvoid to avvoid numerical errors
            if clip_negative:
                if np.any(Y_sim < 0):
                    print('Negative values!')
                    Y_sim = Y_sim.clip(min=1e-8)
        return Y_sim


    def generate_training_data(self, N, uub, ulb, xub, xlb, noise=True, R=None):
        """ Generate training data using latin hypercube design
        
        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)
        """
        # Predefine matrix to collect control inputs
        u_mat = np.zeros((N, self.__Nu))
        # Predefine matrix to collect state inputs
        X_mat = np.zeros((N, self.__Ny))
        # Predefine matrix to collect noisy state outputs
        Y_mat = np.zeros((N, self.__Ny))
    
        # Create a default noise covariance matrix
        if noise and R is None:
            R = np.eye(self.__Ny) * 1e-3 
    
        # Create control input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^ndstate
        u_matrix = pyDOE.lhs(self.__Nu, samples=N, criterion='maximin')
        
        # Scale control inputs to correct range
        for k in range(N):
            u_mat[k, :] = u_matrix[k, :] * (uub - ulb) + ulb
    
        # Create state input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^ndstate
        X_mat = pyDOE.lhs(self.__Nx, samples=N, criterion='maximin')
    
        # Scale state inputs to correct range
        for k in range(N):
            X_mat[k, :] = X_mat[k, :] * (xub - xlb) + xlb
    
        for un in range(N):
            u_t = u_mat[un, :]    # control input for simulation
            x_t = X_mat[un, :]    # state input for simulation
    
            # simulate system with x_s and u_s inputs for deltat time
            res = self.__Sim(x0=x_t, p=u_t)
            y = pylab.array(res['xf'])[:, 0]

            # Add normal white noise to state outputs
            Y_mat[un, :] = y + np.random.multivariate_normal(np.zeros((self.__Ny)), self.__R)
    
        # Concatenate inputs to obtain overall input to GP model
        X_mat = np.hstack([X_mat, u_mat])
        return X_mat, Y_mat
    

    def plot(self, x0, u, T, numcols=2):
        """ Simulate and plot model
        
        # Arguments: 
            x0: Initial state
            u: Matrix with inputs for all time steps (Nt, Nu)
            T: Time horizon for simulation
            numcols: Number of columns in the plot
        """
        y = self.sim(x0, u, T, noise=True)
        Nt = int(T / self.__dt)
        t = np.linspace(0.0, Nt, 100)
        numrows = int(np.ceil(self.__Ny / numcols))
        
        fig_x = plt.figure()
        for i in range(self.__Nx):
            ax = fig_x.add_subplot(numrows, numcols, i + 1)
            ax.plot(t, y[:, i], 'b-', marker='.', linewidth=1.0)
            ax.set_ylabel('x_' + str(i + 1))
            ax.set_xlabel('Time')
        plt.show()
