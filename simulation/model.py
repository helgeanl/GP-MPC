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
    def __init__(self, Nx, Nu, ode, dt, R=None, alg=None, z=None, opt=None):
        """ Initialize dynamic model
        
        # Arguments:
            Nx:   Number of states
            Nu:   Number of inputs
            ode:  ode(x, u)
            dt:   Sampling time
    
        # Arguments (optional):    
            R:   Noise covariance matrix (Ny, Ny)
            alg: Casadi vector with the algabraic equations
            z:   SX symbol vector with algebraic variables
            opt: Options dict to pass to the IDEAS integrator
        """


        # Create a default noise covariance matrix
        if R is None:
            self.__R = np.eye(self.__Ny) * 1e-3 
        else:
            self.__R = R

        self.dt   = dt
        self.__Nu = Nu
        self.__Nx = Nx
        
        """ Create integrator """
        # Integrator options
        opts = {}
        opts['abstol'] = 1e-10  # abs. tolerance
        opts['reltol'] = 1e-10  # rel. tolerance
        opts['t0'] = 0
        opts['tf'] = dt
        if opt is not None:
            opts.update(opt)

        x = ca.SX.sym('x', Nx)
        u = ca.SX.sym('u', Nu)
        dae = {'x': x, 'ode': ode(x,u), 'p':u}
        self.__I = ca.integrator('I', 'idas', dae, opts)
        
        """ Create discrete RK4 model """
        ode_casadi = ca.Function("ode", [x,u], [ode(x,u)])
        k1 = ode_casadi(x, u)
        k2 = ode_casadi(x + dt/2*k1, u)
        k3 = ode_casadi(x + dt/2*k2, u)
        k4 = ode_casadi(x + dt*k3,u)
        xrk4 = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)    
        self.rk4 = ca.Function("ode_rk4", [x,u], [xrk4])

    
    def integrate(self, x0, u):
        """ Integrate one time sample dt

        # Arguments:
            x0: Initial state vector
            u: Input vector
        # Returns:
            x: Numpy array with x at t0 + dt
        """
        out = self.__I(x0=x0, p=u)
        return np.array(out["xf"]).flatten()


    def sim(self, x0, u, noise=False, clip_negative=False):
        """ Simulate system
        
        # Arguments:
            x0: Initial state
            u: Input matrix with the input for each timestep in the simulation horizon
            noise: If True, add gaussian noise using the noise covariance matrix
            clip_negative: If true, clip negative simulated outputs to zero
            
        # Output:
            Y_sim: Matrix with the simulated outputs (Nt, Ny)
        """
        
        Nt = np.size(u, 0)
        
        # Initial state of the system
        x = x0

        # Predefine matrix to collect noisy state outputs
        Y = np.zeros((Nt, self.__Nx))
    
        for t in range(Nt):
            u_t = u[t, :]    # control input for simulation
    
            x = self.integrate(x, u_t)
            Y[t, :] = x

            # Add normal white noise to state outputs
            if noise:
                Y[t, :] += np.random.multivariate_normal(
                                    np.zeros((self.__Nx)), self.__R)

            # Limit values to above 1e-8 to avvoid to avvoid numerical errors
            if clip_negative:
                if np.any(Y < 0):
                    print('Negative values!')
                    Y = Y.clip(min=1e-8)
        return Y


    def generate_training_data(self, N, uub, ulb, xub, xlb, noise=True):
        """ Generate training data using latin hypercube design
        
        # Arguments:
            N:   Number of data points to be generated
            uub: Upper input range (Nu,1)
            ulb: Lower input range (Nu,1)
            xub: Upper state range (Ny,1)
            xlb: Lower state range (Ny,1)
        """
        # Make sure boundry vectors are numpy arrays
        uub = np.array(uub)
        ulb = np.array(ulb)
        xub = np.array(xub)
        xlb = np.array(xlb)


        # Predefine matrix to collect noisy state outputs
        Y = np.zeros((N, self.__Nx))
        X = np.zeros((N, self.__Nx))
    
        # Create control input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Nu
        U = pyDOE.lhs(self.__Nu, samples=N, criterion='maximin')
        
        # Scale control inputs to correct range
        for k in range(N):
            U[k, :] = U[k, :] * (uub - ulb) + ulb
    
        # Create state input design using a latin hypecube
        # Latin hypercube design for unit cube [0,1]^Ny
        X = pyDOE.lhs(self.__Nx, samples=N, criterion='maximin')
        
        # Scale state inputs to correct range
        for k in range(N):
            X[k, :] = X[k, :] * (xub - xlb) + xlb

        for i in range(N):
            u_t = U[i, :]    # control input for simulation
            x_t = X[i, :]    # state input for simulation
    
            # Simulate system with x_t and u_t inputs for deltat time
            Y[i, :] = self.integrate(x_t, u_t)
            
            # Add normal white noise to state outputs
            if noise:
                Y[i, :] += np.random.multivariate_normal(
                                np.zeros((self.__Nx)), self.__R)
    
        # Concatenate previous states and inputs to obtain overall input to GP model
        X = np.hstack([X, U])
        return X, Y
    

    def plot(self, x0, u, numcols=2):
        """ Simulate and plot model
        
        # Arguments: 
            x0: Initial state
            u: Matrix with inputs for all time steps (Nt, Nu)
            numcols: Number of columns in the plot
        """
        y = self.sim(x0, u, noise=True)
        Nt = np.size(u, 0)
        t = np.linspace(0.0, (Nt - 1)* self.dt, Nt )
        numrows = int(np.ceil(self.__Nx / numcols))
        
        fig_x = plt.figure()
        for i in range(self.__Nx):
            ax = fig_x.add_subplot(numrows, numcols, i + 1)
            ax.plot(t, y[:, i], 'b-', marker='.', linewidth=1.0)
            ax.set_ylabel('x_' + str(i + 1))
            ax.set_xlabel('Time')
        plt.show()
