# Control of the Van der Pol
# oscillator using pure casadi.
from sys import path
path.append(r"C:\Users\helgeanl\Google Drive\NTNU\Masteroppgave\casadi-py36-v3.4.0")
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt
import time
#<<ENDCHUNK>>

# Define model and get simulator.
Delta = .03
Nt = 20
Nx = 6
Nu = 3
def ode(x, u):
    # Model Parameters (Gao et al., 2014)
    g   = 9.18                          # Gravity [m/s^2]
    m   = 2050                          # Vehicle mass [kg]
    Iz  = 3344                          # Yaw inertia [kg*m^2]
    Cr   = 65000                        # Tyre corning stiffness [N/rad]
    Cf   = 65000                        # Tyre corning stiffness [N/rad]
    mu  = 0.5                           # Tyre friction coefficient
    l   = 4.0                           # Vehicle length
    lf  = 2.0                           # Distance from CG to the front tyre
    lr  = l - lf                        # Distance from CG to the rear tyre
    Fzf = lr * m * g / (2 * l)          # Vertical load on front wheels
    Fzr = lf * m * g / (2 * l)          # Vertical load on rear wheels
    eps = 1e-8

    dxdt = [
                1/m * (m*x[1]*x[2] + 2*mu*Fzf*u[0] + 2*Cf*u[2]**2
                    - 2*Cf*u[2] * (x[1] + lf*x[2]) / (x[0] + eps) + 2*mu*Fzr*u[1]),
                1/m * (-m*x[0]*x[2] + 2*mu*Fzf*u[2]*u[0]
                    + 2*Cf*(x[1] + lf*x[2]) / (x[0] + eps) - 2*Cf*u[2]
                    + 2*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                1/Iz * (2*lf*mu*Fzf*u[0]*u[2] + 2*lf*Cf*(x[1] + lf*x[2]) / (x[0] + eps)
                    - 2*lf*Cf*u[2] - 2*lr*Cr*(x[1] - lf*x[2]) / (x[0] + eps)),
                x[2],
                x[0]*casadi.cos(x[3]) - x[1]*casadi.sin(x[3]),
                x[0]*casadi.sin(x[3]) + x[1]*casadi.cos(x[3])
            ]
    return np.array(dxdt)

#<<ENDCHUNK>>

# Define symbolic variables.
x = casadi.SX.sym("x",Nx)
u = casadi.SX.sym("u",Nu)

# Make integrator object.
ode_integrator = dict(x=x,p=u,
    ode=ode(x,u))
intoptions = {
    "abstol" : 1e-8,
    "reltol" : 1e-8,
    "tf" : Delta,
}
vdp = casadi.integrator("int_ode",
    "cvodes", ode_integrator, intoptions)

#<<ENDCHUNK>>

# Then get nonlinear casadi functions
# and rk4 discretization.
ode_casadi = casadi.Function(
    "ode",[x,u],[ode(x,u)])

k1 = ode_casadi(x, u)
k2 = ode_casadi(x + Delta/2*k1, u)
k3 = ode_casadi(x + Delta/2*k2, u)
k4 = ode_casadi(x + Delta*k3,u)
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
ode_rk4_casadi = casadi.Function(
    "ode_rk4", [x,u], [xrk4])

#<<ENDCHUNK>>

# Define stage cost and terminal weight.
lfunc = (casadi.mtimes(x.T, x)
    + casadi.mtimes(u.T, u))
l = casadi.Function("l", [x,u], [lfunc])

Pffunc = casadi.mtimes(x.T, x)
Pf = casadi.Function("Pf", [x], [Pffunc])

#<<ENDCHUNK>>

# Bounds on u.
ulb = [-.5, -.5, -.1,]
uub = [.5, .5, .1,]

xlb = [1, -.5, -2.0, -2.0, .0, .0]
xub = [30, .5, 2.0, 2.0, np.inf, np.inf]
#<<ENDCHUNK>>

# Make optimizers.
x0 = np.array([10, 0.0, 0.0, 0.0, 0.0 , 0.0])

# Create variables struct.
var = ctools.struct_symSX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub
    varlb["x",t,:] = xlb
    varub["x",t,:] = xub

dx_s = casadi.SX.sym('dx')
dy_s = casadi.SX.sym('dy')
dpsi_s = casadi.SX.sym('dpsi')
delta_f_s = casadi.SX.sym('delta_f') 
lf  = 2.0 
lr  = 2.0
slip_min = -4 * np.pi / 180
slip_max = 4 * np.pi / 180
slip_f = casadi.Function('slip_f', [dx_s, dy_s, dpsi_s, delta_f_s],
                     [(dy_s + lf*dpsi_s)/(dx_s + 1e-6)   - delta_f_s])
slip_r = casadi.Function('slip_r', [dx_s, dy_s, dpsi_s],
                     [(dy_s - lr*dpsi_s)/(dx_s + 1e-6)])


# Now build up constraints and objective.
obj = casadi.SX(0)
con = []
con_ineq = []
con_ineq_lb = []
con_ineq_ub = []
for t in range(Nt):
    con.append(ode_rk4_casadi(var["x",t],
        var["u",t]) - var["x",t+1])
    
    # Slip angle constraint
    dx = var['x', t, 0]
    dy = var['x', t, 1]
    dpsi = var['x', t, 2]
    delta_f = var['u', t, 2]

    con_ineq.append((dy + 2*dpsi)/(dx+1e-6) -delta_f - slip_max)
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append(slip_min - (dy + 2*dpsi)/(dx+1e-6) + delta_f )
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append((dy - 2*dpsi)/(dx+1e-6) - slip_max )
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    con_ineq.append(slip_min - (dy - 2*dpsi)/(dx+1e-6) )
    con_ineq_ub.append(0)
    con_ineq_lb.append(-np.inf)
    
    
    obj += l(var["x",t], var["u",t])
obj += Pf(var["x",Nt])

# Build solver object.
con = casadi.vertcat(*con, *con_ineq)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))
conlb = casadi.vertcat(*conlb, *con_ineq_lb)
conub = casadi.vertcat(*conub, *con_ineq_ub)

nlp = dict(x=var, f=obj, g=con)
nlpoptions = {
    "ipopt" : {
        "print_level" : 0,
        "max_cpu_time" : 60,
        'linear_solver' : 'ma27',
    },
    "print_time" : False,
}
solver = casadi.nlpsol("solver",
    "ipopt", nlp, nlpoptions)

#<<ENDCHUNK>>

# Now simulate.
Nsim = int(2 / Delta)
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
for t in range(Nsim):
    # Fix initial state.    
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    args = dict(x0=varguess,
                lbx=varlb,
                ubx=varub,
                lbg=conlb,
                ubg=conub)   
    
    #<<ENDCHUNK>>    
    
    # Solve nlp.    
    t1 = -time.time()
    sol = solver(**args)
    status = solver.stats()["return_status"]
    optvar = var(sol["x"])
    t1 += time.time()
    #<<ENDCHUNK>>    
    
    # Print stats.
    print("%d: %s - Time: %f" % (t,status, t1))
    u[t,:] = np.array(optvar["u",0,:]).flatten()
    
    #<<ENDCHUNK>>    
    
    # Simulate.
    vdpargs = dict(x0=x[t,:],
                   p=u[t,:])
    out = vdp(**vdpargs)
    x[t+1,:] = np.array(
        out["xf"]).flatten()

#<<ENDCHUNK>>
    
# Plots.
fig = plt.figure()
numrows = max(Nx,Nu)
numcols = 2

# u plots. Need to repeat last element
# for stairstep plot.
u = np.concatenate((u,u[-1:,:]))
for i in range(Nu):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1))
    ax.step(times,u[:,i],"-k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control %d" % (i + 1))

# x plots.    
for i in range(Nx):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

fig.tight_layout(pad=.5)
#import mpctools.plots # Need to grab one function to show plot.
#mpctools.plots.showandsave(fig,"comparison_casadi.pdf")
