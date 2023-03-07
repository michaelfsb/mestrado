# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from models.photovoltaic_panel import pv_model
from models.electrolyzer import F, N_el, electrolyzer_model
from models.tank import thank_model
from models.input_data import Irradiation, HydrogenDemand

# Preliminaries
Tf = 1440   # Final time (min)
N = 90      # Number of control intervals

M_0 = 0.65      # Initial mass of hydrogen (Nm3)
M_min = 0.6     # Minimum mass of hydrogen (Nm3)
M_max = 1       # Maximum mass of hydrogen (Nm3)
I_e_0 = 30      # Initial current (A)
I_e_min = 1     # Minimum current (A)
I_e_max = 100   # Maximum current (A)

# Declare variables
v_h2 = ca.MX.sym('v_h2') # State - Mass of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
time = ca.MX.sym('time') # Time

# Models equations
[f_h2, v_el] = electrolyzer_model(i_el) # Hydrogen production rate (Nm3/min) and eletrolyzer voltage (V)
[i_ps, v_ps] = pv_model(Irradiation(time)) # Power and voltage of the photovoltaic panel (A, V)
v_h2_dot = thank_model(f_h2, HydrogenDemand(time)/60)

# Lagrange cost function
f_l = ((N_el*v_el*i_el) - v_ps*i_ps)**2

# Integrate dynamics
# Foward Euler integration step
dt = Tf/N

f = ca.Function('f', [v_h2, i_el, time], [v_h2_dot, f_l])

X0 = ca.MX.sym('X0')
U = ca.MX.sym('U')
T = ca.MX.sym('T')

Xdto, Jk = f(X0, U, T)
X = X0+dt*Xdto
Q = Jk*dt

FI = ca.Function('FI', [X0, U, T], [X, Q], ['x0', 'u', 't'], ['xf', 'lf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g = []
lbg = []
ubg = []

# Integrate through time to obtain constraints at each time step
Xk = ca.vertcat(M_0)

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [I_e_min]
    ubw += [I_e_max]
    w0 += [I_e_0]

    # Integrate till the end of the interval
    Fk = FI(x0=Xk, u=Uk, t=k*dt)
    Xk = Fk['xf']
    J = J + Fk['lf']

    # Add inequality constraint: x1 is bound to be between 0 and infinity
    g += [Xk[0]]
    lbg += [M_min]
    ubg += [M_max]

# Solve the NLP
# Creat NPL Solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# NLP solver options
opts = {"ipopt.output_file" : "results/ocp-1-trapezoid-collocation.txt"}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the control
w_opt = sol['x'].full().flatten()

# Simulating the system with the solution
Xs = ca.vertcat(M_0)
m = []          # Simulated hydrogen mass
f_h2_s = []     # Simulated hydrogen production rate
ts = []         # Simulated time [min]
th = []         # Simulated time [h]

for s in range(N):
    Fs = FI(x0=Xs, u=w_opt[s], t=s*dt)
    f_h2_s.append((N_el*w_opt[s]/F)*(11.126/(1000)))
    Xs = Fs['xf'] 
    m.append(Xs.full().flatten()[0])
    ts.append(s*dt)
    th.append(s*dt/60)

# Plot results
fig, axs = plt.subplots(2,1)
fig.suptitle('Simulation results')

axs[0].step(th, w_opt, 'g-', where ='post')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(th, m, 'b-')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [h]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig('results/ocp-1-single-shooting.png', bbox_inches='tight')