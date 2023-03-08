# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from models.photovoltaic_panel import pv_model
from models.electrolyzer import N_el, electrolyzer_model
from models.tank import thank_model
from models.input_data import Irradiation, HydrogenDemand

# Preliminaries
Tf = 1440   # Final time (min)
N = 90      # Number of control intervals

M_0 = 0.65      # Initial volume of hydrogen (Nm3)
M_min = 0.6     # Minimum volume of hydrogen (Nm3)
M_max = 1       # Maximum volume of hydrogen (Nm3)
I_e_0 = 30      # Initial current (A)
I_e_min = 1     # Minimum current (A)
I_e_std = 5     # Standby current (A)
I_e_max = 100   # Maximum current (A)

# Declare variables
v_h2 = ca.MX.sym('v_h2') # State - Volume of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
p_el = ca.MX.sym('p_el') # Control - Electrolyzer state
time = ca.MX.sym('time') # Time

# Models equations
[f_h2, v_el] = electrolyzer_model(i_el) # Hydrogen production rate (Nm3/min) and eletrolyzer voltage (V)
f_h2 = p_el*f_h2 # Switching between electrolyzer states
[i_ps, v_ps] = pv_model(Irradiation(time)) # Power and voltage of the photovoltaic panel (A, V)
v_h2_dot = thank_model(f_h2, HydrogenDemand(time)/60) # Hydrongen volume rate in the tank (Nm3/min)

# Lagrange cost function
f_l = (p_el*(I_e_min-i_el))*((N_el*v_el*i_el) - v_ps*i_ps)**2 

dt = Tf/N
# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
DT = Tf/N/M
f = ca.Function('f', [v_h2, i_el, p_el, time], [v_h2_dot, f_l])
X0 = ca.MX.sym('X0')
U = ca.MX.sym('U', 2)
T = ca.MX.sym('T')
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U[0], U[1], T)
    k2, k2_q = f(X + DT/2 * k1, U[0], U[1], T)
    k3, k3_q = f(X + DT/2 * k2, U[0], U[1], T)
    k4, k4_q = f(X + DT * k3, U[0], U[1], T)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
FI = ca.Function('FI', [X0, U, T], [X, Q], ['x0', 'u', 't'], ['xf', 'qf'])

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
    # New NLP variable for the control [i_el, p_el]
    Uk = ca.MX.sym('U_' + str(k), 2)
    w += [Uk]
    lbw += [I_e_std, 0]
    ubw += [I_e_max, 1]
    w0 += [I_e_0, 1]

    # Integrate till the end of the interval
    Fk = FI(x0=Xk, u=Uk, t=k*dt)
    Xk = Fk['xf']
    J = J + Fk['qf']

    # Add inequality constraint: x1 is bound to be between 0 and infinity
    g += [Xk[0]]
    lbg += [M_min]
    ubg += [M_max]

# Solve the NLP
# Creat NPL Solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# NLP solver options
opts = {"ipopt.output_file" : "results/ocp-2-single-shooting.txt"}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Call the solver
sol = solver(x0=w0,     # Initial guess
             lbx=lbw,   # Lower variable bound
             ubx=ubw,   # Upper variable bound
             lbg=lbg,   # Lower constraint bound
             ubg=ubg)   # Upper constraint bound

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the control
w_opt = sol['x'].full().flatten()
i_el_opt = w_opt[0::2]
p_el_opt = w_opt[1::2]

# Simulating the system with the solution
Xs = ca.vertcat(M_0)
v_h2_s = []     # Simulated hydrogen mass
i_c = []        # Control current

for s in range(N):
    Fs = FI(x0=Xs, u=[i_el_opt[s], p_el_opt[s]], t=s*dt)
    Xs = Fs['xf'] 
    v_h2_s.append(Xs.full().flatten()[0])
    i_c.append(i_el_opt[s]*p_el_opt[s]) 

# Retrieve the optimization status
optimzation_status = ''
with open('results/ocp-2-single-shooting.txt') as file:
    for line in file:
        if line.startswith('EXIT'):
            optimzation_status = line.strip()[5:-1]

# Plot results
ts = np.linspace(0, Tf/60, N)
fig, axs = plt.subplots(2,1)
fig.suptitle('Simulation results: ' + optimzation_status)

axs[0].step(ts, i_c, 'g-', where ='post')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(ts, v_h2_s, 'b-')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [min]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig('results/ocp-2-single-shooting.png', bbox_inches='tight', dpi=300)