# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

from models.photovoltaic_panel import pv_model
from models.electrolyzer import N_el, electrolyzer_model
from models.tank import thank_model
from models.input_data import Irradiation, HydrogenDemand
from utils import files

# Preliminaries
Tf = 1440   # Final time (min)
N = 90      # Number of control intervals

M_0 = 0.65      # Initial volume of hydrogen (Nm3)
M_min = 0.6     # Minimum volume of hydrogen (Nm3)
M_max = 1       # Maximum volume of hydrogen (Nm3)
I_e_0 = 30      # Initial current (A)
I_e_min = 1     # Minimum current (A)
I_e_max = 100   # Maximum current (A)

# Declare variables
v_h2 = ca.MX.sym('v_h2') # State - Volume of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
time = ca.MX.sym('time') # Time

# Models equations
[f_h2, v_el] = electrolyzer_model(i_el) # Hydrogen production rate (Nm3/min) and eletrolyzer voltage (V)
[i_ps, v_ps] = pv_model(Irradiation(time)) # Power and voltage of the photovoltaic panel (A, V)
v_h2_dot = thank_model(f_h2, HydrogenDemand(time)) # Hydrongen volume rate in the tank (Nm3/min)

# Lagrange cost function
f_l = ((N_el*v_el*i_el) - v_ps*i_ps)**2

# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
DT = Tf/N/M
f = ca.Function('f', [v_h2, i_el, time], [v_h2_dot, f_l])
X0 = ca.MX.sym('X0')
U = ca.MX.sym('U')
T = ca.MX.sym('T')
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U, T)
    k2, k2_q = f(X + DT/2 * k1, U, T)
    k3, k3_q = f(X + DT/2 * k2, U, T)
    k4, k4_q = f(X + DT * k3, U, T)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
FI = ca.Function('FI', [X0, U, T], [X, Q], ['x0', 'p', 't'], ['xf', 'qf'])

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
dt = Tf/N

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [I_e_min]
    ubw += [I_e_max]
    w0 += [I_e_0]

    # Integrate till the end of the interval
    Fk = FI(x0=Xk, p=Uk, t=k*dt)
    Xk_end = Fk['xf']
    J = J + Fk['qf']

    # New NLP variable for state at end of interval
    Xk = ca.MX.sym('X_' + str(k+1))
    w   += [Xk]
    lbw += [M_min]
    ubw += [M_max]
    w0  += [M_0]

    # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0]
    ubg += [0]

# Solve the NLP
# Creat NPL Solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# NLP solver options
ipopt_log_file = files.get_log_file_name(__file__)
opts = {"ipopt.output_file" : ipopt_log_file}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the solution
w_opt = sol['x'].full().flatten()
i_el_opt = w_opt[0::2]
v_h2_opt = w_opt[1::2]

# Retrieve the optimization status
optimzation_status = files.get_optimization_status(ipopt_log_file)

# Plot results
ts = np.linspace(0, Tf/60, N)
fig, axs = plt.subplots(2,1)
fig.suptitle('Simulation results: ' + optimzation_status)

axs[0].step(ts, i_el_opt, 'g-', where ='post')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(ts, v_h2_opt, 'b-')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [h]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)