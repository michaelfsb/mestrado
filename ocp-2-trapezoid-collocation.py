# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from models.photovoltaic_panel import pv_model
from models.electrolyzer import electrolyzer_model
from models.tank import thank_model
from models.input_data import Irradiation, HydrogenDemand
from utils import files

# Preliminaries
Tf = 1440   # Final time (min)
N = 90      # Number of control intervals

M_0 = 0.65      # Initial volume of hydrogen (Nm3)
M_min = 0.6     # Minimum volume of hydrogen (Nm3)
M_max = 2.5      # Maximum volume of hydrogen (Nm3)
I_e_0 = 30      # Initial current (A)
I_e_min = 1     # Minimum current (A)
I_e_std = 5     # Standby current (A)
I_e_max = 100   # Maximum current (A)

# Declare variables
v_h2 = ca.MX.sym('v_h2') # State - Volume of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
s_el = ca.MX.sym('s_el') # Control - Electrolyzer state
time = ca.MX.sym('time') # Time

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(i_el) # Hydrogen production rate (Nm3/min) and eletrolyzer voltage (V)
f_h2 = s_el*f_h2 # Switching between electrolyzer states
[i_ps, v_ps, p_ps] = pv_model(Irradiation(time)) # Power and voltage of the photovoltaic panel (A, V)
v_h2_dot = thank_model(f_h2, HydrogenDemand(time)) # Hydrongen volume rate in the tank (Nm3/min)

# Lagrange cost function
f_l = (s_el*(I_e_min-i_el))*(p_el - p_ps)**2

f = ca.Function('f', [v_h2, i_el, s_el, time], [v_h2_dot, f_l], ['x', 'u_i', 'u_p', 't'], ['x_dot', 'L'])

# Creat NPL problem
t = np.linspace(0, Tf, num=N, endpoint=True)
h = [t[k+1]-t[k] for k in range(N-1)]

X = []
U = []

for k in range(N):
    X += [ca.MX.sym('X_' + str(k))]
    U += [ca.MX.sym('U_' + str(k), 2)]

L = 0
g = []
lbg = []
ubg = []
for k in range(N-1):
    # 
    f_k, w_k = f(X[k], U[k][0], U[k][1], t[k])
    f_k_1, w_k_1 = f(X[k+1], U[k+1][0], U[k+1][1], t[k+1])
    L = L + .5*h[k]*(w_k + w_k_1)

    # Add equality constraint
    g +=  [.5*h[k]*(f_k + f_k_1) + X[k] - X[k+1]]
    lbg += [0]
    ubg += [0]

w=[]
w0 = []
lbw = []
ubw = []

# For plotting x and u given w
x_plot = []
u_plot = []

for k in range(N):
    # New NLP variable for the control
    w += [U[k]]
    lbw += [I_e_std, 0]
    ubw += [I_e_max, 1]
    w0 += [I_e_0, 1]

    # New NLP variable for state at end of interval
    w   += [X[k]]
    lbw += [M_min]
    ubw += [M_max]
    w0  += [M_0]
    x_plot += [X[k]]
    u_plot += [U[k]]

# Set the initial condition for the state
lbw[1] = M_0
ubw[1] = M_0

# Concatenate vectors
w = ca.vertcat(*w)
g = ca.vertcat(*g)
x_plot = ca.horzcat(*x_plot)
u_plot = ca.horzcat(*u_plot)

# Solve the NLP
# Creat NPL Solver
prob = {'f': L, 'x': w, 'g': g}

# NLP solver options
ipopt_log_file = files.get_log_file_name(__file__)
opts = {"ipopt.output_file" : ipopt_log_file}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the optimization status
optimzation_status = files.get_optimization_status(ipopt_log_file)

# Retrieve the solution
trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])
x_opt, u_opt = trajectories(sol['x'])
x_opt = x_opt.full()
u_opt = u_opt.full()
i_el_opt = u_opt[0]*u_opt[1] 

# Plot results
t = t/60
f_x_opt = interpolate.interp1d(t, x_opt, kind='quadratic')
f_u_opt = interpolate.interp1d(t, i_el_opt, kind='linear')
t_new = np.arange(0, 24, 0.1)

fig, axs = plt.subplots(2,1)
fig.suptitle('Simulation Results: ' + optimzation_status + '\nCost: ' + str(sol['f']) + ' (W)')

axs[0].plot(t, i_el_opt, '.r', t_new, f_u_opt(t_new), '-b')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(t, *x_opt, '.r', t_new, *f_x_opt(t_new), '-g')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [h]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)