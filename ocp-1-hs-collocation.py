## Hermite-Simpson collocation method (separated form) 

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
I_e_max = 100   # Maximum current (A)

# Declare variables
v_h2 = ca.MX.sym('v_h2') # State - Volume of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
time = ca.MX.sym('time') # Time

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(i_el) # Hydrogen production rate (Nm3/min) and eletrolyzer voltage (V)
[i_ps, v_ps, p_ps] = pv_model(Irradiation(time)) # Power and voltage of the photovoltaic panel (A, V)
v_h2_dot = thank_model(f_h2, HydrogenDemand(time)) # Hydrongen volume rate in the tank (Nm3/min)

# Lagrange cost function
f_l = (p_el - p_ps)**2

# Creat NPL problem
f = ca.Function('f', [v_h2, i_el, time], [v_h2_dot, f_l], ['x', 'u', 't'], ['x_dot', 'L'])
t = np.linspace(0, Tf, num=N, endpoint=True)
h = Tf/N

X = []
U = []

for k in np.arange(0, N-.5, .5):
    X += [ca.MX.sym('X_' + str(k))]
    U += [ca.MX.sym('U_' + str(k))]

L = 0
g = []
lbg = []
ubg = []
for k in np.arange(0, 2*N-2, 2):
    i = int(k/2)
    f_k_0, w_k_0 = f(X[k], U[k], t[i])
    f_k_1, w_k_1 = f(X[k+1], U[k+1], t[i]+h/2)
    f_k_2, w_k_2 = f(X[k+2], U[k+2], t[i+1])
    L = L + h*(w_k_0 + 4*w_k_1 + w_k_2)/6

    # Add equality constraint
    g +=  [X[k+2] - X[k] - h*(f_k_0 + 4*f_k_1 + f_k_2)/6]
    lbg += [0]
    ubg += [0]
    g +=  [X[k+1] - (X[k+2] + X[k])/2 - h*(f_k_0 - f_k_2)/8]
    lbg += [0]
    ubg += [0]

w=[]
w0 = []
lbw = []
ubw = []

# For plotting x and u given w
x_plot = []
u_plot = []

for k in range(2*N-1):
    # New NLP variable for the control
    w += [U[k]]
    lbw += [I_e_min]
    ubw += [I_e_max]
    w0 += [I_e_0]

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
x_opt = x_opt.full().flatten()
u_opt = u_opt.full().flatten()

# Plot results
t = np.linspace(0, Tf, num=2*N-1, endpoint=True)
f_x_opt = interpolate.interp1d(t, x_opt, kind=3) # k=3 is a cubic spline
f_u_opt = interpolate.interp1d(t, u_opt, kind=2) # k=2 is a quadratic spline
t_new = np.arange(0, Tf, 0.1)

fig, axs = plt.subplots(2,1)
fig.suptitle('Simulation Results: ' + optimzation_status + '\nCost: ' + str(sol['f']) + ' (W)')

#axs[0].plot(t/60, u_opt, '.r')
axs[0].plot(t/60, u_opt, '.r', t_new/60, f_u_opt(t_new), '-b')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(t/60, x_opt, '.r', t_new/60, f_x_opt(t_new), '-g')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [h]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)

# Evaluate the error
te = np.linspace(0, Tf, num=10*N, endpoint=True)
f_interpolated = interpolate.interp1d(t, f(x_opt, u_opt, t)[0].full().flatten(), kind=2)
error = f(f_x_opt(te), f_u_opt(te), te)[0] - f_interpolated(te)

# Plot error
fig2 = plt.figure(2)
fig2.suptitle('Erro in differential equations')
plt.plot(te/60, error, '.', t/60, np.zeros(len(t)), '.r')
plt.ylabel('Error')
plt.xlabel('Time [h]')
plt.grid(axis='both',linestyle='-.')
plt.savefig(files.get_plot_error_file_name(__file__), bbox_inches='tight', dpi=300)