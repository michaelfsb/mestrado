# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from utils.math import lambertw
from models.input_data import Irradiation, HydrogenDemand

# Preliminaries
Tf = 1440 # Final time (min)
#N = 2*Tf # Number of control intervals 
N = 30
M_0 = 0.65 # Initial mass of hydrogen (Nm3)
M_min = 0.6 # Minimum mass of hydrogen (Nm3)
M_max = 1 # Maximum mass of hydrogen (Nm3)
I_e_0 = 30 # Initial current (A)
I_e_std = 5 # Standby current (A)
I_e_min = 25 # Minimum current (A)
I_e_max = 100 # Maximum current (A)

# Declare constants
R = 8.314 # Gas constant
F = 96485 # Faraday constant
Q = 1.6e-19 # Elementary charge
K = 1.38e-23 # Boltzmann constant

# Declare electrolyzer parameters
A_el = 212.5            # Stack area
N_el = 22500            # Number of cells
P_h2 = 6.9              # Hydrogen partial pressure
P_o2 = 1.3              # Oxygen partial pressure
I_ao = 1.0631e-6        # Anode current density 
I_co = 1e-3             # Cathode current density
delta_b = 178e-6        # Membrane thickness
lambda_b = 21           # Membrana water content
t_el = 298              # Temperature

# Declare photovoltaic parameters
N_ps = 8            # Number of panels in parallel
N_ss = 300          # Number of panels in series
T_ps =  298         # Temperature
Tr = 298            # Reference temperature
Isc = 3.27          # Short circuit current at Tr
Kl = 0.0017         # Short circuit current temperature coeff
Ior = 2.0793e-6     # Ior - Irs at Tr
Ego = 1.1           # Band gap energy of the semiconductor
A = 1.6             # Factor. cell deviation from de ideal pn junction

# Declare variables
m_h2 = ca.MX.sym('m_h2') # State - Mass of hydrogen
i_el = ca.MX.sym('i_el') # Control - Electrical current in electrolyzer
p_el = ca.MX.sym('p_el') # Control - Electrolyzer state
time = ca.MX.sym('time') # Time

# Intermediate electrolyzer variables
i = i_el/A_el # Current density
ro_b = (0.005139*lambda_b - 0.00326) * ca.exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*ca.log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
v_etd = (R*t_el/F)*ca.asinh(.5*i/I_ao) + (R*t_el/F)*ca.asinh(.5*i/I_co) + i*delta_b/ro_b # Eletrode overpotential
v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

# Intermediate photovoltaic variables
Vt = K*T_ps/Q
Iph = (Isc+Kl*(T_ps-Tr))*Irradiation(time)
Irs = Ior*(T_ps/Tr)** 3*ca.exp(Q*Ego*(1/Tr-1/T_ps)/(K*A))

# Algebraic equations
f_h2 =  p_el*(N_el*i_el/F)*(11.126/(60*1000)) # Hydrogen production rate (Nm3/min)
v_el = v_el_0 + v_etd + v_el_hom_ion
v_ps = (N_ss*Vt*A*(lambertw(ca.exp(1)*(Iph/Irs+1))-1))
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 

# Lagrange cost function
f_q = (p_el*(I_e_min-i_el))*((N_el*v_el*i_el) - v_ps*i_ps)**2

# Diferential equations
m_h2_dot = f_h2 - HydrogenDemand(time)/60

f = ca.Function('f', [m_h2, i_el, p_el, time], [m_h2_dot, f_q], ['x', 'u_i', 'u_p', 't'], ['x_dot', 'L'])

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
opts = {"ipopt.output_file" : "results/ocp-2-trapezoid-collocation.txt"}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Retrieve the optimization status
optimzation_status = ''
with open('results/ocp-2-trapezoid-collocation.txt') as file:
    for line in file:
        if line.startswith('EXIT'):
            optimzation_status = line.strip()[5:-1]

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
fig.suptitle('Simulation Results: ' + optimzation_status)

axs[0].plot(t, i_el_opt, '.r', t_new, f_u_opt(t_new), '-b')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(t, *x_opt, '.r', t_new, *f_x_opt(t_new), '-g')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].set_xlabel('Time [h]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

plt.savefig('results/ocp-2-trapezoid-collocation.png', bbox_inches='tight', dpi=300)