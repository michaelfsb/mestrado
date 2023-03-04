# Imports
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Define Lambert W function
def lambertw(x):
    E = 0.4586887;
    return (1+E)*ca.log(6/5*x/ca.log(12/5*x/ca.log(1+12/5*x)) ) -E*ca.log(2*x/ca.log(1+2*x))

# Preliminaries
Tf = 1440 # Final time (min)
#N = 2*Tf # Number of control intervals 
N = 90
M_0 = 0.7 # Initial mass of hydrogen (Nm3)
M_min = 0.1 # Minimum mass of hydrogen (Nm3)
M_max = 1 # Maximum mass of hydrogen (Nm3)
I_e_0 = 30 # Initial current (A)
I_e_std = 5 # Standby current (A)
I_e_min = 25 # Minimum current (A)
I_e_max = 100 # Maximum current (A)

# Read irradiation and demand data from file
mat_contents = loadmat('problem-1/vetores_sol_carga.mat') # Local
#mat_contents = loadmat('vetores_sol_carga.mat') # Remote

ini = Tf
fim = 2*Tf # Take second day

sol_real = mat_contents['sol_real'][ini:fim]
fdemanda = -0.18+mat_contents['carga_real'][ini:fim]
t_file = np.arange(1,len(sol_real)+1,1)

# Create interpolants
Irradiation = ca.interpolant("Irradiation", "bspline", [t_file], sol_real.flatten()) # Normalized irradiation
HydrogenDemand = ca.interpolant("HydrogenDemand", "bspline", [t_file], fdemanda.flatten()) # (Nm3/h)

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
f_h2 = p_el*(N_el*i_el/F)*(11.126/(60*1000)) # Hydrogen production rate (Nm3/min)
v_el = v_el_0 + v_etd + v_el_hom_ion
v_ps = (N_ss*Vt*A*(lambertw(ca.exp(1)*(Iph/Irs+1))-1))
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 

# Lagrange cost function
f_q = (p_el*(I_e_min-i_el))*((N_el*v_el*i_el) - v_ps*i_ps)**2 

# Diferential equations
m_h2_dot = f_h2 - HydrogenDemand(time)/60

# Integrate dynamics
# Foward Euler integration step

dt = Tf/N

f = ca.Function('f', [m_h2, i_el, p_el, time], [m_h2_dot, f_q])

X0 = ca.MX.sym('X0')
U = ca.MX.sym('U',2)
T = ca.MX.sym('T')

Xdto, Jk = f(X0, U[0], U[1], T)
X = X0+dt*Xdto
Q = Jk*dt

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

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob)

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
w_opt_i = w_opt[0::2]
w_opt_p = w_opt[1::2]

# Simulating the system with the solution
Xs = ca.vertcat(M_0)
m = []          # Simulated hydrogen mass
f_h2_s = []     # Simulated hydrogen production rate
ts = []         # Simulated time [min]
th = []         # Simulated time [h]
i_c = []        # Control current

for s in range(N):
    i_c.append(w_opt_i[s]*w_opt_p[s]) 
    Fs = FI(x0=Xs, u=[w_opt_i[s], w_opt_p[s]], t=s*dt)
    f_h2_s.append(w_opt_p[s]*(N_el*i_c[s]/F)*(11.126/(1000)))
    Xs = Fs['xf'] 
    m.append(Xs.full().flatten()[0])
    ts.append(s*dt)
    th.append(s*dt/60)

# Plot results
fig, axs = plt.subplots(4,1)
fig.suptitle('Simulation results')
fig.set_size_inches(6, 8)

axs[0].step(th, i_c, 'g-', where ='post')
axs[0].set_ylabel('Electrolyzer current [A]')
axs[0].grid(axis='both',linestyle='-.')
axs[0].set_xticks(np.arange(0, 26, 2))

axs[1].plot(th, m, 'b-')
axs[1].set_ylabel('Hydrogen [Nm3]')
axs[1].grid(axis='both',linestyle='-.')
axs[1].set_xticks(np.arange(0, 26, 2))

axs[2].plot(th, Irradiation(ts), 'g-')
axs[2].set_ylabel('Solar irradiation')
axs[2].grid(axis='both',linestyle='-.')
axs[2].set_xticks(np.arange(0, 26, 2))

axs[3].plot(th, HydrogenDemand(ts), 'r-', label='Demd')
axs[3].plot(th, f_h2_s, 'b-', label='Prod')
axs[3].grid(axis='both',linestyle='-.')
axs[3].set_xticks(np.arange(0, 26, 2))
axs[3].legend()
axs[3].set_ylabel('H2 [Nm3/h]')
axs[3].set_xlabel('Time [min]')

plt.show()