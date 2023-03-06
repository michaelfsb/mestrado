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
M_0 = 0.65 # Initial mass of hydrogen (Nm3)
M_min = 0.6 # Minimum mass of hydrogen (Nm3)
M_max = 1 # Maximum mass of hydrogen (Nm3)
I_e_0 = 30 # Initial current (A)
I_e_min = 1 # Minimum current (A)
I_e_max = 100 # Maximum current (A)

# Read irradiation and demand data from file
mat_contents = loadmat('problem-1/vetores_sol_carga.mat')

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
f_h2 = (N_el*i_el/F)*(11.126/(60*1000)) # Hydrogen production rate (Nm3/min)
v_el = v_el_0 + v_etd + v_el_hom_ion
v_ps = (N_ss*Vt*A*(lambertw(ca.exp(1)*(Iph/Irs+1))-1))
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 

# Lagrange cost function
f_q = ((N_el*v_el*i_el) - v_ps*i_ps)**2

# Diferential equations
m_h2_dot = f_h2 - HydrogenDemand(time)/60

f = ca.Function('f', [m_h2, i_el, time], [m_h2_dot, f_q], ['x', 'u', 't'], ['x_dot', 'L'])

#
t = np.linspace(0, Tf, num=N, endpoint=True)
h = [t[k+1]-t[k] for k in range(N-1)]

X = []
U = []

for k in range(N):
    X += [ca.MX.sym('X_' + str(k))]
    U += [ca.MX.sym('U_' + str(k))]

L = 0
g = []
lbg = []
ubg = []
for k in range(N-1):
    # 
    f_k, w_k = f(X[k], U[k], t[k])
    f_k_1, w_k_1 = f(X[k+1], U[k+1], t[k+1])
    L = L + .5*h[k]*(w_k + w_k_1)

    # Add equality constraint
    g +=  [.5*h[k]*(f_k + f_k_1) + X[k] - X[k+1]]
    lbg += [0]
    ubg += [0]

w=[]
w0 = []
lbw = []
ubw = []

for k in range(N):
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

# Set the initial condition for the state
lbw[1] = M_0
ubw[1] = M_0

# Solve the NLP
# Creat NPL Solver
prob = {'f': L, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# # Retrieve the control
# w_opt = sol['x'].full().flatten()
# i_el_opt = w_opt[0::2]
# m_h2_opt = w_opt[1::2]

# # Simulating the system with the solution
# Xs = ca.vertcat(M_0)
# f_h2_s = []     # Simulated hydrogen production rate
# ts = []         # Simulated time [min]
# th = []         # Simulated time [h]

# for s in range(N):
#     f_h2_s.append((N_el*i_el_opt[s]/F)*(11.126/(1000)))
#     ts.append(s*dt)
#     th.append(s*dt/60)

# # Plot results
# fig, axs = plt.subplots(4,1)
# fig.suptitle('Simulation results')
# fig.set_size_inches(6, 8)

# axs[0].step(th, i_el_opt, 'g-', where ='post')
# axs[0].set_ylabel('Electrolyzer current [A]')
# axs[0].grid(axis='both',linestyle='-.')
# axs[0].set_xticks(np.arange(0, 26, 2))

# axs[1].plot(th, m_h2_opt, 'b-')
# axs[1].set_ylabel('Hydrogen [Nm3]')
# axs[1].grid(axis='both',linestyle='-.')
# axs[1].set_xticks(np.arange(0, 26, 2))

# axs[2].plot(th, Irradiation(ts), 'g-')
# axs[2].set_ylabel('Solar irradiation')
# axs[2].grid(axis='both',linestyle='-.')
# axs[2].set_xticks(np.arange(0, 26, 2))

# axs[3].plot(th, HydrogenDemand(ts), 'r-', label='Demd')
# axs[3].plot(th, f_h2_s, 'b-', label='Prod')
# axs[3].grid(axis='both',linestyle='-.')
# axs[3].set_xticks(np.arange(0, 26, 2))
# axs[3].legend()
# axs[3].set_ylabel('H2 [Nm3/h]')
# axs[3].set_xlabel('Time [min]')

# plt.savefig('problem-1/ocp-trapezoid-collocation.png', bbox_inches='tight')