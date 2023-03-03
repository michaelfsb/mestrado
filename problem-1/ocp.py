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

# Read irradiation and demand data from file
#mat_contents = loadmat('problem-1/vetores_sol_carga.mat') # Local
mat_contents = loadmat('vetores_sol_carga.mat') # Remote

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

# Integrate dynamics
# Foward Euler integration step

dt = Tf/N

f = ca.Function('f', [m_h2, i_el, time], [m_h2_dot, f_q])

X0 = ca.MX.sym('X0')
U = ca.MX.sym('U')
T = ca.MX.sym('T')

Xdto, Jk = f(X0, U, T)
X = X0+dt*Xdto
Q = Jk*dt

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
Xk = ca.vertcat(0.5)

for k in range(N):
    # New NLP variable for the control
    Uk = ca.MX.sym('U_' + str(k))
    w += [Uk]
    lbw += [1]
    ubw += [150]
    w0 += [30]

    # Integrate till the end of the interval
    Fk = FI(x0=Xk, p=Uk, t=k*dt)
    Xk = Fk['xf']
    J = J + Fk['qf']

    # Add inequality constraint: x1 is bound to be between 0 and infinity
    g += [Xk[0]]
    lbg += [.6]
    ubg += [.8]

# Solve the NLP
# Creat NPL Solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the control
w_opt = sol['x'].full().flatten()

# Simulating the system with the solution
Xs = ca.vertcat(0.5)
m = []          # Simulated hydrogen mass
f_h2_s = []     # Simulated hydrogen production rate
ts = []         # Simulated time [min]
th = []         # Simulated time [h]

for s in range(N):
    Fs = FI(x0=Xs, p=w_opt[s], t=s*dt)
    f_h2_s.append((N_el*w_opt[s]/F)*(11.126/(60*1000)))
    Xs = Fs['xf'] 
    m.append(Xs.full().flatten()[0])
    ts.append(s*dt)
    th.append(s*dt/60)

# Plot results
fig, axs = plt.subplots(4,1)
fig.suptitle('Simulation results')
fig.set_size_inches(10, 10)

axs[0].step(th, w_opt, 'g-', where ='post')
axs[0].set_ylabel('Electrolyzer current [A]')

axs[1].plot(th, m, 'b-')
axs[1].set_ylabel('Hydrogen [Nm3]')

axs[2].plot(th, Irradiation(ts), 'g-')
axs[2].set_ylabel('Solar irradiation')

axs[3].plot(th, HydrogenDemand(ts)/60, 'r-', label='Demand')
axs[3].plot(th, f_h2_s, 'b-', label='Production')
axs[3].legend()
axs[3].set_ylabel('H2 [Nm3/min]')
axs[3].set_xlabel('Time [min]')

plt.show()