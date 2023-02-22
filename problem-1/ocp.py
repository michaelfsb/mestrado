# Imports
import casadi as ca
import numpy as np
from scipy.io import loadmat

# Define Lambert W function
def lambertw(x, max_iter=10):
    """Approximate the Lambert W function using Newton's method"""
    w = ca.MX.zeros(x.shape)
    for i in range(max_iter):
        w_next = w - (w*ca.exp(w) - x) / (ca.exp(w) + w*(ca.exp(w)+1))
        w = w_next
    return w

# Preliminaries
Tf = 1440 # Final time (min)
#N = 2*Tf # Number of control intervals 
N = 360

# Read irradiation and demand data from file
mat_contents = loadmat('problem-1/vetores_sol_carga.mat')

ini = Tf
fim = 2*Tf # Take second day

sol_real = mat_contents['sol_real'][ini:fim]
fdemanda = -0.18+mat_contents['carga_real'][ini:fim]
t_file = np.arange(1,len(sol_real)+1,1)

# Create interpolants
Irradiation = ca.interpolant("Irradiation", "bspline", [t_file], sol_real.flatten())
HydrogenDemand = ca.interpolant("HydrogenDemand", "bspline", [t_file], fdemanda.flatten())

# Declare constants
R = 8.314 # Gas constant
F = 96485 # Faraday constant
Q = 1.6e-19 # Elementary charge
K = 1.38e-23 # Boltzmann constant

# Declare electrolyzer parameters
N_c = 120        # Number of cells
T = 298          # Temperature cell
delta = 100e-6   # Thickness of membrane (tirei do google)
lambda_ = 17     # degree of humidification of the membrane
alpha_A = 2      # 
i_0A = 1e-10     # 
alpha_C = .5     # 
i_0C = 1e-10     # 

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
E_cell = 1.51184 - 1.5421e-3*T + 9.523e-5*T*ca.log(T) + 9.84e-8*T**2
V_actA = ca.asinh(i_el/(2* i_0A))*R*T/(alpha_A*F)
V_actC = ca.asinh(i_el/(2* i_0C))*R*T/(alpha_C*F)
ro = (0.005139*lambda_-0.00326)**(1268*(303**-1 - T**-1))
R_cell = delta/ro;

# Intermediate photovoltaic variables
Vt = K*T_ps/Q
Iph = (Isc+Kl*(T-Tr))*Irradiation(time)
Irs = Ior*(T_ps/Tr)** 3*ca.exp(Q*Ego*(1/Tr-1/T_ps)/(K*A))

# Algebraic equations
f_h2 = N_c*i_el/F 
v_el = (E_cell + V_actC + V_actA + i_el*R_cell)
v_ps = 100
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 

# Lagrange cost function
f_q = (N_c**2*(v_el*i_el) - v_ps*i_ps)**2

# Diferential equations
m_h2_dot = f_h2 - HydrogenDemand(time)

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

F = ca.Function('F', [X0, U, T], [X, Q], ['x0', 'p', 't'], ['xf', 'qf'])

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
    ubw += [1000]
    w0 += [1]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk, t=k*dt)
    Xk = Fk['xf']
    J = J + Fk['qf']

    # Add inequality constraint: x1 is bound to be between 0 and infinity
    g += [Xk[0]]
    lbg += [1]
    ubg += [15]

# Solve the NLP
# Creat NPL Solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}

# Use IPOPT as the NLP solver
solver = ca.nlpsol('solver', 'ipopt', prob)

# Call the solver
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

# Print the optimal cost
print('Optimal cost: ' + str(sol['f']))

# Retrieve the solution
w_opt = sol['x'].full().flatten()

# Plot the solution
import matplotlib.pyplot as plt
plt.figure()
plt.step(range(N), w_opt, '--')
plt.xlabel('Time')
plt.ylabel('Control')
plt.grid()
plt.show()

# Simulating the system with the solution
Xs = ca.vertcat(0.5)
m = []

for s in range(N):
    Fs = F(x0=Xs, p=w_opt[s], t=s*dt)
    Xs = Fs['xf'] 
    m.append(Xs.full().flatten()[0])

# Plot the simulation
plt.figure()
plt.step(range(N), m, '--')
plt.xlabel('Time')
plt.ylabel('State')
plt.grid()
plt.show()