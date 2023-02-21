# Imports
import casadi as ca
import numpy as np
from math import asinh, exp, log
from scipy.special import lambertw
from scipy.io import loadmat

# Read irradiation and demand data from file
mat_contents = loadmat('vetores_sol_carga.mat')

ini = 24*60
fim = 2*24*60

sol_real = mat_contents['sol_real'][ini:fim]
fdemanda = -0.18+mat_contents['carga_real'][ini:fim]
t_file = np.arange(1,len(sol_real)+1,1)

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
m_h2 = ca.SX.sym('m_h2') # State - Mass of hydrogen
f_h2 = ca.SX.sym('f_h2') # Algebric - Produced hydrogen in the electrolyzer
i_ps = ca.SX.sym('i_ps') # Algebric - Electrical current in photovoltaic panel
v_ps = ca.SX.sym('v_ps') # Algebric - Voltage of photovoltaic panel
v_el = ca.SX.sym('v_el') # Algebric - Voltage of electrolyzer
i_el = ca.SX.sym('i_el') # Control - Electrical current in electrolyzer
x = ca.vertcat(m_h2)                    # State vector
z = ca.vertcat(f_h2, v_el, v_ps, i_ps)  # Algebraic vector
u = ca.vertcat(i_el)                    # Control vector

# Intermediate electrolyzer variables
E_cell = 1.51184 - 1.5421e-3*T + 9.523e-5*T*log(T) + 9.84e-8*T**2
V_actA = asinh(i_el/(2* i_0A))*R*T/(alpha_A*F)
V_actC = asinh(i_el/(2* i_0C))*R*T/(alpha_C*F)
ro = (0.005139*lambda_-0.00326)**(1268*(303**-1 - T**-1))
R_cell = delta/ro;

# Intermediate photovoltaic variables
Vt = K*T_ps/Q
Iph = (Isc+Kl*(T-Tr))*Irradiation(time)
Irs = Ior*(T_ps/Tr)** 3*ca.exp(Q*Ego*(1/Tr-1/T_ps)/(K*A))

# Diferential equations
m_h2_dot = f_h2 - HydrogenDemand(time)
f_x = ca.vertcat(m_h2_dot)

# Algebraic equations
f_h2 = N_c*i_el/F 
v_el = (E_cell + V_actC + V_actA + i_el*R_cell)
v_ps = (N_ss*Vt*A*(lambertw(exp(1)*(Iph/Irs+1))-1)).real
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 
f_z = ca.vertcat(f_h2, v_el, v_ps, i_ps)

# Lagrange cost function
f_q = (N_c**2*(v_el*i_el) - v_ps*i_ps)**2

# Create an integrator
dae = {'x':x, 'z':z, 'p':u, 'ode':f_x, 'alg':f_z, 'quad':f_q}
opts = {'tf':0.5} # interval length
I = ca.integrator('I', 'idas', dae, opts)

# Number of intervals
nk = 20

# Start with an empty NLP
w = []   # List of variables
lbw = [] # Lower bounds on w
ubw = [] # Upper bounds on w
G = []   # Constraints
J = 0    # Cost function