# Imports
import casadi as ca

# Declare constants
R = 8.314 # Gas constant
F = 96485 # Faraday constant
Q = 1.6e-19 # Elementary charge
K = 1.38e-23 # Boltzmann constant

# Declare electrolyzer parameters
A_el = 212.5        # Stack area
N_el = 1            # Number of cells
P_h2 = 6.9          # Hydrogen partial pressure
P_o2 = 1.3          # Oxygen partial pressure
I_ao = 1.0631e-6    # Anode current density 
I_co = 1e-3         # Cathode current density
delta_b = 178.5     # Membrane thickness
lambda_b = 21       # Membrana water content
C_el = 0.5          # Thermal capacitance
t_el = 298          # Temperature
t_ab = 298          # Temperature 
tau_el = 0          # NÃO ACHEI NO ARTIGO !!!
P_el = 0            # NÃO ACHEI NO ARTIGO !!!
R_I = 0             # NÃO ACHEI NO ARTIGO !!!

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
d_h2 = ca.SX.sym('d_h2') # Algebric - Hydrogen demand
i_ps = ca.SX.sym('i_ps') # Algebric - Electrical current in photovoltaic panel
v_ps = ca.SX.sym('v_ps') # Algebric - Voltage of photovoltaic panel
v_el = ca.SX.sym('v_el') # Algebric - Voltage of electrolyzer
i_el = ca.SX.sym('i_el') # Control - Electrical current in electrolyzer
x = ca.vertcat(m_h2)                    # State vector
z = ca.vertcat(f_h2, v_el, v_ps, i_ps)  # Algebraic vector
u = ca.vertcat(i_el)                    # Control vector

# Intermediate electrolyzer variables
v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*ca.log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
ro_b = (0.005139*lambda_b - 0.00326) * ca.exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
v_etd = (R*t_el/F)*ca.sinh(.5*i_el/(I_ao))**-1 + (R*t_el/F)*ca.sinh(.5*i_el/(I_co))**-1 + i_el*delta_b/ro_b + R_I*i_el # Eletrode overpotential
v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

# Intermediate photovoltaic variables
V_t = K*T_ps/Q;
Iph = (Isc+Kl*(T-Tr))*lambda_;
Irs = Ior*(T_ps/Tr)** 3*ca.exp(Q*Ego*(1/Tr-1/T_ps)/(K*A));	

# Diferential equations
m_h2_dot = f_h2 - d_h2
f_x = ca.vertcat(m_h2_dot)

# Algebraic equations
f_h2 = N_el*i_el/F # Produced hydrogen in the electrolyzer
v_el = v_el_0 + v_etd + v_el_hom_ion # Voltage of electrolyzer
v_ps = 0 # Voltage of photovoltaic panel
i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) # Current in photovoltaic panel
f_z = ca.vertcat(f_h2, v_el)

# Lagrange cost function
f_q = (v_el*i_el - v_ps*i_ps)**2

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