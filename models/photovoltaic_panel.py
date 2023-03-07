import casadi as ca
from models.input_data import Irradiation

def lambertw(x):
    """ 
    Lambert W function analytic approximation
    """    
    E = 0.4586887
    return (1+E)*ca.log(6/5*x/ca.log(12/5*x/ca.log(1+12/5*x)) ) -E*ca.log(2*x/ca.log(1+2*x))

def pv_model(time):
    """
    Photovoltaic panel model
    """    
    
    # Declare constants
    Q = 1.6e-19 # Elementary charge
    K = 1.38e-23 # Boltzmann constant

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

    # Intermediate photovoltaic variables
    Vt = K*T_ps/Q
    Iph = (Isc+Kl*(T_ps-Tr))*Irradiation(time)
    Irs = Ior*(T_ps/Tr)** 3*ca.exp(Q*Ego*(1/Tr-1/T_ps)/(K*A))

    # Algebraic equations
    v_ps = (N_ss*Vt*A*(lambertw(ca.exp(1)*(Iph/Irs+1))-1))
    i_ps = N_ps*(Iph-Irs*(ca.exp(v_ps/(N_ss*Vt))-1)) 

    return [i_ps, v_ps]