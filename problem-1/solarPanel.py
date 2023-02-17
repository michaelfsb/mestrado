from math import exp
from scipy.special import lambertw
import matplotlib.pyplot as plt

# Model of solar panel
def solar_panel_model(lambda_ps: float):
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
    V_t = K*T_ps/Q;
    Iph = (Isc+Kl*(T_ps-Tr))*lambda_ps;
    Irs = Ior*(T_ps/Tr)**3*exp(Q*Ego*(1/Tr-1/T_ps)/(K*A));	

    # Equation of the solar panel
    v_ps = (N_ss*V_t*A*(lambertw(exp(1)*(Iph/Irs+1))-1)).real;
    i_ps = N_ps*(Iph-Irs*(exp(v_ps/V_t/N_ss/A)-1));
    
    return [i_ps, v_ps]


# Simulation of the solar panel
irradiation = range(0, 100, 1)
current = []
voltage = []
power = []
for i in irradiation:
    [i_ps, v_ps] = solar_panel_model(i/100)
    current.append(i_ps)
    voltage.append(v_ps)
    power.append(i_ps*v_ps)

# Plot of the simulation
plt.figure(1)
plt.title('Solar panel simulation')
plt.xlabel('Voltage')
plt.ylabel('Current')
plt.grid()
plt.plot(voltage, current)


plt.figure(2)
plt.title('Solar panel simulation')
plt.xlabel('Voltage')
plt.ylabel('Power')
plt.grid()
plt.plot(voltage, power)
plt.show()






