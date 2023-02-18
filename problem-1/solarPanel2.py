#
# Description: This script simulates the behavior of a solar panel withouth calculating the voltage.
#   The voltage is varied from 0 to 175 and the irradiation is fixed at 1.
#

from math import exp
from scipy.special import lambertw
import matplotlib.pyplot as plt

# Model of solar panel
def solar_panel_model(lambda_ps: float, v_ps: float):
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
    i_ps = N_ps*(Iph-Irs*(exp(v_ps/V_t/N_ss/A)-1));
    
    return i_ps


# Simulation of the solar panel
voltage = range(0, 175, 1)
current = []
power = []
for i in voltage:
    i_ps = solar_panel_model(1, voltage[i])
    current.append(i_ps)
    power.append(i_ps*voltage[i])


# Plot results
fig, ax1 = plt.subplots()
plt.title('Solar panel simulation')
plt.grid(axis='both')

ax2 = ax1.twinx()
ax1.plot(voltage, current, 'g-')
ax2.plot(voltage, power, 'b-')

ax1.set_xlabel('Voltage')
ax1.set_ylabel('Current', color='g')
ax2.set_ylabel('Power', color='b')

plt.show()







