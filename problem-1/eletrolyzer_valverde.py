#
# Description: This script simulates the behavior of an PEM electrolyzer.
#
# https://doi.org/10.1016/j.ijhydene.2013.06.113
#

from math import exp, asinh, log
import numpy as np
import matplotlib.pyplot as plt

# Model of PEM electrolyzer
def electrolyzer_model(i_el: float):

    # Declare constants
    R = 8.314 # Gas constant
    F = 96485 # Faraday constant

    # Declare electrolyzer parameters
    A_el = 212.5            # Stack area
    N_el = 40               # Number of cells
    P_h2 = 6.9              # Hydrogen partial pressure
    P_o2 = 1.3              # Oxygen partial pressure
    I_ao = 1.0631*10**-6    # Anode current density 
    I_co = 1*10**-3         # Cathode current density
    delta_b = 178e-6        # Membrane thickness
    lambda_b = 21           # Membrana water content
    t_el = 298              # Temperature

    # Intermediate electrolyzer variables
    i = i_el/A_el # Current density
    ro_b = (0.005139*lambda_b - 0.00326) * exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
    v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
    v_etd = (R*t_el/F)*asinh(.5*i/I_ao) + (R*t_el/F)*asinh(.5*i/I_co) + i*delta_b/ro_b # Eletrode overpotential
    v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

    # Algebraic equations
    v_el = v_el_0 + v_etd + v_el_hom_ion # Voltage of electrolyzer

    return v_el

# Simulation of the electrolyzer
current = np.arange(1, 60, 1)
voltage = []
power = []
for i in range(len(current)):
    v_el = electrolyzer_model(current[i])
    voltage.append(v_el)
    power.append(v_el*40*current[i])

# Plot results
fig, ax1 = plt.subplots()
plt.grid(axis='both',linestyle='-.')
#plt.xlim(0, 180)
fig.set_figwidth(7)

ax2 = ax1.twinx()
ax1.plot(current, voltage, 'g-',  label='Current')
ax2.plot(current, power, 'b-',  label='Power')

ax1.set_xlabel('Voltage')
ax1.set_ylabel('Current')
ax2.set_ylabel('Power')

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

#plt.legend(handles,labels, loc=[0.03, 0.72]) 

#ax1.set_ylim([0, 29])
#ax2.set_ylim([0, 3600])

fig.tight_layout()

plt.show()