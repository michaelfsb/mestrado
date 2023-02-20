#
# Description: This script simulates the behavior of an PEM electrolyzer.
#
# https://doi.org/10.1016/j.jclepro.2020.121184
#

from math import asinh, log
from numpy import arange
import matplotlib.pyplot as plt

def electrolyzer(i_el: float):
    # Declare constants
    R = 8.314 # Gas constant
    F = 96485 # Faraday constant

    # Declare parameters
    N_c = 1          # Number of cells
    T = 298          # Temperature cell
    delta = 100e-6   # Thickness of membrane (tirei do google)
    lambda_ = 17     # degree of humidification of the membrane
    alpha_A = 2      # 
    i_0A = 1e-10     # 
    alpha_C = .5     # 
    i_0C = 1e-10     # 

    # Intermediate variables
    E_cell = 1.51184 - 1.5421e-3*T + 9.523e-5*T*log(T) + 9.84e-8*T**2
    V_actA = asinh(i_el/(2* i_0A))*R*T/(alpha_A*F)
    V_actC = asinh(i_el/(2* i_0C))*R*T/(alpha_C*F)
    ro = (0.005139*lambda_-0.00326)**(1268*(303**-1 - T**-1))
    R_cell = delta/ro;

    # Algebraic equations
    v_el = (N_c*(E_cell + V_actC + V_actA + i_el*R_cell))
    return v_el

# Simulation of the electrolyzer
current = arange(0, 10, 0.1)
voltage = []
power = []
for i in range(len(current)):
    v_el = electrolyzer(current[i])
    voltage.append(v_el)
    power.append(v_el*current[i])

# Plot results
fig, ax1 = plt.subplots()
plt.grid(axis='both',linestyle='-.')
# plt.xlim(0, 180)
fig.set_figwidth(7)

ax2 = ax1.twinx()
ax1.plot(current, voltage, 'g-',  label='Current')
ax2.plot(current, power, 'b-',  label='Power')

ax1.set_xlabel('Current')
ax1.set_ylabel('Voltage')
ax2.set_ylabel('Power')

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles,labels, loc=[0.03, 0.72]) 

# ax1.set_ylim([0, 29])
# ax2.set_ylim([0, 3600])

fig.tight_layout()

plt.show()

# Generate a Latex plot of the results (cant be used with plt.show() and plt.legend() functions)
# import tikzplotlib
# tikzplotlib.save("solarPanel2.tex")
