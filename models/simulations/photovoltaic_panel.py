#
# Description: This script simulates the behavior of a solar panel withouth calculating the voltage.
#   The voltage is varied from 0 to 177 and the irradiation is fixed at 1.
#

import matplotlib.pyplot as plt
from models.photovoltaic_panel import pv_model

# Simulation of the solar panel
voltage = range(0, 177, 1)
current = []
power = []
Irradiation = 1
for i in voltage:
    [i_ps, v_ps, p_ps] = pv_model(Irradiation, voltage[i])
    current.append(i_ps)
    power.append(p_ps)

# Plot results
fig, ax1 = plt.subplots()
plt.grid(axis='both',linestyle='-.')
plt.xlim(0, 180)
fig.set_figwidth(7)

ax2 = ax1.twinx()
ax1.plot(voltage, current, 'g-',  label='Current')
ax2.plot(voltage, power, 'b-',  label='Power')

ax1.set_xlabel('Voltage')
ax1.set_ylabel('Current')
ax2.set_ylabel('Power')

handles,labels = [],[]
for ax in fig.axes:
    for h,l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)

plt.legend(handles,labels, loc=[0.03, 0.72]) 

ax1.set_ylim([0, 29])
ax2.set_ylim([0, 3600])

fig.tight_layout()

plt.show()

# Generate a Latex plot of the results (cant be used with plt.show() and plt.legend() functions)
# import tikzplotlib
# tikzplotlib.save("models/simulations/photovoltaic_panel.tex")
