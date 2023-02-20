#
# Description: This script simulates the behavior of an PEM electrolyzer.
#
# https://doi.org/10.1016/j.ijhydene.2013.06.113
#

from math import exp, sinh, log
import numpy as np

# Model of PEM electrolyzer
def electrolyzer_model(i_el: float):

    # Declare constants
    R = 8.314 # Gas constant
    F = 96485 # Faraday constant

    # Declare electrolyzer parameters
    A_el = 212.5        # Stack area
    N_el = 1            # Number of cells
    P_h2 = 6.9          # Hydrogen partial pressure
    P_o2 = 1.3          # Oxygen partial pressure
    I_ao = 1.0631**-6   # Anode current density 
    I_co = 1**-3        # Cathode current density
    delta_b = 178e-6    # Membrane thickness
    lambda_b = 21       # Membrana water content
    t_el = 298          # Temperature
    R_I = 0             # Can`t find this parameter in article!

    # Intermediate electrolyzer variables
    ro_b = (0.005139*lambda_b - 0.00326) * exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
    v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
    v_etd = (R*t_el/F)*sinh(.5*i_el/I_ao)**-1 + (R*t_el/F)*sinh(.5*i_el/I_co)**-1 + i_el*delta_b/ro_b + R_I*i_el # Eletrode overpotential
    v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

    # Algebraic equations
    v_el = v_el_0 + v_etd + v_el_hom_ion # Voltage of electrolyzer

    return v_el

# Simulation of the electrolyzer
current = np.arange(1, 60, 1)
voltage = []
for i in range(len(current)):
    print(i)
    v_el = electrolyzer_model(current[i])
    voltage.append(v_el)

# Plot results
import matplotlib.pyplot as plt
plt.plot(current, voltage, 'b-',  label='Voltage')
plt.xlabel('Current')
plt.ylabel('Voltage')
plt.show()
