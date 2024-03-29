from matplotlib import pyplot as plt
from models.electrolyzer import electrolyzer_model

# Simulation of the electrolyzer
current = range(1, 60, 1)
voltage = []
power = []
h2_production = []
for i in range(len(current)):
    [f_h2, v_el, p_el] = electrolyzer_model(current[i])
    voltage.append(v_el)
    power.append(v_el*current[i])
    h2_production.append(f_h2)

# Plot the results
f1 = plt.figure(1)
plt.plot(current, h2_production, label="H2 production")
plt.xlabel("Current (A)")
plt.ylabel("H2 production (Nm3/min)")

f2 = plt.figure(2)
plt.plot(current, voltage, label="Voltage")
plt.xlabel("Current (A)")
plt.ylabel("Voltage (V)")

f3 = plt.figure(3)
plt.plot(current, power, label="Power")
plt.xlabel("Current (A)")
plt.ylabel("Power (W)")

plt.show()

# Generate a Latex plot of the results (cant be used with plt.show() and plt.legend() functions)
# import tikzplotlib
# tikzplotlib.save("models/simulations/electrolyzer.tex")
