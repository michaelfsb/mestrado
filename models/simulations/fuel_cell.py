from matplotlib import pyplot as plt
from models.fuel_cell import fuel_cell_model

# Simulation of the fuel cell
current = range(0, 70, 1)
voltage = []
power = []
h2_consumption = []
for i in range(len(current)):
    [f_h2, v_fc, p_fc] = fuel_cell_model(current[i])
    voltage.append(v_fc)
    power.append(v_fc*current[i])
    h2_consumption.append(f_h2)

# Plot the results
f1 = plt.figure(1)
plt.plot(current, h2_consumption, label="H2 consumption")
plt.xlabel("Current (A)")
plt.ylabel("H2 consumption (Nl/min)")

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
# tikzplotlib.save("models/simulations/fuel_cell.tex")
