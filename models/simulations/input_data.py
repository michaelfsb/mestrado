import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from models.input_data import Irradiation, HydrogenDemand

# Plot Irradiation and HydrogenDemand
t = np.arange(0, 1440, 1)

total_demand = trapezoid(HydrogenDemand(t).full().flatten(), dx=1)

f1 = plt.figure(1)
plt.plot(t/60, Irradiation(t), label="Irradiation")
plt.xlabel("Time (h)")
plt.ylabel("Irradiation")

f2 = plt.figure(2)
plt.plot(t/60, HydrogenDemand(t), label="HydrogenDemand")
plt.suptitle("Total Hydrogen Demand: " + "{:4f}".format(total_demand) + " Nm3")
plt.xlabel("Time (h)")
plt.ylabel("HydrogenDemand (Nm3/min)")

plt.show()