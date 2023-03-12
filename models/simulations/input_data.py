import numpy as np
import matplotlib.pyplot as plt

from models.input_data import Irradiation, HydrogenDemand

# Plot Irradiation and HydrogenDemand
t = np.arange(0, 1440, 1)

f1 = plt.figure(1)
plt.plot(t/60, Irradiation(t), label="Irradiation")
plt.xlabel("Time (h)")
plt.ylabel("Irradiation")

f2 = plt.figure(2)
plt.plot(t/60, HydrogenDemand(t), label="HydrogenDemand")
plt.xlabel("Time (h)")
plt.ylabel("HydrogenDemand (Nl/min)")

plt.show()