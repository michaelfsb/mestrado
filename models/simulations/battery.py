import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from matplotlib import pyplot as plt

from models.battery import battery_model

# Build signal for the battery discharge current
Tf = 12*60      # Final time (min)
A = 25          # Amplitude (A)
duty = 1/6      # Duty cycle
freq = 1/45     # Frequency (Hz)
dt = 0.1        # Time step (min)

t = np.arange(0, Tf, dt) # Time (min)
current = A*(1-signal.square(2*np.pi*freq*t, duty=duty))/2

# Simulate the battery
voltage = []
power = []
c_out = []
for i in range(len(current)):
    c_out.append(trapezoid(current[:i], dx=dt)/60)
    [v_bt, p_bt] = battery_model(-current[i], c_out[i])
    voltage.append(v_bt.full().flatten())
    power.append(p_bt.full().flatten())

#Plot the results
f1 = plt.figure(1)
plt.plot(t/60, current, label="Current")
plt.xlabel("Time (h)")
plt.ylabel("Current (A)")

f2 = plt.figure(2)
plt.plot(t/60, voltage, label="Voltage")
plt.xlabel("Time (h)")
plt.ylabel("Voltage (V)")
plt.ylim(0, 70)

f3 = plt.figure(3)
plt.plot(t/60, c_out, label="C_out")
plt.xlabel("Time (h)")
plt.ylabel("C_out (Ah)")

plt.show()

# Generate a Latex plot of the results (cant be used with plt.show() and plt.legend() functions)
#import tikzplotlib
#tikzplotlib.save("models/simulations/battery.tex")

