# Imports
import casadi as ca
from scipy.special import lambertw


# Define Lambert W function
def ca_lambertw(x):
    E = 0.4586887;
    return (1+E)*ca.log(6/5*x/ca.log(12/5*x/ca.log(1+12/5*x)) ) -E*ca.log(2*x/ca.log(1+2*x))

for i in range(200):
    x = ca.DM([i])
    print("x: ", x)
    print("casadi: ", ca_lambertw(x))
    print("scipy: ", lambertw(i))