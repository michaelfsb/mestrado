# Imports
import casadi as ca
from scipy.special import lambertw


# Define Lambert W function
def ca_lambertw(x, max_iter=100):
    """Approximate the Lambert W function using Newton's method"""
    w = ca.log(x) - ca.log(ca.log(x)) # Initial approximation using log
    for i in range(max_iter):
        ew = ca.exp(w)
        wew = w * ew
        f = wew - x
        df = (w + 2) * wew / (2 * w + 2)
        w_next = w - f / df
        w = w_next
    return w

for i in range(200):
    x = ca.DM([i])
    print("x: ", x)
    print("casadi: ", ca_lambertw(x))
    print("scipy: ", lambertw(i))