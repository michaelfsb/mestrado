import numpy as np

def quadratic_spline(t, x, ts):
    """ Quadratic spline interpolation
    args: t - vector of length 2, x - vector of length 3, ts - vector of length n
    returns: vector of length n with the quadratic spline interpolation of x at ts """

    if len(t) != 2:
        raise Exception('t must be a vector of length 2')
    if len(x) != 3:
        raise Exception('x must be a vector of length 3')
    if any(i < t[0] for i in ts) or any(i > t[1] for i in ts):
        raise Exception('ts must be between t[0] and t[1]')

    h = t[1] - t[0]

    delta = ts - t[0]

    beta_1 = -(3*x[0] - 4*x[1] + x[2])/h
    beta_2 = 2*(x[0] - 2*x[1] + x[2])/(h**2)

    return [x[0] + beta_1*delta[i] + beta_2*delta[i]**2 for i in range(len(ts))]


def cubic_spline(t, x, ts):
    """ Cubic spline interpolation
    args: t - vector of length 2, x - vector of length 3, ts - vector of length n
    returns: vector of length n with the cubic spline interpolation of x at ts """

    if len(t) != 2:
        raise Exception('t must be a vector of length 2')
    if len(x) != 3:
        raise Exception('x must be a vector of length 3')
    if any(i < t[0] for i in ts) or any(i > t[1] for i in ts):
        raise Exception('ts must be between t[0] and t[1]')

    h = t[1] - t[0]

    delta = ts - t[0]

    gamma_2 = -(3*x[0] - 4*x[1] + x[2])/(2*h)
    gamma_3 = 2*(x[0] - 2*x[1] + x[2])/(3*h**2)

    return [x[0] + delta*x[0] + gamma_2*delta[i]**2 + gamma_3*delta[i]**3 for i in range(len(ts))]

def control_spline(t, x):
    n = len(t)
    ts = []
    xs = []
    h = (t[1] - t[0])/2
    
    for i in range(0, n-1, 1):
        j = 2*i
        ts += np.linspace(t[i], t[i]+h, 5, endpoint=False).tolist()
        ts += np.linspace(t[i]+h, t[i+1], 5, endpoint=False).tolist()
        xs += quadratic_spline(t[i:i+2], x[j:j+3], ts[10*i:10*i+10])
    
    return [ts, xs]

def state_spline(t, x):
    n = len(t)
    ts = []
    xs = []
    h = (t[1] - t[0])/2
    
    for i in range(0, n-1, 1):
        j = 2*i
        ts += np.linspace(t[i], t[i]+h, 5, endpoint=False).tolist()
        ts += np.linspace(t[i]+h, t[i+1], 5, endpoint=False).tolist()
        xs += cubic_spline(t[i:i+2], x[j:j+3], ts[10*i:10*i+10])
    
    return [ts, xs]