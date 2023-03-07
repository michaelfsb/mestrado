import casadi as ca

def lambertw(x):
    """ 
    Lambert W function analytic approximation
    x >= 3e-5 -> error < 0.2%

    Args:
        x (float | ca.DM | ca.SX | ca.MX )
        

    Returns:
        (float | ca.DM | ca.SX | ca.MX )
    """    
    E = 0.4586887
    return (1+E)*ca.log(6/5*x/ca.log(12/5*x/ca.log(1+12/5*x)) ) -E*ca.log(2*x/ca.log(1+2*x))