import casadi as ca

def fuel_cell_model(i_fc):
    """ Fuel cell model """	

    T_0 = 296   # Nominal operating temperature (K)
    A_fc = 61   # Stack area (cm2)
    P_H2 = 1.25 # H2 partial pressure (atm)

    # P_O2 need to be a state variable
    # t_fc nedd to be defined if will be fixed or variable 

    i_fc_d = i_fc/A_fc # Current density
    v_fc_0 = 1.046 + 0.003*(t_fc - T_0) + 0.244*(.5*ca.log(P_O2) + ca.log(P_H2))# Open circuit voltage
    v_act = 0.066*(1 - ca.exp(-i_fc_d/0.013)) # Activation losses
    v_ohm = 0.299*i_fc # Ohmic losses
    v_conc = 0.028*i_fc_d**9.001 #

    v_fc = v_fc_0 - v_act - v_ohm - v_conc
    p_fc = v_fc*i_fc
    return [v_fc, p_fc]