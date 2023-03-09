import casadi as ca

def battery_model(i_bt, c_out):
    """ Battery model """
    
    A_bt = 11.053       # (V)
    B_bt = 2.452        # (Ah**-1)
    C_120 = 367         # (Ah)
    K_bt = 0.006215     # (V)
    R = 0               # Internal resistance (cant find it on paper!)
    V_bt_0 = 51.58      # (V)

    i_bt_f = i_bt # battery current filtered by a first-order filter (filter need to be implemented!)

    # Charging 
    v_bt_int_c = V_bt_0 - K_bt*i_bt_f*(C_120/(C_120 - c_out)) - K_bt*c_out*(C_120/(C_120 - c_out)) + A_bt*ca.exp(-B_bt*c_out)

    # Discharging
    v_bt_int_d = V_bt_0 - K_bt*i_bt_f*(C_120/(C_120 - 0.1*c_out)) - K_bt*c_out*(C_120/(C_120 - c_out)) + A_bt*ca.exp(-B_bt*c_out)

    v_bt = ca.if_else(i_bt>=0, v_bt_int_c,  v_bt_int_d) - R*i_bt
    p_bt = v_bt*i_bt
    return [v_bt, p_bt]