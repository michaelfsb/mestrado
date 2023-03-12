import casadi as ca

from utils.convesions import h2_flow_converter

def fuel_cell_model(i_fc):
    """ Fuel cell model """	

    # Declare constants
    F = 96485.33289 # Faraday constant

    # Declare fuel cell parameters
    N_fc = 60           # Number of cells
    T_0 = 296           # Nominal operating temperature (K)
    A_fc = 61           # Stack area (cm2)
    P_H2 = 1.25         # H2 partial pressure (atm)
    U_0 = 1.046         # Open circuit voltage (V)
    K_1T = 0.003        # Temperature coefficient (V/K)
    K_1_p = 0.244       # Pressure coefficient (V/atm)s
    K_1_act = 0.066     # Activation losses coefficient (V/A)
    K_2_act = 0.013     # Activation losses coefficient (A)
    R_fc = 0.299        # Resistance (ohm)
    K_1_conc = 0.028    # Concentration coefficient (V/A)
    K_2_conc = 9.001    # Concentration coefficient (V/A)
    t_fc = 296          # Temperature
    
    P_O2 = (0.09/12.10)*i_fc + 0.15 # Oxygen partial pressure (atm)
    i_fc_d = i_fc/A_fc # Current density
    v_fc_0 = U_0 + K_1T*(t_fc - T_0) + K_1_p*(.5*ca.log(P_O2) + ca.log(P_H2)) # Open circuit voltage
    v_act = K_1_act*(1 - ca.exp(-i_fc_d/K_2_act)) # Activation losses
    v_ohm = R_fc*i_fc_d # Ohmic losses
    v_conc = K_1_conc*i_fc_d**K_2_conc # Concentration losses

    f_h2_cons = (N_fc*i_fc/(F*1000))*h2_flow_converter("Nl/min") # Hydrogen consumption rate
    v_fc = N_fc*(v_fc_0 - v_act - v_ohm - v_conc) 
    p_fc = v_fc*i_fc
    return [f_h2_cons, v_fc, p_fc]