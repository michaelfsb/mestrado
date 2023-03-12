import casadi as ca
from utils.convesions import h2_flow_converter

def electrolyzer_model(i_el):
    """ Eletrolyzer model """

    # Declare constants
    R = 8.314 # Gas constant
    F = 96485.33289 # Faraday constant
    
    # Declare electrolyzer parameters
    A_el = 212.5            # Stack area
    N_el = 6                # Number of cells
    P_h2 = 6.9              # Hydrogen partial pressure
    P_o2 = 1.3              # Oxygen partial pressure
    I_ao = 1.0631e-6        # Anode current density 
    I_co = 1e-3             # Cathode current density
    delta_b = 178e-6        # Membrane thickness
    lambda_b = 21           # Membrana water content
    t_el = 298              # Temperature	

    # Intermediate electrolyzer variables
    i_el_d = i_el/A_el # Current density
    ro_b = (0.005139*lambda_b - 0.00326) * ca.exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
    v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*ca.log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
    v_etd = (R*t_el/F)*ca.asinh(.5*i_el_d/I_ao) + (R*t_el/F)*ca.asinh(.5*i_el_d/I_co) + i_el_d*delta_b/ro_b # Eletrode overpotential
    v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

    f_h2_prd = (N_el*i_el/(F*1000))*h2_flow_converter("Nl/min") # Hydrogen production rate
    v_el = N_el*(v_el_0 + v_etd + v_el_hom_ion) # Eletrolyzer voltage
    p_el = i_el*v_el # Eletrolyzer consumed power

    return [f_h2_prd, v_el, p_el]