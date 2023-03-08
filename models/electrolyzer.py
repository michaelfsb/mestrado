import casadi as ca

# Declare constants
R = 8.314 # Gas constant
F = 96485 # Faraday constant

# Declare electrolyzer parameters
A_el = 212.5            # Stack area
N_el = 22500            # Number of cells
P_h2 = 6.9              # Hydrogen partial pressure
P_o2 = 1.3              # Oxygen partial pressure
I_ao = 1.0631e-6        # Anode current density 
I_co = 1e-3             # Cathode current density
delta_b = 178e-6        # Membrane thickness
lambda_b = 21           # Membrana water content
t_el = 298              # Temperature

def electrolyzer_model(i_el):
    """ Eletrolyzer model """	

    # Intermediate electrolyzer variables
    i = i_el/A_el # Current density
    ro_b = (0.005139*lambda_b - 0.00326) * ca.exp(1268*(1/303 - 1/t_el)) # Membrane conductivity
    v_el_0 = 1.23 - 0.0009*(t_el-298) + 2.3*R*t_el*ca.log(P_h2**2*P_o2)/(4*F) # Reversible potential of the electrolyzer
    v_etd = (R*t_el/F)*ca.asinh(.5*i/I_ao) + (R*t_el/F)*ca.asinh(.5*i/I_co) + i*delta_b/ro_b # Eletrode overpotential
    v_el_hom_ion = delta_b*i_el/(A_el*ro_b) # Ohmic overvoltage and ionic overpotential

    f_h2 = (N_el*i_el/F)*(11.126/(60*1000)) # Hydrogen production rate (Nm3/min)
    v_el = v_el_0 + v_etd + v_el_hom_ion # Eletrolyzer voltage

    return [f_h2, v_el]