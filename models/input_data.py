import casadi as ca
import numpy as np
from scipy.signal import butter, lfilter
from scipy.io import loadmat

# Read irradiation and demand data from file
mat_contents = loadmat('models/vetores_sol_carga.mat')

Tf = 1440 # Final time (min)
ini = Tf
fim = 2*Tf # Take second day

sol_real = mat_contents['sol_real'][ini:fim]
fdemanda = (mat_contents['carga_real'][ini:fim]-0.18)*(1/60) # (Nm3/min)
# t_file = np.arange(1,len(sol_real)+1,1)

mat_contents = loadmat('models/vetores_sol_carga_filtrado.mat')
sol_real = mat_contents['lambda']
#fdemanda = (mat_contents['fdemanda'])*(1/60)
t_file = np.arange(1,len(sol_real)+1,1)
            
# Create interpolants
Irradiation = ca.interpolant("Irradiation", "bspline", [t_file], sol_real.flatten()) # Normalized irradiation
HydrogenDemand = ca.interpolant("HydrogenDemand", "bspline", [t_file], fdemanda.flatten()) # (Nm3/min)