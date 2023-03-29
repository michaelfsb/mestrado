# Imports
from ocp.opdas import state, control, time, ocp
from models.electrolyzer import electrolyzer_model
from models.input_data import HydrogenDemand, Irradiation
from models.photovoltaic_panel import pv_model
from models.tank import thank_model

t = time(initial=0, final=1440, nGrid=100)
v_h2 = state(name='v_h2', min=0.6, max=2.5)
controls = [control(name='i_el', min=1, max=100), control(name='s_el', min=0, max=1)]
i_el = control(name='i_el', min=1, max=100)
s_el = control(name='s_el', min=0, max=1)

controls.get_values()

problem = ocp(name='ocp-3', controls=[i_el, s_el], states=v_h2, time=t)

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(i_el.value)
f_h2 = s_el.value*f_h2 # Switching between electrolyzer states

[i_ps, v_ps, p_ps] = pv_model(Irradiation(t.value)) 
v_h2_dot = thank_model(f_h2, HydrogenDemand(t.value)) 

# Lagrange cost function
f_l = (s_el.value*(20-i_el.value))*(p_el - p_ps)**2

problem.set_dynamic(dynamic=v_h2_dot)
problem.set_langrange_cost(l_cost=f_l)

problem.set_guess(control=[30, 1], state=0.65)

problem.solve()

problem.plot_solution()

problem.evaluate_error()