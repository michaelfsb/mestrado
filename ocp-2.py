# Imports
from ocp.opdas import OptimalControlProblem, Time, VariableList, Phase
from models.electrolyzer import electrolyzer_model
from models.input_data import HydrogenDemand, Irradiation
from models.photovoltaic_panel import pv_model
from models.tank import thank_model

t = Time(initial=0, final=1440, nGridPerPhase=10)

states = VariableList()
states.add(name='v_h2', min=0.6, max=2.5)

controls = VariableList()
controls.add(name='i_el', min=1, max=100)

problem = OptimalControlProblem(name='ocp-2', controls=controls, states=states, time=t)

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(controls['i_el'])
[i_ps, v_ps, p_ps] = pv_model(Irradiation(t.value)) 
v_h2_dot = thank_model(f_h2, HydrogenDemand(t.value)) 
phase_on = Phase(name='on', model=v_h2_dot)

v_h2_dot_off = thank_model(0, HydrogenDemand(t.value)) 
phase_off = Phase(name='off', model=v_h2_dot_off)

problem.set_phases([phase_off, phase_on])

# Lagrange cost function
f_l = (p_el - p_ps)**2

problem.set_langrange_cost(f_l)

problem.set_langrange_cost(f_l)

problem.set_guess(control=[30], state=[2.5])

problem.solve()

problem.plot_solution()

problem.evaluate_error()