# Imports
from ocp.opdas import OptimalControlProblem, Time, VariableList, Phase
from models.electrolyzer import electrolyzer_model
from models.input_data import HydrogenDemand, Irradiation
from models.photovoltaic_panel import pv_model
from models.tank import thank_model

t = Time(initial=0, final=1440, nGridPerPhase=20)

states = VariableList()
states.add(name='v_h2', min=0.6, max=2.5)

controls = VariableList()
controls.add(name='i_el', min=0, max=100)

problem = OptimalControlProblem(name='ocp-1', controls=controls, states=states, time=t)

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(controls['i_el'])
[i_ps, v_ps, p_ps] = pv_model(Irradiation(t.value)) 
v_h2_dot = thank_model(f_h2, HydrogenDemand(t.value)) 

phase_1 = Phase(name='on', model=v_h2_dot, cost=(p_el - p_ps)**2)
problem.set_phases([phase_1])

problem.set_guess(control=[30], state=[2.5])

problem.solve()

problem.plot_solution()

problem.evaluate_error()

problem.plot_error()