import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from models.electrolyzer import electrolyzer_model
from models.input_data import HydrogenDemand, Irradiation
from models.photovoltaic_panel import pv_model

from models.tank import thank_model
from utils import files

class time():
    def __init__(self, initial: int, final: int, nGrid: int):
        self.name = 'time'
        self.value = ca.MX.sym(self.name)
        self.initial = initial
        self.final = final
        self.nGrid = nGrid
        self.dt = (final - initial)/nGrid

class control():
    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.value = ca.MX.sym(name)
        self.min = min
        self.max = max
    
class state():
    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.value = ca.MX.sym(name)
        self.min = min
        self.max = max

class ocp():
    def __init__(self, controls: control, states: state, time: time):
        self.controls = controls
        self.states = states
        self.time = time
        self.tGrid = np.linspace(time.initial, time.final, num=time.nGrid, endpoint=True)

    def set_dynamic(self, dynamic):
        self.dynamic = ca.Function('F', [self.states.value, self.controls.value, self.time.value], [dynamic], ['x', 'u', 't'], ['x_dot'])
    
    def set_langrange_cost(self, l_cost):
        self.langrange_cost = ca.Function('L', [self.states.value, self.controls.value, self.time.value], [l_cost], ['x', 'u', 't'], ['L'])

    def set_guess(self, control, state):
        self.guess = type('guess', (object,), {})()
        self.guess.controls = control
        self.guess.states = state

    def __create_internal_variables(self):
        self.__npl = type('npl', (object,), {})()
        self.__npl.f = 0
        self.__npl.g = []
        self.__npl.x = []
        self.__npl.x0 = []
        self.__npl.lbg = []
        self.__npl.ubg = []
        self.__npl.lbx = []
        self.__npl.ubx = []
        self.__plot = type('plot', (object,), {})()
        self.__plot.x = []
        self.__plot.u = []
        self.__X = []
        self.__U = []
        for k in np.arange(0, self.time.nGrid-.5, .5):
            self.__X += [ca.MX.sym('X_' + str(k))]
            self.__U += [ca.MX.sym('U_' + str(k))]

    def __build_npl(self):
        self.__create_internal_variables()

        for k in np.arange(0, 2*self.time.nGrid-2, 2):
            i = int(k/2)

            # Defects
            f_k_0 = self.dynamic(self.__X[k], self.__U[k], self.tGrid[i])
            f_k_1 = self.dynamic(self.__X[k+1], self.__U[k+1], self.tGrid[i]+self.time.dt/2)
            f_k_2 = self.dynamic(self.__X[k+2], self.__U[k+2], self.tGrid[i+1])
            
            self.__npl.g += [self.__X[k+2] - self.__X[k] - self.time.dt*(f_k_0 + 4*f_k_1 + f_k_2)/6]
            self.__npl.lbg += [0]
            self.__npl.ubg += [0]
            self.__npl.g += [self.__X[k+1] - (self.__X[k+2] + self.__X[k])/2 - self.time.dt*(f_k_0 - f_k_2)/8]
            self.__npl.lbg += [0]
            self.__npl.ubg += [0]

            # Langrange cost
            w_k_0 = self.langrange_cost(self.__X[k], self.__U[k], self.tGrid[i])
            w_k_1 = self.langrange_cost(self.__X[k+1], self.__U[k+1], self.tGrid[i]+self.time.dt/2)
            w_k_2 = self.langrange_cost(self.__X[k+2], self.__U[k+2], self.tGrid[i+1])

            self.__npl.f += self.time.dt*(w_k_0 + 4*w_k_1 + w_k_2)/6

        for k in range(2*self.time.nGrid-1):
            self.__npl.x += [self.__U[k]]
            self.__npl.lbx += [self.controls.min]
            self.__npl.ubx += [self.controls.max]
            self.__npl.x0 += [self.guess.controls]

            self.__npl.x   += [self.__X[k]]
            self.__npl.lbx += [self.states.min]
            self.__npl.ubx += [self.states.max]
            self.__npl.x0  += [self.guess.states]

            self.__plot.x += [self.__X[k]]
            self.__plot.u += [self.__U[k]]

        # Set the initial condition for the state
        self.__npl.lbx[1] = self.guess.states
        self.__npl.ubx[1] = self.guess.states

        # Concatenate vectors
        self.__npl.x = ca.vertcat(*self.__npl.x)
        self.__npl.g = ca.vertcat(*self.__npl.g)
        self.__plot.x = ca.horzcat(*self.__plot.x)
        self.__plot.u = ca.horzcat(*self.__plot.u)

        # Creat NPL Solver
        prob = {'f': self.__npl.f, 'x': self.__npl.x, 'g': self.__npl.g}

        # NLP solver options
        self.__ipopt_log_file = files.get_log_file_name(__file__)
        opts = {"ipopt.output_file" : self.__ipopt_log_file}

        # Use IPOPT as the NLP solver
        self.__npl.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    
    def solve(self):
        self.__build_npl()

        self.__sol = self.__npl.solver(
            x0=self.__npl.x0, 
            lbx=self.__npl.lbx, 
            ubx=self.__npl.ubx, 
            lbg=self.__npl.lbg, 
            ubg=self.__npl.ubg)
        
        # Retrieve the solution
        trajectories = ca.Function(
            'trajectories', 
            [self.__npl.x], 
            [self.__plot.x, self.__plot.u], 
            ['w'], 
            ['x', 'u'])
        x_opt, u_opt = trajectories(self.__sol['x'])
        self.solution = type('solution', (object,), {})()
        self.solution.x = x_opt.full().flatten()
        self.solution.u = u_opt.full().flatten()
        self.solution.t = np.linspace(0, self.time.final, num=2*self.time.nGrid-1, endpoint=True)
        self.solution.f_x = interpolate.interp1d(self.solution.t, self.solution.x, kind=3)
        self.solution.f_u = interpolate.interp1d(self.solution.t, self.solution.u, kind=2) 
        
        print('Optimal cost: ' + str(self.__sol['f']))

    def plot_solution(self, t=None):
        if t is None:
            t = np.linspace(0, self.time.final, num=10*self.time.nGrid, endpoint=True)

        fig, axs = plt.subplots(2,1)
        #fig.suptitle('Simulation Results: ' + optimzation_status + '\nCost: ' + str(self.__sol['f']))

        axs[0].plot(t/60, self.solution.f_u(t), '-b')
        axs[0].set_ylabel('Electrolyzer current [A]')
        axs[0].grid(axis='both',linestyle='-.')
        #axs[0].set_xticks(np.arange(0, 26, 2))

        axs[1].plot(t/60, self.solution.f_x(t), '-g')
        axs[1].set_ylabel('Hydrogen [Nm3]')
        axs[1].set_xlabel('Time [h]')
        axs[1].grid(axis='both',linestyle='-.')
        #axs[1].set_xticks(np.arange(0, 26, 2))

        plt.show()
        #plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)

   

# Declare variables
v_h2 = state(name='v_h2', min=0.6, max=2.5) 
i_el = control(name='i_el', min=1, max=100)        
t = time(initial=0, final=1440, nGrid=80)        

ocp = ocp(controls=i_el, states=v_h2, time=t)

# Models equations
[f_h2, v_el, p_el] = electrolyzer_model(i_el.value) 
[i_ps, v_ps, p_ps] = pv_model(Irradiation(t.value)) 
v_h2_dot = thank_model(f_h2, HydrogenDemand(t.value)) 

# Lagrange cost function
f_l = (p_el - p_ps)**2

ocp.set_dynamic(dynamic=v_h2_dot)
ocp.set_langrange_cost(l_cost=f_l)

ocp.set_guess(control=30, state=0.65)

ocp.solve()

ocp.plot_solution()
