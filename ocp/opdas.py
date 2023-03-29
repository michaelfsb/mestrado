import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

from utils import files

class Time():
    def __init__(self, initial: int, final: int, nGrid: int):
        self.name = 'time'
        self.value = ca.MX.sym(self.name)
        self.initial = initial
        self.final = final
        self.nGrid = nGrid
        self.dt = (final - initial)/nGrid

class Variable:
    def __init__(self, name, max, min):
        self.name = name
        self.value = ca.MX.sym(self.name)
        self.max = max
        self.min = min

class VariableList:
    def __init__(self):
        self.variables = []

    def __len__(self):
        return len(self.variables)
    
    def __getitem__(self, key):
        return self.get_variable(key).value
    
    def add_variable(self, name, max, min):
        variable = Variable(name, max, min)
        self.variables.append(variable)

    def remove_variable(self, name):
        for variable in self.variables:
            if variable.name == name:
                self.variables.remove(variable)
                return True
        return False

    def get_variable(self, name):
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def get_all_variables(self):
        return self.variables
    
    def get_all_values(self):
        values = []
        for variable in self.variables:
            values.append(variable.value)
        return values
    
    def get_all_min_values(self):
        min_values = []
        for variable in self.variables:
            min_values.append(variable.min)
        return min_values
    
    def get_all_max_values(self):
        max_values = []
        for variable in self.variables:
            max_values.append(variable.max)
        return max_values

class OptimalControlProblem():
    def __init__(self, name: str, controls: VariableList, states: VariableList, time: Time):
        self.name = name
        self.controls = controls
        self.states = states
        self.time = time
        self.tGrid = np.linspace(time.initial, time.final, num=time.nGrid, endpoint=True)
        
    def set_dynamic(self, dynamic):
        self.dynamic = ca.Function('F', [ca.hcat(self.states.get_all_values()), ca.hcat(self.controls.get_all_values()), self.time.value], [dynamic], ['x', 'u', 't'], ['x_dot'])
    
    def set_langrange_cost(self, l_cost):
        self.langrange_cost = ca.Function('L', [ca.hcat(self.states.get_all_values()), ca.hcat(self.controls.get_all_values()), self.time.value], [l_cost], ['x', 'u', 't'], ['L'])

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
            self.__X += [ca.MX.sym('X_' + str(k), len(self.states))]
            self.__U += [ca.MX.sym('U_' + str(k), len(self.controls))]

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
            self.__npl.lbx += self.controls.get_all_min_values()
            self.__npl.ubx += self.controls.get_all_max_values()
            self.__npl.x0 += [self.guess.controls]

            self.__npl.x   += [self.__X[k]]
            self.__npl.lbx += self.states.get_all_min_values()
            self.__npl.ubx += self.states.get_all_max_values()
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
        self.__ipopt_log_file = 'results/'+self.name+'.txt'
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
    
    def get_optimization_status(self):
        optimzation_status = ''
        with open(self.__ipopt_log_file) as file:
            for line in file:
                if line.startswith('EXIT'):
                    optimzation_status = line.strip()[5:-1]
        return optimzation_status

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

    def evaluate_error(self, t=None):
        if t is None:
            t = []
            for i in range(0, len(self.solution.t)-2, 2):
                t += np.linspace(self.solution.t[i], self.solution.t[i+1], 10, endpoint=False).tolist()
                t += np.linspace(self.solution.t[i+1], self.solution.t[i+2], 10, endpoint=False).tolist()
            t += [self.time.final]

        f_interpolated = interpolate.interp1d(self.solution.t, self.dynamic(self.solution.x, self.solution.u, self.solution.t).full().flatten(), kind=2)
        error = self.dynamic(self.solution.f_x(t), self.solution.f_u(t), t) - f_interpolated(t)

        # Plot error
        fig2 = plt.figure(2)
        fig2.suptitle('Erro in differential equations')
        plt.plot(t, error, self.solution.t, np.zeros(len(self.solution.t)), '.r')
        plt.ylabel('Error')
        plt.xlabel('Time [h]')
        plt.grid(axis='both',linestyle='-.')
        #plt.show()
        plt.savefig(files.get_plot_error_file_name(__file__), bbox_inches='tight', dpi=300)
