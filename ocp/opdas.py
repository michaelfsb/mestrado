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
        if isinstance(key, int):
            return self.variables[key]
        else:
            return self.get(key).value
    
    def add(self, name, max, min):
        variable = Variable(name, max, min)
        self.variables.append(variable)

    def remove(self, name):
        for variable in self.variables:
            if variable.name == name:
                self.variables.remove(variable)
                return True
        return False

    def get(self, name):
        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def get_all(self):
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
    
class NonlinearProgrammingProblem():
    def __init__(self, option=None):
        self.option = option
        self.f = 0
        self.g = []
        self.x = []
        self.x0 = []
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []
    
    def create_variables(self, nControls, nStates, nTime):
        self.aux = type('aux', (object,), {})()
        self.aux.X = []
        self.aux.U = []
        self.sym = type('sym', (object,), {})()
        self.sym.X = []
        self.sym.U = []

        for k in range(nStates):
            self.aux.X.append([])

        for k in range(nControls):
            self.aux.U.append([])

        for k in np.arange(0, nTime-.5, .5):
            variable_X = []
            for j in range(nStates):
                variable_X += [ca.MX.sym('X_' + str(j) + '_' + str(k))]
            self.sym.X += [variable_X]

            variable_U = []
            for j in range(nControls):
                variable_U += [ca.MX.sym('U_' + str(j) + '_' + str(k))]
            self.sym.U += [variable_U]
    
    def add_constraint(self, g, lbg, ubg):
        self.g += [g]
        self.lbg += [lbg]
        self.ubg += [ubg]

    def add_variable_u(self, xList, minlist, maxList, guessList):
        for i in range(len(xList)):
            self.x += [xList[i]]
            self.lbx += [minlist[i]]
            self.ubx += [maxList[i]]
            self.x0 += [guessList[i]]
            self.aux.U[i] += [xList[i]]

    def add_variable_x(self, xList, minlist, maxList, guessList):
        for i in range(len(xList)):
            self.x += [xList[i]]
            self.lbx += [minlist[i]]
            self.ubx += [maxList[i]]
            self.x0 += [guessList[i]]
            self.aux.X[i] += [xList[i]]

    def build_npl(self, f: ca.Function, l: ca.Function, controls: VariableList, states: VariableList, tGrid, guess):
        self.create_variables(len(controls), len(states), len(tGrid))

        for k in np.arange(0, 2*len(tGrid) - 2, 2):
            i = int(k/2)
            dt = tGrid[i+1]-tGrid[i]

            # Defects
            f_k_0 = f(self.sym.X[k][0], ca.hcat(self.sym.U[k]), tGrid[i])
            f_k_1 = f(self.sym.X[k+1][0], ca.hcat(self.sym.U[k+1]), tGrid[i] + dt/2)
            f_k_2 = f(self.sym.X[k+2][0], ca.hcat(self.sym.U[k+2]), tGrid[i+1])

            g = self.sym.X[k+2][0] - self.sym.X[k][0] - dt*(f_k_0 + 4*f_k_1 + f_k_2)/6
            self.add_constraint(g, 0, 0)
            
            g = self.sym.X[k+1][0] - (self.sym.X[k+2][0] + self.sym.X[k][0])/2 - dt*(f_k_0 - f_k_2)/8
            self.add_constraint(g, 0, 0)

            # Langrange cost
            w_k_0 = l(self.sym.X[k][0], ca.hcat(self.sym.U[k]), tGrid[i])
            w_k_1 = l(self.sym.X[k+1][0], ca.hcat(self.sym.U[k+1]), tGrid[i] + dt/2)
            w_k_2 = l(self.sym.X[k+2][0], ca.hcat(self.sym.U[k+2]), tGrid[i+1])

            self.f += dt*(w_k_0 + 4*w_k_1 + w_k_2)/6

        for k in range(2*len(tGrid) - 1):
            self.add_variable_u(
                self.sym.U[k], 
                controls.get_all_min_values(), 
                controls.get_all_max_values(), 
                guess.controls)

            self.add_variable_x(
                self.sym.X[k],
                states.get_all_min_values(),
                states.get_all_max_values(),
                guess.states)
            
        # Set the initial condition for the state
        self.lbx[len(controls)] = guess.states[0]
        self.ubx[len(controls)] = guess.states[0]

        # Concatenate vectors
        self.x = ca.vertcat(*self.x)
        self.g = ca.vertcat(*self.g)

        for i in range(len(self.aux.X)):
            self.aux.X[i] = ca.horzcat(*self.aux.X[i])

        for i in range(len(self.aux.U)):
            self.aux.U[i] = ca.horzcat(*self.aux.U[i])

        # Creat NPL Solver
        prob = {'f': self.f, 'x': self.x, 'g': self.g}

        # NLP solver options
        #self.__ipopt_log_file = 'results/'+self.name+'.txt'
        self.__ipopt_log_file = 'log.txt'
        opts = {"ipopt.output_file" : self.__ipopt_log_file}

        # Use IPOPT as the NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    
    def solve(self):
        self.__sol = self.solver(
            x0=self.x0, 
            lbx=self.lbx, 
            ubx=self.ubx, 
            lbg=self.lbg, 
            ubg=self.ubg)
        
        aux_input = []
        aux_output = []
        for i in range(len(self.aux.X)):
            aux_input.append(self.aux.X[i])
            aux_output.append('x_'+str(i))
        for i in range(len(self.aux.U)):
            aux_input.append(self.aux.U[i])
            aux_output.append('u_'+str(i))

        trajectories = ca.Function(
            'trajectories', 
            [self.x], 
            aux_input, 
            ['w'], 
            aux_output)
        
        traj_opt = trajectories(self.__sol['x'])
        
        return [self.__sol['f'], traj_opt]

class OptimalTrajectory():
    def __init__(self, name: str, values, f):
        self.name = name
        self.values = values
        self.f = f

class OptimalTrajectoryList():
    def __init__(self):
        self.trajectories = []
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.trajectories[key]
        else:
            return self.get(key)
        
    def add(self, trajectory):
        self.trajectories.append(trajectory)
        
    def get(self, name):
        for trajectory in self.trajectories:
            if trajectory.name == name:
                return trajectory
        return None

class OptimalControlProblem():
    def __init__(self, name: str, controls: VariableList, states: VariableList, time: Time):
        self.name = name
        self.controls = controls
        self.states = states
        self.time = time
        self.tGrid = np.linspace(time.initial, time.final, num=time.nGrid, endpoint=True)
        self.npl = NonlinearProgrammingProblem()
        
    def set_dynamic(self, dynamic):
        if isinstance(dynamic, list):
            dynamic = ca.hcat(dynamic)
            
        self.dynamic = ca.Function(
            'F', 
            [ca.hcat(self.states.get_all_values()), ca.hcat(self.controls.get_all_values()), self.time.value],
            [dynamic], 
            ['x', 'u', 't'], 
            ['x_dot'])
    
    def set_langrange_cost(self, l_cost):
        self.langrange_cost = ca.Function(
            'L', 
            [ca.hcat(self.states.get_all_values()), ca.hcat(self.controls.get_all_values()), self.time.value],
            [l_cost],
            ['x', 'u', 't'],
            ['L'])
    
    def set_mayer_cost(self, m_cost):
        self.mayer_cost = ca.Function('M', 
            [ca.hcat(self.states.get_all_values()), ca.hcat(self.controls.get_all_values()), self.time.value],
            [m_cost],
            ['x', 'u', 't'],
            ['M'])

    def set_guess(self, control, state):
        self.guess = type('guess', (object,), {})()
        self.guess.controls = control
        self.guess.states = state

    def solve(self):
        self.npl.build_npl(
            self.dynamic,
            self.langrange_cost,
            self.controls,
            self.states,
            self.tGrid,
            self.guess)
        
        [cost, traj_opt] = self.npl.solve()

        self.solution = type('solution', (object,), {})()
        self.solution.t = np.linspace(0, self.time.final, num=2*self.time.nGrid-1, endpoint=True)
        self.solution.cost = cost
        self.solution.traj = OptimalTrajectoryList()

        for i in range(len(self.states)):
            x_opt = traj_opt[i].full().flatten()
            fx = interpolate.interp1d(self.solution.t, x_opt, kind=3)
            self.solution.traj.add(OptimalTrajectory(self.states[i].name, x_opt, fx))

        for i in range(len(self.controls)):
            u_opt = traj_opt[i+len(self.states)].full().flatten()
            fu = interpolate.interp1d(self.solution.t, u_opt, kind=2)
            self.solution.traj.add(OptimalTrajectory(self.controls[i].name, u_opt, fu))

    def get_optimization_status(self):
        optimzation_status = ''
        with open(self.__ipopt_log_file) as file:
            for line in file:
                if line.startswith('EXIT'):
                    optimzation_status = line.strip()[5:-1]
        return optimzation_status

    def plot_solution(self, t_plot=None):
        if t_plot is None:
            t_plot = np.linspace(0, self.time.final, num=10*self.time.nGrid, endpoint=True)

        fig, axs = plt.subplots(2,1)
        #fig.suptitle('Simulation Results: ' + optimzation_status + '\nCost: ' + str(self.solution.cost))
        fig.suptitle('Simulation Results \nCost: ' + str(self.solution.cost))
        axs[0].plot(t_plot/60, self.solution.traj['i_el'].f(t_plot), '-b')
        axs[0].set_ylabel('Electrolyzer current [A]')
        axs[0].grid(axis='both',linestyle='-.')
        axs[0].set_xticks(np.arange(0, 26, 2))

        axs[1].plot(t_plot/60, self.solution.traj['v_h2'].f(t_plot), '-g')
        axs[1].set_ylabel('Hydrogen [Nm3]')
        axs[1].set_xlabel('Time [h]')
        axs[1].grid(axis='both',linestyle='-.')
        axs[1].set_xticks(np.arange(0, 26, 2))

        plt.show()
        #plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)

    def evaluate_error(self, t=None, plot=False):
        if t is None:
            t = []
            for i in range(0, len(self.solution.t)-2, 2):
                t += np.linspace(self.solution.t[i], self.solution.t[i+1], 5, endpoint=False).tolist()
                t += np.linspace(self.solution.t[i+1], self.solution.t[i+2], 5, endpoint=False).tolist()
            t += [self.time.final]

        f_interpolated = interpolate.interp1d(self.solution.t, self.dynamic(self.solution.traj[0].values, self.solution.traj[1].values, self.solution.t).full().flatten(), kind=2)
        
        # Error in differential equations
        self.solution.diff_error = self.dynamic(self.solution.traj[0].f(t), self.solution.traj[1].f(t), t) - f_interpolated(t)
        self.solution.t_error = t

    def plot_error(self):
        fig2 = plt.figure(2)
        fig2.suptitle('Erro in differential equations')
        plt.plot(self.solution.t_error, self.solution.diff_error, self.solution.t, np.zeros(len(self.solution.t)), '.r')
        plt.ylabel('Error')
        plt.xlabel('Time [h]')
        plt.grid(axis='both',linestyle='-.')
        
        plt.show()
        #plt.savefig(files.get_plot_error_file_name(__file__), bbox_inches='tight', dpi=300)