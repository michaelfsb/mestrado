import casadi as ca
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

class Time():
    """ 
    Rrepresents the time domain for a OptimalControlProblem.

    :param name: A string representing the name of the time domain.
    :param value: A MX symbol representing the value of the time domain.
    :param initial: An integer representing the initial time value of the time domain.
    :param final: An integer representing the final time value of the time domain.
    :param nGrid: An integer representing the number of points in the time domain.
    :param dt: A numeric value representing the time step size.
    """

    def __init__(self, initial: float, final: float, nGridPerPhase: int):
        """
        Creates a new Time object with the given bounds.

        :param initial: the initial time value of the time domain.
        :type initial: float
        :param final: the final time value of the time domain.
        :type final: float
        :param nGrid: the number of points in the time domain.
        :type nGrid: int
        """
                
        self.name = 'time'
        self.value = ca.MX.sym(self.name)
        self.initial = initial
        self.final = final
        self.nGrid = 0
        self.nGridPerPhase = nGridPerPhase
        self.tGrid = []
        self.dt = (final - initial)/nGridPerPhase

class Variable():
    def __init__(self, name, max, min):
        """
        Creates a new Variable object with the given name and bounds.

        :param name: the name of the variable.
        :type name: str
        :param max: the maximum allowed value of the variable.
        :type max: float
        :param min: the minimum allowed value of the variable.
        :type min: float
        """

        self.name = name
        self.value = ca.MX.sym(self.name)
        self.max = max
        self.min = min

class VariableList():
    def __init__(self):
        """
        Creates an empty list of variables.
        """

        self.variables = []

    def __len__(self) -> int:
        return len(self.variables)
    
    def __getitem__(self, key) -> Variable:
        if isinstance(key, int):
            return self.variables[key]
        else:
            return self.get(key).value
    
    def add(self, name, max, min):
        """
        Adds a new variable to the list with the given name and bounds.

        :param name: the name of the variable.
        :type name: str
        :param max: the maximum allowed value of the variable.
        :type max: float
        :param min: the minimum allowed value of the variable.
        :type min: float
        """
            
        variable = Variable(name, max, min)
        self.variables.append(variable)

    def remove(self, name):
        """
        Removes the variable with the given name from the list.

        :param name: the name of the variable to remove.
        :type name: str
        :return: True if a variable with the given name was found and removed, False otherwise.
        :rtype: bool
        """

        for variable in self.variables:
            if variable.name == name:
                self.variables.remove(variable)
                return True
        return False

    def get(self, name) -> Variable:
        """
        Gets the variable with the given name from the list.

        :param name: the name of the variable to get.
        :type name: str
        :return: the Variable object with the given name, or None if not found.
        :rtype: Variable or None
        """

        for variable in self.variables:
            if variable.name == name:
                return variable
        return None

    def get_all(self) -> list:
        """
        Gets a list of all the variables in the list.

        :return: a list of all the Variable objects in the list.
        :rtype: list of Variable
        """

        return self.variables
    
    def get_all_values(self) -> list:
        """
        Gets a list of the values of all the variables in the list.

        :return: a list of the values of all the variables in the list.
        :rtype: list of casadi.MX
        """

        values = []
        for variable in self.variables:
            values.append(variable.value)
        return values
    
    def get_all_min_values(self) -> list:
        """
        Gets a list of the minimum bounds of all the variables in the list.

        :return: a list of the minimum bounds of all the variables in the list.
        :rtype: list of float
        """
            
        min_values = []
        for variable in self.variables:
            min_values.append(variable.min)
        return min_values
    
    def get_all_max_values(self) -> list:
        """
        Gets a list of the maximum bounds of all the variables in the list.

        :return: a list of the maximum bounds of all the variables in the list.
        :rtype: list of float
        """
                
        max_values = []
        for variable in self.variables:
            max_values.append(variable.max)
        return max_values

class Phase():
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.modelFunc = []
    
    def set_model_function(self, func):
        self.modelFunc = func

class OptimalTrajectory():
    def __init__(self, name: str, values, f):
        self.name = name
        self.values = values
        self.f = f

class OptimalTrajectoryList():
    """
    The OptimalTrajectoryList class represents a list of optimal trajectories. 
    Each trajectory is an instance of the OptimalTrajectory class and is identified by a unique name. 
    The class provides the ability to add new trajectories to the list and retrieve trajectories by either their index or name.
    """

    def __init__(self):
        """
        Initializes an empty OptimalTrajectoryList object.
        """

        self.trajectories = []
    
    def __getitem__(self, key) -> OptimalTrajectory:
        """
        Returns the trajectory at the given index or with the given name.

        Args:
            key: int or str
                Index of the trajectory or name of the trajectory.

        Returns:
            OptimalTrajectory or None
                Returns the OptimalTrajectory object at the given index or with the given name.
                Returns None if no trajectory is found.
        """

        if isinstance(key, int):
            return self.trajectories[key]
        else:
            return self.get(key)
        
    def add(self, trajectory):
        """
        Adds a new OptimalTrajectory object to the list.

        Args:
            trajectory: OptimalTrajectory
                OptimalTrajectory object to add to the list.
        """

        self.trajectories.append(trajectory)
        
    def get(self, name) -> OptimalTrajectory:
        """
        Returns the trajectory with the given name.

        Args:
            name: str
                Name of the trajectory.

        Returns:
            OptimalTrajectory or None
                Returns the OptimalTrajectory object with the given name.
                Returns None if no trajectory is found.
        """

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
        self.phases = []
        self.npl = type('npl', (object,), {})()

    def set_phases(self, phases: list):
        for i in range(len(phases)):
            phases[i].set_model_function(self.__buil_model_function(phases[i].model, phases[i].name))
        self.phases = phases
        self.time.nGrid = len(self.phases)*self.time.nGridPerPhase
        
        
    def __buil_model_function(self, dynamic, phase_name: str):
        if isinstance(dynamic, list):
            dynamic = ca.hcat(dynamic)

        return ca.Function(
            'F_' + phase_name, 
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
    
    def set_guess(self, control, state):
        self.guess = type('guess', (object,), {})()
        self.guess.controls = control
        self.guess.states = state

    def solve(self):
        self.create_npl()
        self.build_npl()
        [cost, traj_opt] = self.solve_npl()
        self.get_solution(cost, traj_opt)

    def create_npl(self):
        self.npl.f = 0
        self.npl.g = []
        self.npl.x = []
        self.npl.x0 = []
        self.npl.lbg = []
        self.npl.ubg = []
        self.npl.lbx = []
        self.npl.ubx = []
        self.npl.sym = type('sym', (object,), {})()
        self.npl.sym.X = []
        self.npl.sym.U = []
        self.npl.sym.T = []
        self.npl.aux = type('aux', (object,), {})()
        self.npl.aux.X = []
        self.npl.aux.U = []
        self.npl.aux.T = []

        for i in range(len(self.phases)):
            for k in np.arange(0, self.time.nGridPerPhase-.5, .5):
                variable_X = []
                for j in range(len(self.states)):
                    variable_X += [ca.MX.sym('X_' + str(j) + '_' + str(i) + '_' + str(k))]
                self.npl.sym.X += [variable_X]

                variable_U = []
                for j in range(len(self.controls)):
                    variable_U += [ca.MX.sym('U_' + str(j) + '_' + str(i) + '_' + str(k))]
                self.npl.sym.U += [variable_U]
        
            for k in range(self.time.nGridPerPhase):
                self.npl.sym.T.append(ca.MX.sym('T_' + str(i) + '_' + str(k)))
        
        for k in range(len(self.states)):
            self.npl.aux.X.append([])

        for k in range(len(self.controls)):
            self.npl.aux.U.append([])
        
    def build_npl(self):
        tInitial = self.time.initial
        for i in range(len(self.phases)):
            ini = i*(2*self.time.nGridPerPhase - 1)
            end = 2*(i+1)*self.time.nGridPerPhase - 1
            iniT = i*self.time.nGridPerPhase
            endT = (i+1)*self.time.nGridPerPhase
            self.hermit_simpson_collocation(
                self.phases[i].modelFunc, 
                self.langrange_cost,
                self.npl.sym.X[ini:end],
                self.npl.sym.U[ini:end],
                self.npl.sym.T[iniT:endT])
            
            # Set time constraints to assegure that each instant of time is greater than the previous one
            tFinal = (i+1)*self.time.final/len(self.phases)
            self.time.tGrid = np.concatenate((self.time.tGrid, np.linspace(tInitial, tFinal, num=self.time.nGridPerPhase, endpoint=True)), axis=None)
            tInitial = tFinal
            self.set_time_constraints(i)
        
        # Conect the phases
        for i in range(len(self.phases) - 1):
            # Conect the time between phases
            j = (i+1)*self.time.nGridPerPhase - 1
            g = self.npl.sym.T[j] - self.npl.sym.T[j+1]
            self.add_constraint(g, 0, 0)

            # Conect the states between phases
            k = (i+1)*(2*self.time.nGridPerPhase - 1) - 1
            g = self.npl.sym.X[k][0] - self.npl.sym.X[k+1][0] # TO DO - Corrigir para tratar quando tem mais de um estado
            self.add_constraint(g, 0, 0)


        # Add states, controls, and time as optimization variables to the NLP
        self.set_optimization_variables()
            
        # Set the initial condition for the state
        self.npl.lbx[len(self.npl.sym.U)] = self.guess.states[0]
        self.npl.ubx[len(self.npl.sym.U)] = self.guess.states[0]

        # Set the initial condition for the time
        # Start time
        self.npl.lbx[-self.time.nGrid] = self.time.initial
        self.npl.ubx[-self.time.nGrid] = self.time.initial
        # Final time
        self.npl.lbx[-1] = self.time.final
        self.npl.ubx[-1] = self.time.final

        # Concatenate vectors
        self.npl.x = ca.vertcat(*self.npl.x)
        self.npl.g = ca.vertcat(*self.npl.g)

        for i in range(len(self.npl.aux.X)):
            self.npl.aux.X[i] = ca.horzcat(*self.npl.aux.X[i])

        for i in range(len(self.npl.aux.U)):
            self.npl.aux.U[i] = ca.horzcat(*self.npl.aux.U[i])
        
        self.npl.aux.T = ca.horzcat(*self.npl.aux.T)

        # Creat NPL Solver
        prob = {'f': self.npl.f, 'x': self.npl.x, 'g': self.npl.g}

        # NLP solver options
        #self.__ipopt_log_file = 'results/'+self.name+'.txt'
        self.__ipopt_log_file = 'log.txt'
        opts = {"ipopt.output_file" : self.__ipopt_log_file}

        # Use IPOPT as the NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)
    
    def hermit_simpson_collocation(self, F: ca.Function, L: ca.Function, X: ca.SX, U: ca.SX, T: ca.SX):
        for k in np.arange(0, 2*self.time.nGridPerPhase - 2, 2):
            i = int(k/2)
            dt = T[i+1]-T[i]

            # Defects
            f_k_0 = F(X[k][0], ca.hcat(U[k]), T[i])
            f_k_1 = F(X[k+1][0], ca.hcat(U[k+1]), T[i] + dt/2)
            f_k_2 = F(X[k+2][0], ca.hcat(U[k+2]), T[i+1])

            g = X[k+2][0] - X[k][0] - dt*(f_k_0 + 4*f_k_1 + f_k_2)/6
            self.add_constraint(g, 0, 0)
            
            g = X[k+1][0] - (X[k+2][0] + X[k][0])/2 - dt*(f_k_0 - f_k_2)/8
            self.add_constraint(g, 0, 0)

            # Langrange cost
            w_k_0 = L(X[k][0], ca.hcat(U[k]), T[i])
            w_k_1 = L(X[k+1][0], ca.hcat(U[k+1]), T[i] + dt/2)
            w_k_2 = L(X[k+2][0], ca.hcat(U[k+2]), T[i+1])

            self.npl.f += dt*(w_k_0 + 4*w_k_1 + w_k_2)/6
    
    def add_constraint(self, g, lbg, ubg):
        self.npl.g += [g]
        self.npl.lbg += [lbg]
        self.npl.ubg += [ubg]

    def add_variable_u(self, vList, minlist, maxList, guessList):
        for i in range(len(vList)):
            self.npl.x += [vList[i]]
            self.npl.lbx += [minlist[i]]
            self.npl.ubx += [maxList[i]]
            self.npl.x0 += [guessList[i]]
            self.npl.aux.U[i] += [vList[i]]

    def add_variable_x(self, vList, minlist, maxList, guessList):
        for i in range(len(vList)):
            self.npl.x += [vList[i]]
            self.npl.lbx += [minlist[i]]
            self.npl.ubx += [maxList[i]]
            self.npl.x0 += [guessList[i]]
            self.npl.aux.X[i] += [vList[i]]
    
    def add_variable_t(self, v, min, max, guess):
        self.npl.x += [v]
        self.npl.lbx += [min]
        self.npl.ubx += [max]
        self.npl.x0 += [guess]
        self.npl.aux.T += [v]

    def set_optimization_variables(self):
        for k in range(len(self.npl.sym.U)):
            self.add_variable_u(
                self.npl.sym.U[k], 
                self.controls.get_all_min_values(), 
                self.controls.get_all_max_values(), 
                self.guess.controls)

        for k in range(len(self.npl.sym.X)):
            self.add_variable_x(
                self.npl.sym.X[k],
                self.states.get_all_min_values(),
                self.states.get_all_max_values(),
                self.guess.states)
        
        for k in range(len(self.npl.sym.T)):
            self.add_variable_t(
                self.npl.sym.T[k],
                self.time.initial,
                self.time.final,
                self.time.tGrid[k])

    def set_time_constraints(self, nPhase : int):
        for i in range(self.time.nGridPerPhase-1):
            k = nPhase*self.time.nGridPerPhase + i
            g = self.npl.sym.T[k+1] - self.npl.sym.T[k]
            self.add_constraint(g, 0, ca.inf)
    
    def solve_npl(self) -> list:
        aux_debgug = self.solver(
            x0=self.npl.x0, 
            lbx=self.npl.lbx, 
            ubx=self.npl.ubx, 
            lbg=self.npl.lbg, 
            ubg=self.npl.ubg)
        
        self.__sol = aux_debgug
        
        aux_input = []
        aux_output = []
        for i in range(len(self.npl.aux.X)):
            aux_input.append(self.npl.aux.X[i])
            aux_output.append('x_'+str(i))
        for i in range(len(self.npl.aux.U)):
            aux_input.append(self.npl.aux.U[i])
            aux_output.append('u_'+str(i))
        
        aux_input.append(self.npl.aux.T)
        aux_output.append('t')

        trajectories = ca.Function(
            'trajectories', 
            [self.npl.x], 
            aux_input, 
            ['w'], 
            aux_output)
        
        traj_opt = trajectories(self.__sol['x'])
        
        return [self.__sol['f'], traj_opt]   

    def get_solution(self, cost, traj_opt):
        self.solution = type('solution', (object,), {})()
        self.solution.cost = cost
        self.solution.traj = OptimalTrajectoryList()
        self.get_solution_time(traj_opt)
        self.get_solution_variables(traj_opt)

    def get_solution_time(self, traj_opt):
        t_opt = traj_opt[2].full().flatten()
        self.solution.t = []
        for i in range(len(self.phases)):
            for j in range(self.time.nGridPerPhase-1):
                k = i*self.time.nGridPerPhase + j
                self.solution.t += [t_opt[k]]
                self.solution.t += [t_opt[k] + (t_opt[k+1]-t_opt[k])/2]
            self.solution.t += [t_opt[(i+1)*self.time.nGridPerPhase-1]]
    
    def get_solution_variables(self, traj_opt):
        for i in range(len(self.states)):
            x_opt = traj_opt[i].full().flatten()
            #fx = interpolate.interp1d(self.solution.t, x_opt, kind=3)
            fx = 0
            self.solution.traj.add(OptimalTrajectory(self.states[i].name, x_opt, fx))

        for i in range(len(self.controls)):
            u_opt = traj_opt[i+len(self.states)].full().flatten()
            #fu = interpolate.interp1d(self.solution.t, u_opt, kind=2)
            fu = 0
            self.solution.traj.add(OptimalTrajectory(self.controls[i].name, u_opt, fu))

####################################################################################################   

    def plot_solution(self, t_plot=None):
        if t_plot is None:
            t_plot = np.linspace(0, self.time.final, num=10*self.time.nGrid, endpoint=True)

        fig, axs = plt.subplots(2,1)
        #fig.suptitle('Simulation Results: ' + optimzation_status + '\nCost: ' + str(self.solution.cost))
        fig.suptitle('Simulation Results \nCost: ' + str(self.solution.cost))
        #axs[0].plot(t_plot/60, self.solution.traj['i_el'].f(t_plot), '-b')
        axs[0].plot(self.solution.t, self.solution.traj['i_el'].values, '-b')
        axs[0].plot(self.solution.t, self.solution.traj['i_el'].values, '.b')
        axs[0].set_ylabel('Electrolyzer current [A]')
        axs[0].grid(axis='both',linestyle='-.')
        #axs[0].set_xticks(np.arange(0, 26, 2))

        #axs[1].plot(t_plot/60, self.solution.traj['v_h2'].f(t_plot), '-g')
        axs[1].plot(self.solution.t, self.solution.traj['v_h2'].values, '-g')
        axs[1].plot(self.solution.t, self.solution.traj['v_h2'].values, '.g')
        axs[1].set_ylabel('Hydrogen [Nm3]')
        axs[1].set_xlabel('Time [h]')
        axs[1].grid(axis='both',linestyle='-.')
        #axs[1].set_xticks(np.arange(0, 26, 2))

        plt.show()
        #plt.savefig(files.get_plot_file_name(__file__), bbox_inches='tight', dpi=300)

    def evaluate_error(self, t=None, plot=False):
        if t is None:
            t = []
            for i in range(0, len(self.solution.t)-2, 2):
                t += np.linspace(self.solution.t[i], self.solution.t[i+1], 5, endpoint=False).tolist()
                t += np.linspace(self.solution.t[i+1], self.solution.t[i+2], 5, endpoint=False).tolist()
            t += [self.time.final]

        f_interpolated = interpolate.interp1d(self.solution.t, self.phases[0].modelFunc(self.solution.traj[0].values, self.solution.traj[1].values, self.solution.t).full().flatten(), kind=2)
        
        # Error in differential equations
        self.solution.diff_error = self.phases[0].modelFunc(self.solution.traj[0].f(t), self.solution.traj[1].f(t), t) - f_interpolated(t)
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

