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

    def __init__(self, initial: float, final: float, nGrid: int):
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
        self.nGrid = nGrid
        self.tGrid = np.linspace(initial, final, num=nGrid, endpoint=True)
        self.dt = (final - initial)/nGrid

class Variable:
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

class VariableList:
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

class Phase:
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
        self.npl = NonlinearProgrammingProblem()
        self.phases = []

    def set_phases(self, phases: list):
        for i in range(len(phases)):
            phases[i].set_model_function(self.buil_model_function(phases[i].model))
        self.phases = phases
        
    def buil_model_function(self, dynamic):
        if isinstance(dynamic, list):
            dynamic = ca.hcat(dynamic)

        return ca.Function(
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
            self.phases[0].modelFunc,
            self.langrange_cost,
            self.controls,
            self.states,
            self.time,
            self.guess)
        
        [cost, traj_opt] = self.npl.solve()

        self.solution = type('solution', (object,), {})()
        #self.solution.t = np.linspace(0, self.time.final, num=2*self.time.nGrid-1, endpoint=True)
        self.solution.cost = cost
        self.solution.traj = OptimalTrajectoryList()
        
        t_opt = traj_opt[2].full().flatten()
        t_aux = []
        for i in range(self.time.nGrid-1):
            t_mid = (t_opt[i+1]-t_opt[i])/2
            t_aux += [t_opt[i]]
            t_aux += [t_opt[i] + t_mid]
        t_aux += [t_opt[-1]]
        self.solution.t_opt = t_aux

        for i in range(len(self.states)):
            x_opt = traj_opt[i].full().flatten()
            fx = interpolate.interp1d(self.solution.t_opt, x_opt, kind=3)
            self.solution.traj.add(OptimalTrajectory(self.states[i].name, x_opt, fx))

        for i in range(len(self.controls)):
            u_opt = traj_opt[i+len(self.states)].full().flatten()
            fu = interpolate.interp1d(self.solution.t_opt, u_opt, kind=2)
            self.solution.traj.add(OptimalTrajectory(self.controls[i].name, u_opt, fu))

    def get_optimization_status(self) -> str:
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
            for i in range(0, len(self.solution.t_opt)-2, 2):
                t += np.linspace(self.solution.t_opt[i], self.solution.t_opt[i+1], 5, endpoint=False).tolist()
                t += np.linspace(self.solution.t_opt[i+1], self.solution.t_opt[i+2], 5, endpoint=False).tolist()
            t += [self.time.final]

        f_interpolated = interpolate.interp1d(self.solution.t_opt, self.phases[0].modelFunc(self.solution.traj[0].values, self.solution.traj[1].values, self.solution.t_opt).full().flatten(), kind=2)
        
        # Error in differential equations
        self.solution.diff_error = self.phases[0].modelFunc(self.solution.traj[0].f(t), self.solution.traj[1].f(t), t) - f_interpolated(t)
        self.solution.t_error = t

    def plot_error(self):
        fig2 = plt.figure(2)
        fig2.suptitle('Erro in differential equations')
        plt.plot(self.solution.t_error, self.solution.diff_error, self.solution.t_opt, np.zeros(len(self.solution.t_opt)), '.r')
        plt.ylabel('Error')
        plt.xlabel('Time [h]')
        plt.grid(axis='both',linestyle='-.')
        
        plt.show()
        #plt.savefig(files.get_plot_error_file_name(__file__), bbox_inches='tight', dpi=300)

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
        self.aux.T = []
        self.sym = type('sym', (object,), {})()
        self.sym.X = []
        self.sym.U = []
        self.sym.T = []

        for k in range(nTime):
            self.sym.T.append(ca.MX.sym('T_' + str(k)))

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

    def add_variable_u(self, vList, minlist, maxList, guessList):
        for i in range(len(vList)):
            self.x += [vList[i]]
            self.lbx += [minlist[i]]
            self.ubx += [maxList[i]]
            self.x0 += [guessList[i]]
            self.aux.U[i] += [vList[i]]

    def add_variable_x(self, vList, minlist, maxList, guessList):
        for i in range(len(vList)):
            self.x += [vList[i]]
            self.lbx += [minlist[i]]
            self.ubx += [maxList[i]]
            self.x0 += [guessList[i]]
            self.aux.X[i] += [vList[i]]
    
    def add_variable_t(self, v, min, max, guess):
        self.x += [v]
        self.lbx += [min]
        self.ubx += [max]
        self.x0 += [guess]
        self.aux.T += [v]
    
    def build_npl(self, F: ca.Function, L: ca.Function, controls: VariableList, states: VariableList, time: Time, guess):
        self.create_variables(len(controls), len(states), time.nGrid)

        for k in np.arange(0, 2*time.nGrid - 2, 2):
            i = int(k/2)
            dt = self.sym.T[i+1]-self.sym.T[i]

            # Defects
            f_k_0 = F(self.sym.X[k][0], ca.hcat(self.sym.U[k]), self.sym.T[i])
            f_k_1 = F(self.sym.X[k+1][0], ca.hcat(self.sym.U[k+1]), self.sym.T[i] + dt/2)
            f_k_2 = F(self.sym.X[k+2][0], ca.hcat(self.sym.U[k+2]), self.sym.T[i+1])

            g = self.sym.X[k+2][0] - self.sym.X[k][0] - dt*(f_k_0 + 4*f_k_1 + f_k_2)/6
            self.add_constraint(g, 0, 0)
            
            g = self.sym.X[k+1][0] - (self.sym.X[k+2][0] + self.sym.X[k][0])/2 - dt*(f_k_0 - f_k_2)/8
            self.add_constraint(g, 0, 0)

            # Langrange cost
            w_k_0 = L(self.sym.X[k][0], ca.hcat(self.sym.U[k]), self.sym.T[i])
            w_k_1 = L(self.sym.X[k+1][0], ca.hcat(self.sym.U[k+1]), self.sym.T[i] + dt/2)
            w_k_2 = L(self.sym.X[k+2][0], ca.hcat(self.sym.U[k+2]), self.sym.T[i+1])

            self.f += dt*(w_k_0 + 4*w_k_1 + w_k_2)/6

            # Time constraints
            g = self.sym.T[i+1] - self.sym.T[i]
            self.add_constraint(g, 0, ca.inf)

        # Add states and controls as optimization variables to the NLP
        for k in range(2*time.nGrid - 1):
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

        # Add time as optimization variable to the NLP
        for k in range(time.nGrid):
            self.add_variable_t(
                self.sym.T[k],
                time.initial,
                time.final,
                time.tGrid[k])
        
        # Set the initial condition for the time
        # Start time
        self.lbx[-time.nGrid] = time.initial
        self.ubx[-time.nGrid] = time.initial
        # Final time
        self.lbx[-1] = time.final
        self.ubx[-1] = time.final


        # Concatenate vectors
        self.x = ca.vertcat(*self.x)
        self.g = ca.vertcat(*self.g)

        for i in range(len(self.aux.X)):
            self.aux.X[i] = ca.horzcat(*self.aux.X[i])

        for i in range(len(self.aux.U)):
            self.aux.U[i] = ca.horzcat(*self.aux.U[i])
        
        self.aux.T = ca.horzcat(*self.aux.T)

        # Creat NPL Solver
        prob = {'f': self.f, 'x': self.x, 'g': self.g}

        # NLP solver options
        #self.__ipopt_log_file = 'results/'+self.name+'.txt'
        self.__ipopt_log_file = 'log.txt'
        opts = {"ipopt.output_file" : self.__ipopt_log_file}

        # Use IPOPT as the NLP solver
        self.solver = ca.nlpsol('solver', 'ipopt', prob, opts)
    
    def solve(self) -> list:
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
        
        aux_input.append(self.aux.T)
        aux_output.append('t')

        trajectories = ca.Function(
            'trajectories', 
            [self.x], 
            aux_input, 
            ['w'], 
            aux_output)
        
        traj_opt = trajectories(self.__sol['x'])
        
        return [self.__sol['f'], traj_opt]
