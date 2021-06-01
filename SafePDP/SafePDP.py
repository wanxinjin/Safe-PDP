import time

import casadi
import numpy as np
from casadi import *
import numpy
from SafePDP import PDP


# This class contains the following  utilities:
# 1) define an parameterized constraint optimal control system
# 2) provide a constraint optimal control solver
# 3) differentiate the Pontryagin maximum principle (differential PMP)
# 4) generate the auxiliary control system
# The variable styles used in this class is the same as the notations in the constraint PDP paper.
class COCsys:

    def __init__(self, project_name='my cpdp object'):
        self.project_name = project_name
        self.inf = 1e20

    # define the tunable parameter (if exists)
    def setAuxvarVariable(self, auxvar=None):
        if auxvar is None or auxvar.numel() == 0:
            self.auxvar = SX.sym('auxvar')
        else:
            self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()

    # define the state variable and its allowable upper and lower bounds
    def setStateVariable(self, state, state_lb=None, state_ub=None):
        self.state = state
        self.n_state = self.state.numel()

        if state_lb is not None:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-self.inf]

        if state_ub is not None:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [self.inf]

    # define the control input variable and its allowable upper and lower bounds
    def setControlVariable(self, control, control_lb=None, control_ub=None):
        self.control = control
        self.n_control = self.control.numel()

        if control_lb is not None:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-self.inf]

        if control_ub is not None:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [self.inf]

    # set the dynamics model
    def setDyn(self, ode):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control, self.auxvar], [self.dyn])

    # set the path cost
    def setPathCost(self, path_cost):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        assert path_cost.numel() == 1, "path_cost must be a scalar function"

        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control, self.auxvar], [self.path_cost])

    # set the final cost
    def setFinalCost(self, final_cost=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        if final_cost is not None:
            self.final_cost = final_cost
        else:
            self.final_cost = 0

        self.final_cost_fn = casadi.Function('final_cost', [self.state, self.auxvar], [self.final_cost])

    # set the path inequality constraints (if exists)
    def setPathInequCstr(self, path_inequality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.path_inequ_cstr = path_inequality_cstr

        if self.path_inequ_cstr is not None:

            self.path_inequ_cstr_fn = casadi.Function('path_inequality_constraint',
                                                      [self.state, self.control, self.auxvar],
                                                      [self.path_inequ_cstr])
            self.n_path_inequ_cstr = self.path_inequ_cstr_fn.numel_out()
        else:
            self.n_path_inequ_cstr = 0

    # set the path equality constraints (if exists)
    def setPathEquCstr(self, path_equality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.path_equ_cstr = path_equality_cstr

        if self.path_equ_cstr is not None:
            self.path_equ_cstr_fn = casadi.Function('path_equality_constraint', [self.state, self.control, self.auxvar],
                                                    [self.path_equ_cstr])
            self.n_path_equ_cstr = self.path_equ_cstr_fn.numel_out()
        else:
            self.n_path_equ_cstr = 0

    # set the final inequality constraints (if exists)
    def setFinalInequCstr(self, final_inequality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.final_inequ_cstr = final_inequality_cstr

        if self.final_inequ_cstr is not None:
            self.final_inequ_cstr_fn = casadi.Function('final_inequality_constraint', [self.state, self.auxvar],
                                                       [self.final_inequ_cstr])
            self.n_final_inequ_cstr = self.final_inequ_cstr_fn.numel_out()
        else:
            self.n_final_inequ_cstr = 0

    # set the final equality constraints (if exists)
    def setFinalEquCstr(self, final_equality_cstr=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.final_equ_cstr = final_equality_cstr

        if self.final_equ_cstr is not None:
            self.final_equ_cstr_fn = casadi.Function('path_equality_constraint', [self.state, self.auxvar],
                                                     [self.final_equ_cstr])
            self.n_final_equ_cstr = self.final_equ_cstr_fn.numel_out()
        else:
            self.n_final_equ_cstr = 0

    # set the initial condition function (if exists)
    def setInitCondition(self, initial_condition=None):
        if not hasattr(self, 'auxvar'):
            self.setAuxvarVariable()

        self.init_condition = initial_condition

        if self.init_condition is not None:
            self.init_condition_fn = casadi.Function('initial_condition', [self.auxvar], [self.init_condition])

    # optimal control solver (using direct methods)
    def ocSolver(self, horizon, init_state=None, auxvar_value=1, print_level=0):

        if not hasattr(self, 'final_equ_cstr'):
            self.setFinalEquCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'path_equ_cstr'):
            self.setPathEquCstr()
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'final_cost'):
            self.setFinalCost()

        if init_state is None:
            init_state = self.init_condition_fn(auxvar_value).full().flatten().tolist()
        else:
            init_state = casadi.DM(init_state).full().flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w += [Xk]
        lbw += init_state
        ubw += init_state
        w0 += init_state

        # formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w += [Uk]
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            # Add constraint for the path inequality if exist
            if self.path_inequ_cstr is not None:
                g += [self.path_inequ_cstr_fn(Xk, Uk, auxvar_value)]
                lbg += self.n_path_inequ_cstr * [- self.inf]
                ubg += self.n_path_inequ_cstr * [0]

            # Add constraint for the path equality if exist
            if self.path_equ_cstr is not None:
                g += [self.path_equ_cstr_fn(Xk, Uk, auxvar_value)]
                lbg += self.n_path_equ_cstr * [0]
                ubg += self.n_path_equ_cstr * [0]

            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk, auxvar_value)
            Ck = self.path_cost_fn(Xk, Uk, auxvar_value)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w += [Xk]
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add constraint for the dynamics
            g += [Xnext - Xk]
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # Add final inequality constraint if exist
        if self.final_inequ_cstr is not None:
            g += [self.final_inequ_cstr_fn(Xk, auxvar_value)]
            lbg += self.n_final_inequ_cstr * [- self.inf]
            ubg += self.n_final_inequ_cstr * [0]

        # Add final equality constraint if exist
        if self.final_equ_cstr is not None:
            g += [self.final_equ_cstr_fn(Xk, auxvar_value)]
            lbg += self.n_final_equ_cstr * [0]
            ubg += self.n_final_equ_cstr * [0]

        # Add the final cost
        J = J + self.final_cost_fn(Xk, auxvar_value)

        # Create an NLP solver and solve
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        lam_g = sol['lam_g'].full().flatten()
        g = sol['g'].full().flatten()

        # extract the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # compute the costate trajectory, mu trajectory and nu trajectory (the latter two is Lagrangian multipliers)
        v_w_lam_path = numpy.reshape(lam_g[:lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr],
                                     (-1, self.n_state + self.n_path_equ_cstr + self.n_path_inequ_cstr))
        v_path = v_w_lam_path[:, 0:self.n_path_inequ_cstr]
        w_path = v_w_lam_path[:, self.n_path_inequ_cstr:self.n_path_inequ_cstr + self.n_path_equ_cstr]
        costate_traj = v_w_lam_path[:, self.n_path_inequ_cstr + self.n_path_equ_cstr:]
        v_final = lam_g[
                  lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr:lam_g.size - self.n_final_equ_cstr]
        w_final = lam_g[lam_g.size - self.n_final_equ_cstr:]

        # compute the inequality constraint value
        g_h_f_traj = numpy.reshape(g[:g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr],
                                   (-1, self.n_state + self.n_path_equ_cstr + self.n_path_inequ_cstr))
        inequ_path = g_h_f_traj[:, 0:self.n_path_inequ_cstr]
        inequ_final = g[lam_g.size - self.n_final_inequ_cstr - self.n_final_equ_cstr:lam_g.size - self.n_final_equ_cstr]

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "costate_traj_opt": costate_traj,
                   "inequ_path": inequ_path,
                   "inequ_final": inequ_final,
                   "v_path": v_path,
                   "v_final": v_final,
                   "w_path": w_path,
                   "w_final": w_final,
                   'auxvar_value': auxvar_value,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    # get differential CPMP
    def diffCPMP(self, ):

        if not hasattr(self, 'final_equ_cstr'):
            self.setFinalEquCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'path_equ_cstr'):
            self.setPathEquCstr()
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'init_condition'):
            self.setInitCondition()

        # Differentiating dynamics; notations here are consistent with the PDP paper
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control, self.auxvar], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control, self.auxvar], [self.dfu])
        self.dfe = jacobian(self.dyn, self.auxvar)
        self.dfe_fn = casadi.Function('dfe', [self.state, self.control, self.auxvar], [self.dfe])

        # Define the path Hamiltonian-Lagrangian function (path_LP)
        costate = casadi.SX.sym('lambda', self.n_state)
        v_path = casadi.SX.sym('v_path', self.n_path_inequ_cstr)
        w_path = casadi.SX.sym('nu_path', self.n_path_equ_cstr)
        # specifically handle the none inequality and equality cases
        self.L_path = self.path_cost + dot(self.dyn, costate)
        if self.path_inequ_cstr is not None:
            self.L_path = self.L_path + dot(self.path_inequ_cstr, v_path)
        if self.path_equ_cstr is not None:
            self.L_path = self.L_path + dot(self.path_equ_cstr, w_path)

        # First-order derivative of path Hamiltonian-Lagrangian
        self.dLx_path = jacobian(self.L_path, self.state).T
        self.dLx_path_fn = casadi.Function('dLx_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLx_path])
        self.dLu_path = jacobian(self.L_path, self.control).T
        self.dLu_path_fn = casadi.Function('dLu_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLu_path])
        self.dLe_path = jacobian(self.L_path, self.auxvar).T
        self.dLe_path_fn = casadi.Function('dLe_path',
                                           [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                           [self.dLe_path])

        # Second-order derivative of Hamiltonian-Lagrangian
        self.ddLxx_path = jacobian(self.dLx_path, self.state)
        self.ddLxx_path_fn = casadi.Function('ddLxx_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxx_path])
        self.ddLxu_path = jacobian(self.dLx_path, self.control)
        self.ddLxu_path_fn = casadi.Function('ddLxu_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxu_path])
        self.ddLxe_path = jacobian(self.dLx_path, self.auxvar)
        self.ddLxe_path_fn = casadi.Function('ddLxe_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLxe_path])
        self.ddLux_path = jacobian(self.dLu_path, self.state)
        self.ddLux_path_fn = casadi.Function('ddLux_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLux_path])
        self.ddLuu_path = jacobian(self.dLu_path, self.control)
        self.ddLuu_path_fn = casadi.Function('ddLuu_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLuu_path])
        self.ddLue_path = jacobian(self.dLu_path, self.auxvar)
        self.ddLue_path_fn = casadi.Function('ddHue_path',
                                             [self.state, self.control, costate, v_path, w_path, self.auxvar],
                                             [self.ddLue_path])

        # Define the final Hamiltonian-Lagrangian function (final_H)
        v_final = casadi.SX.sym('v_final', self.n_final_inequ_cstr)
        w_final = casadi.SX.sym('nu_final', self.n_final_equ_cstr)
        # specifically handle the none inequality and equality cases
        self.L_final = self.final_cost
        if self.final_inequ_cstr is not None:
            self.L_final = self.L_final + dot(self.final_inequ_cstr, v_final)
        if self.final_equ_cstr is not None:
            self.L_final = self.L_final + dot(self.final_equ_cstr, w_final)

        # First-order derivative of final Hamiltonian-Lagrangian
        self.dLx_final = jacobian(self.L_final, self.state).T
        self.dLx_final_fn = casadi.Function('dLx_final',
                                            [self.state, v_final, w_final, self.auxvar],
                                            [self.dLx_final])

        # Second order differential of final Hamiltonian-Lagrangian
        self.ddLxx_final = jacobian(self.dLx_final, self.state)
        self.ddLxx_final_fn = casadi.Function('ddLxx_final', [self.state, v_final, w_final, self.auxvar],
                                              [self.ddLxx_final])
        self.ddLxe_final = jacobian(self.dLx_final, self.auxvar)
        self.ddLxe_final_fn = casadi.Function('ddLxe_final', [self.state, v_final, w_final, self.auxvar],
                                              [self.ddLxe_final])

        # differentiate the path equality constraint if exist
        if self.path_equ_cstr is not None:
            self.dHx_path = jacobian(self.path_equ_cstr, self.state)
            self.dHx_path_fn = Function('dHx_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHx_path])
            self.dHu_path = jacobian(self.path_equ_cstr, self.control)
            self.dHu_path_fn = Function('dHu_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHu_path])
            self.dHe_path = jacobian(self.path_equ_cstr, self.auxvar)
            self.dHe_path_fn = Function('dHe_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dHe_path])

        # differentiate the final equality constraint if exist
        if self.final_equ_cstr is not None:
            self.dHx_final = jacobian(self.final_equ_cstr, self.state)
            self.dHx_final_fn = Function('dHx_final_fn', [self.state, self.auxvar],
                                         [self.dHx_final])
            self.dHe_final = jacobian(self.final_equ_cstr, self.auxvar)
            self.dHe_final_fn = Function('dHe_final_fn', [self.state, self.auxvar],
                                         [self.dHe_final])

        # differentiate the path inequality constraint if exist
        if self.path_inequ_cstr is not None:
            self.dGx_path = jacobian(self.path_inequ_cstr, self.state)
            self.dGx_path_fn = Function('dGx_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGx_path])
            self.dGu_path = jacobian(self.path_inequ_cstr, self.control)
            self.dGu_path_fn = Function('dGu_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGu_path])
            self.dGe_path = jacobian(self.path_inequ_cstr, self.auxvar)
            self.dGe_path_fn = Function('dGe_path_fn', [self.state, self.control, self.auxvar],
                                        [self.dGe_path])

        # differentiate the final inequality constraint if exist
        if self.final_inequ_cstr is not None:
            self.dGx_final = jacobian(self.final_inequ_cstr, self.state)
            self.dGx_final_fn = Function('dHx_final_fn', [self.state, self.auxvar],
                                         [self.dGx_final])
            self.dGe_final = jacobian(self.final_inequ_cstr, self.auxvar)
            self.dGe_final_fn = Function('dHe_final_fn', [self.state, self.auxvar],
                                         [self.dGe_final])

        # differentiate the initial condition if parameterized
        if self.init_condition is not None:
            self.dX0 = jacobian(self.init_condition, self.state)
            self.dx0_fn = Function('dx0_fn', [self.auxvar], [self.dX0])
        else:
            self.dX0_fn = Function('dx0_fn', [self.auxvar], [SX.zeros(self.n_state, self.n_auxvar)])

    # get the auxiliary control system (here the threshold is to determine the active inequality constraints
    def getAuxSys(self, opt_sol, threshold=1e-2):

        # parse the optimal solution argument opt_sol
        state_traj_opt = opt_sol['state_traj_opt']
        control_traj_opt = opt_sol['control_traj_opt']
        costate_traj = opt_sol['costate_traj_opt']
        auxvar_value = opt_sol['auxvar_value']

        v_path = opt_sol['v_path']
        g_path = opt_sol['inequ_path']
        v_final = opt_sol['v_final']
        g_final = opt_sol['inequ_final']
        w_path = opt_sol['w_path']
        w_final = opt_sol['w_final']

        # in case of not differentiating the PMP
        if not hasattr(self, 'dLx_path'):
            self.diffCPMP()

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the constraint PDP paper.
        dynFx_t, dynFu_t, dynFe_t = [], [], []
        Lxx_t, Lxu_t, Lxe_t, Lux_t, Luu_t, Lue_t = [], [], [], [], [], []
        GbarHx_t, GbarHu_t, GbarHe_t = [], [], []  # this is the concatenated matrix G_bar_t and H_t in the constraint PDP paper
        GbarHx_T, GbarHe_T, = [], []  # this is the concatenated matrix G_bar_T and H_T in the constraint PDP paper

        horizon = numpy.size(control_traj_opt, 0)
        for t in range(horizon):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            next_lambda = costate_traj[t, :]
            curr_v = v_path[t, :]
            curr_w = w_path[t, :]
            curr_g = g_path[t, :]

            dynFx_t += [self.dfx_fn(curr_x, curr_u, auxvar_value).full()]
            dynFu_t += [self.dfu_fn(curr_x, curr_u, auxvar_value).full()]
            dynFe_t += [self.dfe_fn(curr_x, curr_u, auxvar_value).full()]

            Lxx_t += [self.ddLxx_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lxu_t += [self.ddLxu_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lxe_t += [self.ddLxe_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]

            Lux_t += [self.ddLux_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Luu_t += [self.ddLuu_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]
            Lue_t += [self.ddLue_path_fn(curr_x, curr_u, next_lambda, curr_v, curr_w, auxvar_value).full()]

            # generate the G_bar_t and H_t, where B_bar_t is identified using the threshold
            if self.path_inequ_cstr is not None:
                Gbarx_t = self.dGx_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
                Gbaru_t = self.dGu_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
                Gbare_t = self.dGe_path_fn(curr_x, curr_u, auxvar_value).full()[(curr_g > -threshold)]
            else:
                Gbarx_t = np.empty((0, self.n_state))
                Gbaru_t = np.empty((0, self.n_control))
                Gbare_t = np.empty((0, self.n_auxvar))

            if self.path_equ_cstr is not None:
                Hx_t = self.dHx_path_fn(curr_x, curr_u, auxvar_value).full()
                Hu_t = self.dHu_path_fn(curr_x, curr_u, auxvar_value).full()
                He_t = self.dHe_path_fn(curr_x, curr_u, auxvar_value).full()
            else:
                Hx_t = np.empty((0, self.n_state))
                Hu_t = np.empty((0, self.n_control))
                He_t = np.empty((0, self.n_auxvar))

            GbarHx_t += [np.vstack((Gbarx_t, Hx_t))]
            GbarHu_t += [np.vstack((Gbaru_t, Hu_t))]
            GbarHe_t += [np.vstack((Gbare_t, He_t))]

        # handle the final cost, inequality, equality constraints
        Lxx_T = [self.ddLxx_final_fn(state_traj_opt[-1, :], v_final, w_final, auxvar_value).full()]
        Lxe_T = [self.ddLxe_final_fn(state_traj_opt[-1, :], v_final, w_final, auxvar_value).full()]

        if self.final_inequ_cstr is not None:
            Gbarx_T = self.dGx_final_fn(state_traj_opt[-1, :], auxvar_value).full()[(v_final > -threshold)]
            Gbare_T = self.dGe_final_fn(state_traj_opt[-1, :], auxvar_value).full()[(g_final > -threshold)]
        else:
            Gbarx_T = np.empty((0, self.n_state))
            Gbare_T = np.empty((0, self.n_auxvar))

        if self.final_equ_cstr is not None:
            Hx_T = self.dHx_final_fn(state_traj_opt[-1, :], auxvar_value).full()
            He_T = self.dHe_final_fn(state_traj_opt[-1, :], auxvar_value).full()
        else:
            Hx_T = np.empty((0, self.n_state))
            He_T = np.empty((0, self.n_auxvar))

        GbarHx_T += [np.vstack((Gbarx_T, Hx_T))]
        GbarHe_T += [np.vstack((Gbare_T, He_T))]

        # print(GbarHx_T, GbarHe_T)

        # return the axuliary control system
        X0 = self.dX0_fn(auxvar_value).full()
        auxSys = {"dynFx_t": dynFx_t,
                  "dynFu_t": dynFu_t,
                  "dynFe_t": dynFe_t,
                  "Lxx_t": Lxx_t,
                  "Lxu_t": Lxu_t,
                  "Lxe_t": Lxe_t,
                  "Lux_t": Lux_t,
                  "Luu_t": Luu_t,
                  "Lue_t": Lue_t,
                  "Lxx_T": Lxx_T,
                  "Lxe_T": Lxe_T,
                  "GbarHx_t": GbarHx_t,
                  "GbarHu_t": GbarHu_t,
                  "GbarHe_t": GbarHe_t,
                  "GbarHx_T": GbarHx_T,
                  "GbarHe_T": GbarHe_T,
                  "X0": X0,
                  "horizon": horizon
                  }

        return auxSys

    '''
    The following it to solve a constrained optimal control by converting to an unconstrained optimal control then 
    solve it using PDP.
    '''

    # This function is to convert a constrained optimal control system into an unconstrained optimal control then
    # using PDP
    def convert2BarrierOC(self, gamma=1e-2):

        # in case of not differentiating the PMP
        if not hasattr(self, 'dLx_path'):
            self.diffCPMP()

        if not hasattr(self, 'path_equ_cstr'):
            self.setPathEquCstr()
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'final_equ_cstr'):
            self.setFinalEquCstr()

        self.barrier_oc = PDP.OCSys()
        self.barrier_oc.setAuxvarVariable(self.auxvar)
        self.barrier_oc.setStateVariable(self.state)
        self.barrier_oc.setControlVariable(self.control)
        self.barrier_oc.setDyn(self.dyn)

        # natural log barrier for the inequality path constraints
        path_inequ_barrier = 0
        if self.n_path_inequ_cstr == 1:
            path_inequ_barrier += -log(-self.path_inequ_cstr)
        else:
            for k in range(self.n_path_inequ_cstr):
                path_inequ_barrier += -log(-self.path_inequ_cstr[k])

        # second-order barrier for the equality path constraints
        path_equ_barrier = 0
        if self.n_path_equ_cstr == 1:
            path_equ_barrier += (self.path_inequ_cstr) ** 2
        else:
            for k in range(self.n_path_equ_cstr):
                path_equ_barrier += (self.path_inequ_cstr[k]) ** 2

        # overall cost plus the barrier in path
        path_costbarrier = self.path_cost + gamma * path_inequ_barrier + 0.5 / gamma * path_equ_barrier
        self.barrier_oc.setPathCost(path_costbarrier)

        # natural log barrier for the inequality final constraints
        final_inequ_barrier = 0
        if self.n_final_inequ_cstr == 1:
            final_inequ_barrier += -log(-self.final_inequ_cstr)
        else:
            for k in range(self.n_final_inequ_cstr):
                final_inequ_barrier += -log(-self.final_inequ_cstr[k])

        # second-order barrier for the equality final constraints
        final_equ_barrier = 0
        if self.n_final_equ_cstr == 1:
            final_equ_barrier += (self.final_equ_cstr) ** 2
        else:
            for k in range(self.n_final_equ_cstr):
                final_equ_barrier += (self.final_equ_cstr[k]) ** 2

        # overall cost plus the barrier at final
        final_costbarrier = self.final_cost + gamma * final_inequ_barrier + 0.5 / gamma * final_equ_barrier
        self.barrier_oc.setFinalCost(final_costbarrier)

        # differentiating PDP for the barrier optimal control
        self.barrier_oc.diffPMP()

        # create the equality constraints lqr solver object
        self.lqr_solver_barrierOC = PDP.LQR()

    # compute the unconstrained optimal control using PDP
    def solveBarrierOC(self, horizon, init_state=None, auxvar_value=1):

        if init_state is None:
            init_state = self.init_condition_fn(auxvar_value).full().flatten().tolist()
        else:
            init_state = casadi.DM(init_state).full().flatten().tolist()

        opt_sol = self.barrier_oc.ocSolver(ini_state=init_state, horizon=horizon, auxvar_value=auxvar_value)

        return opt_sol

    # generate the auxiliary control system using the optimal trajectory
    def auxSysBarrierOC(self, opt_sol):
        horizon = numpy.size(opt_sol['control_traj_opt'], 0)

        auxsys_barrierOC = self.barrier_oc.getAuxSys(state_traj_opt=opt_sol['state_traj_opt'],
                                                     control_traj_opt=opt_sol['control_traj_opt'],
                                                     costate_traj_opt=opt_sol['costate_traj_opt'],
                                                     auxvar_value=opt_sol['auxvar_value'])
        self.lqr_solver_barrierOC.setDyn(dynF=auxsys_barrierOC['dynF'], dynG=auxsys_barrierOC['dynG'],
                                         dynE=auxsys_barrierOC['dynE'])
        self.lqr_solver_barrierOC.setPathCost(Hxx=auxsys_barrierOC['Hxx'], Huu=auxsys_barrierOC['Huu'],
                                              Hxu=auxsys_barrierOC['Hxu'], Hux=auxsys_barrierOC['Hux'],
                                              Hxe=auxsys_barrierOC['Hxe'], Hue=auxsys_barrierOC['Hue'])
        self.lqr_solver_barrierOC.setFinalCost(hxx=auxsys_barrierOC['hxx'], hxe=auxsys_barrierOC['hxe'])

        X0 = self.dX0_fn(opt_sol['auxvar_value']).full()

        aux_sol = self.lqr_solver_barrierOC.lqrSolver(X0, horizon)

        return aux_sol


# This equality constraint LQR solver is mainly based on the paper
# Efficient Computation of Feedback Control for Equality-Constrained LQR by Claire Tomlin.
class EQCLQR:

    def __init__(self, project_name='my constraint lqr solver'):
        self.project_name = project_name
        self.threshold = 1e-5  # this threshold is used to detect the rank of the matrix

    def setDyn(self, dynFx_t, dynFu_t, dynFe_t):
        self.dynFx_t = dynFx_t
        self.dynFu_t = dynFu_t
        self.dynFe_t = dynFe_t
        self.n_state = dynFx_t[0].shape[1]
        self.n_control = dynFu_t[0].shape[1]

    def setPathCost(self, Lxx_t, Lxu_t, Lxe_t, Lux_t, Luu_t, Lue_t):
        self.Lxx_t = Lxx_t
        self.Lxu_t = Lxu_t
        self.Lxe_t = Lxe_t
        self.Lux_t = Lux_t
        self.Luu_t = Luu_t
        self.Lue_t = Lue_t

    def setFinalCost(self, Lxx_T, Lxe_T):
        self.Lxx_T = Lxx_T
        self.Lxe_T = Lxe_T

    def setPathConstraints(self, Gx_t, Gu_t, Ge_t):
        self.Gx_t = Gx_t
        self.Gu_t = Gu_t
        self.Ge_t = Ge_t

    def setFinalConstraints(self, Gx_T, Ge_T):
        self.Gx_T = Gx_T
        self.Ge_T = Ge_T

    def eqctlqrSolver(self, init_state=None, horizon=None, threshold=None):

        if init_state is None:
            init_state = self.init_state
        if horizon is None:
            horizon = self.horizon

        if threshold is not None:
            self.threshold = threshold

        # backward pass in time to compute the feedback control matrices
        # pre-set the feedback matrices storage
        Kt = []
        kt = []
        # final cost initialization
        next_V = self.Lxx_T[0]
        next_v = self.Lxe_T[0]
        next_H = self.Gx_T[0]
        next_h = self.Ge_T[0]

        for t in range(horizon - 1, -1, -1):
            # take the current dynamics
            curr_Fx = self.dynFx_t[t]
            curr_Fu = self.dynFu_t[t]
            curr_Fe = self.dynFe_t[t]

            # take the current path cost
            curr_Lxx = self.Lxx_t[t]
            curr_Lxu = self.Lxu_t[t]
            curr_Lxe = self.Lxe_t[t]
            curr_Lux = self.Lux_t[t]
            curr_Luu = self.Luu_t[t]
            curr_Lue = self.Lue_t[t]

            # take the current equality constraints
            curr_Gx = self.Gx_t[t]
            curr_Gu = self.Gu_t[t]
            curr_Ge = self.Ge_t[t]

            # generate the new matrix (Claire Tomlin's paper has some errors, I corrected it here)
            curr_Mxe = curr_Lxe + curr_Fx.T @ next_v + curr_Fx.T @ next_V @ curr_Fe
            curr_Mue = curr_Lue + curr_Fu.T @ next_v + curr_Fu.T @ next_V @ curr_Fe
            curr_Mxx = curr_Lxx + curr_Fx.T @ next_V @ curr_Fx
            curr_Muu = curr_Luu + curr_Fu.T @ next_V @ curr_Fu
            curr_Mux = curr_Lux + curr_Fu.T @ next_V @ curr_Fx
            curr_Nx = np.vstack((curr_Gx, next_H @ curr_Fx))
            curr_Nu = np.vstack((curr_Gu, next_H @ curr_Fu))
            curr_Ne = np.vstack((curr_Ge, next_H @ curr_Fe + next_h))

            # Generate the matrix Py and Zw in the equality constraint lqr paper
            U, S, VT = np.linalg.svd(curr_Nu)
            V = VT.T
            rank_S = (S > self.threshold).size
            if S.size == 0 or rank_S == 0:  # i.e., if the rank of curr_Nu is zero
                Py = np.empty((self.n_control, 0))
                Zw = np.eye(self.n_control)
            elif rank_S == self.n_control:  # i.e., if the rank of curr_Nu is the full column rank
                Py = np.eye(self.n_control)
                Zw = np.empty((self.n_control, 0))
            else:
                Py = V[:, 0:rank_S]
                Zw = V[:, rank_S:]

            # Generate the feedback control policy matrices curr_K, and curr_k
            pinv_NuPy = np.linalg.pinv(curr_Nu @ Py)
            inv_ZwTMuuZw = np.linalg.inv(Zw.T @ curr_Muu @ Zw)
            curr_K = -(Py @ pinv_NuPy @ curr_Nx + Zw @ inv_ZwTMuuZw @ Zw.T @ curr_Mux)
            curr_k = -(Py @ pinv_NuPy @ curr_Ne + Zw @ inv_ZwTMuuZw @ Zw.T @ curr_Mue)

            # store
            Kt += [curr_K]
            kt += [curr_k]

            # reiterate the  next_V, next_v, next_H, next_h
            next_H = curr_Nx - curr_Nu @ Py @ pinv_NuPy @ curr_Nx
            next_h = curr_Ne - curr_Nu @ Py @ pinv_NuPy @ curr_Ne
            next_V = curr_Mxx + 2 * curr_Mux.T @ curr_K + curr_K.T @ curr_Muu @ curr_K
            next_v = curr_Mxe + curr_K.T @ curr_Mue + (curr_Mux.T + curr_K.T @ curr_Muu) @ curr_k

            # print('P', curr_k)

        # forward pass in time to compute the optimal trajectory
        # pre-set the trajectory storage
        state_traj_opt = [init_state]
        control_traj_opt = []
        for t in range(horizon):
            # take the current control matrices
            curr_K = Kt[horizon - t - 1]
            curr_k = kt[horizon - t - 1]
            # take the current state
            curr_x = state_traj_opt[t]
            # apply the control law to compute the current control input
            curr_u = curr_K @ curr_x + curr_k
            # apply the dynamics
            curr_Fx = self.dynFx_t[t]
            curr_Fu = self.dynFu_t[t]
            curr_Fe = self.dynFe_t[t]
            next_x = curr_Fx @ curr_x + curr_Fu @ curr_u + curr_Fe
            # store
            state_traj_opt += [next_x]
            control_traj_opt += [curr_u]

        # output the trajectory
        time = [k for k in range(horizon + 1)]
        opt_sol = {'state_traj_opt': state_traj_opt,
                   'control_traj_opt': control_traj_opt,
                   'time': time}

        return opt_sol

    # This method here is specifically to solve the auxiliary control system in PDP
    # Since there are some notation/variable differences between the auxiliary system and LQR here.
    # This method is to convert the auxiliary control system to the language of lqr solver object
    def auxsys2Eqctlqr(self, auxsys):
        self.setDyn(dynFx_t=auxsys['dynFx_t'],
                    dynFu_t=auxsys['dynFu_t'],
                    dynFe_t=auxsys['dynFe_t'])
        self.setPathCost(Lxx_t=auxsys['Lxx_t'],
                         Lxu_t=auxsys['Lxu_t'],
                         Lxe_t=auxsys['Lxe_t'],
                         Lux_t=auxsys['Lux_t'],
                         Luu_t=auxsys['Luu_t'],
                         Lue_t=auxsys['Lue_t'])
        self.setFinalCost(Lxx_T=auxsys['Lxx_T'],
                          Lxe_T=auxsys['Lxe_T'])
        self.setPathConstraints(Gx_t=auxsys['GbarHx_t'],
                                Gu_t=auxsys['GbarHu_t'],
                                Ge_t=auxsys['GbarHe_t'])
        self.setFinalConstraints(Gx_T=auxsys['GbarHx_T'],
                                 Ge_T=auxsys['GbarHe_T'])
        self.init_state = auxsys['X0']
        self.horizon = auxsys['horizon']

        # print(np.array(self.Gx_t).size, np.array(self.Ge_t).size)


# This class is used to solve the constrained policy optimization or trajectory optimization
class CSysOPT:

    def __init__(self, project_name='constraint_planner'):
        self.project_name = project_name

    # define the state variable and its allowable upper and lower bounds
    def setStateVariable(self, state, ):
        self.state = state
        self.n_state = self.state.numel()

    # define the control input variable and its allowable upper and lower bounds
    def setControlVariable(self, control, ):
        self.control = control
        self.n_control = self.control.numel()

    # set the dynamics model
    def setDyn(self, ode):

        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control], [self.dyn])

    # set the path cost
    def setPathCost(self, path_cost):
        assert path_cost.numel() == 1, "path_cost must be a scalar function"
        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control], [self.path_cost])

    # set the final cost
    def setFinalCost(self, final_cost=None):
        if final_cost is not None:
            self.final_cost = final_cost
        else:
            self.final_cost = 0
        self.final_cost_fn = casadi.Function('final_cost', [self.state], [self.final_cost])

    # set the path inequality constraints (if exists)
    def setPathInequCstr(self, path_inequality_cstr=None):
        self.path_inequ_cstr = path_inequality_cstr
        if self.path_inequ_cstr is not None:

            self.path_inequ_cstr_fn = casadi.Function('path_inequality_constraint',
                                                      [self.state, self.control],
                                                      [self.path_inequ_cstr])
            self.n_path_inequ_cstr = self.path_inequ_cstr_fn.numel_out()
        else:
            self.n_path_inequ_cstr = 0

    # set the final inequality constraints (if exists)
    def setFinalInequCstr(self, final_inequality_cstr=None):
        self.final_inequ_cstr = final_inequality_cstr

        if self.final_inequ_cstr is not None:
            self.final_inequ_cstr_fn = casadi.Function('final_inequality_constraint', [self.state],
                                                       [self.final_inequ_cstr])
            self.n_final_inequ_cstr = self.final_inequ_cstr_fn.numel_out()
        else:
            self.n_final_inequ_cstr = 0

    # convert the constrained problem into an unconstrained problem
    def convert2BarrierOC(self, gamma=1e-2):

        # in case of not has path and final constraints
        if not hasattr(self, 'path_inequ_cstr'):
            self.setPathInequCstr()
        if not hasattr(self, 'final_inequ_cstr'):
            self.setFinalInequCstr()
        if not hasattr(self, 'final_cost'):
            self.setFinalCost()

        # natural log barrier for the inequality path constraints
        path_inequ_barrier = 0
        if self.n_path_inequ_cstr == 1:
            path_inequ_barrier += -log(-self.path_inequ_cstr)
        else:
            for k in range(self.n_path_inequ_cstr):
                path_inequ_barrier += -log(-self.path_inequ_cstr[k])

        # overall cost plus the barrier in path
        self.path_costbarrier = self.path_cost + gamma * path_inequ_barrier
        self.path_costbarrier_fn = casadi.Function('path_costbarrier_fn', [self.state, self.control],
                                                   [self.path_costbarrier])

        # natural log barrier for the inequality final constraints
        final_inequ_barrier = 0
        if self.n_final_inequ_cstr == 1:
            final_inequ_barrier += -log(-self.final_inequ_cstr)
        else:
            for k in range(self.n_final_inequ_cstr):
                final_inequ_barrier += -log(-self.final_inequ_cstr[k])

        # overall cost plus the barrier at final
        self.final_costbarrier = self.final_cost + gamma * final_inequ_barrier
        self.final_costbarrier_fn = casadi.Function('final_costbarrier_fn', [self.state], [self.final_costbarrier])

    # set the trajectory parameterization for planning task
    def setPolyTraj(self, horizon, n_poly=5):
        # Use the Lagrange polynomial to represent the control trajectory function: u_t=u(t,control_auxvar).
        # The state trajectory is a result of dynamics and the polynomial trajectory.
        # Note that  we parameterize the control trajectory as Lagrangian polynomial with the parameters being the pivot

        # set the control polynomial policy
        pivots = numpy.linspace(0, horizon, n_poly + 1)

        # time variable
        self.t = SX.sym('t')

        # pivots of time steps for the Lagrange polynomial
        control_traj = 0
        control_pivots = []
        for i in range(len(pivots)):
            Ui = SX.sym('U_' + str(i), self.n_control)
            control_pivots += [Ui]
            bi = 1
            for j in range(len(pivots)):
                if j != i:
                    bi = bi * (self.t - pivots[j]) / (pivots[i] - pivots[j])
            control_traj = control_traj + bi * Ui
        self.control_auxvar = vcat(control_pivots)
        self.n_control_auxvar = self.control_auxvar.numel()
        self.control_fn = casadi.Function('control_fn', [self.t, self.state, self.control_auxvar],
                                          [control_traj])

    # set the policy parameterization for policy optimization
    def setNeuralPolicy(self, hidden_layers):
        # Use neural network to represent the policy function: u_t=u(t,x,auxvar).
        # Note that here we use auxvar to denote the parameter of the neural policy
        layers = hidden_layers + [self.n_control]

        assert type(hidden_layers) is list, "The hidden_layers needs to be a list"

        # time variable
        if not hasattr(self, 't'):
            self.t = SX.sym('t')

        # construct the neural policy with the argument inputs to specify the hidden layers of the neural policy
        a = self.state
        auxvar = []
        Ak = SX.sym('Ak', layers[0], self.n_state)  # weights matrix
        bk = SX.sym('bk', layers[0])  # bias vector
        auxvar += [Ak.reshape((-1, 1))]
        auxvar += [bk]
        a = mtimes(Ak, a) + bk
        for i in range(len(layers) - 1):
            a = tanh(a)
            Ak = SX.sym('Ak', layers[i + 1], layers[i])  # weights matrix
            bk = SX.sym('bk', layers[i + 1])  # bias vector
            auxvar += [Ak.reshape((-1, 1))]
            auxvar += [bk]
            a = mtimes(Ak, a) + bk
        self.control_auxvar = vcat(auxvar)
        self.n_control_auxvar = self.control_auxvar.numel()
        neural_policy = a
        self.neural_policy_fn = casadi.Function('neural_policy_fn', [self.state, self.control_auxvar], [neural_policy])
        self.control_fn = casadi.Function('control_fn', [self.t, self.state, self.control_auxvar], [neural_policy])

    # differentiate cost and dynamics
    def diffSys(self):
        # in case of there is no t variable
        assert hasattr(self, 't'), "set the control parameterization first! Either use setPolyTraj or setNeuralPolicy."

        # subsitute the policy into the dynamics ode
        self.controlled_dyn_fn = casadi.Function('controlled_dyn_fn',
                                                 [self.t, self.state, self.control_auxvar],
                                                 [self.dyn_fn(self.state, self.control_fn(self.t, self.state,
                                                                                          self.control_auxvar))])
        # differentiate the controlled dynamics w.r.t. state and control_auxvar
        dfx = jacobian(self.controlled_dyn_fn(self.t, self.state, self.control_auxvar), self.state)
        self.dfx_fn = casadi.Function('dfx_fn', [self.t, self.state, self.control_auxvar], [dfx])
        dfe = jacobian(self.controlled_dyn_fn(self.t, self.state, self.control_auxvar), self.control_auxvar)
        self.dfe_fn = casadi.Function('dfe_fn', [self.t, self.state, self.control_auxvar], [dfe])

        # in case of not has path cost_barrier and final cost_barrier
        if not hasattr(self, 'path_costbarrier'):
            self.convert2BarrierOC()

        # differentiate path cost_barrier
        self.controlled_path_cb_fn = casadi.Function('controlled_path_cb_fn', [self.t, self.state, self.control_auxvar],
                                                     [self.path_costbarrier_fn(self.state,
                                                                               self.control_fn(self.t, self.state,
                                                                                               self.control_auxvar))])
        dCBx_path = jacobian(self.controlled_path_cb_fn(self.t, self.state, self.control_auxvar), self.state).T
        self.dCBx_path_fn = casadi.Function('dCBx_path_fn', [self.t, self.state, self.control_auxvar], [dCBx_path])
        dCBe_path = jacobian(self.controlled_path_cb_fn(self.t, self.state, self.control_auxvar), self.control_auxvar).T
        self.dCBe_path_fn = casadi.Function('dCBe_path_fn', [self.t, self.state, self.control_auxvar], [dCBe_path])
        ddCBxx_path = jacobian(dCBx_path, self.state)
        self.ddCBxx_path_fn = casadi.Function('ddCBxx_path_fn', [self.t, self.state, self.control_auxvar],
                                              [ddCBxx_path])
        ddCBxe_path = jacobian(dCBx_path, self.control_auxvar)
        self.ddCBxe_path_fn = casadi.Function('ddCBxe_path_fn', [self.t, self.state, self.control_auxvar],
                                              [ddCBxe_path])
        ddCBee_path = jacobian(dCBe_path, self.control_auxvar)
        self.ddCBee_path_fn = casadi.Function('ddCBee_path_fn', [self.t, self.state, self.control_auxvar],
                                              [ddCBee_path])

        # differentiate final cost_barrier
        self.controlled_final_cb_fn = casadi.Function('controlled_final_cb_fn',
                                                      [self.state], [self.final_costbarrier_fn(self.state)])
        dCBx_final = jacobian(self.controlled_final_cb_fn(self.state), self.state).T
        self.dCBx_final_fn = casadi.Function('dCBx_final_fn', [self.state], [dCBx_final])
        ddCBxx_final = jacobian(dCBx_final, self.state)
        self.ddCBxx_final_fn = casadi.Function('ddCBxx_final_fn', [self.state], [ddCBxx_final])

    # one step to compute gradient descent
    def step(self, init_state, horizon, control_auxvar_value, damping_flag=False, damping_lambda=0.001):
        # in case of not differentating the system
        if not hasattr(self, 'ddCBxx_final_fn'):
            self.diffSys()

        if type(init_state) == list:
            init_state = numpy.array(init_state)

        # integrate the system to obtain the state trajectory based on the current control_auxvar_value
        # at the same time obtain the differential matrices of the system
        state_traj = [init_state]
        control_traj = []
        cost_barrier_value = 0
        cost_value = 0
        Fx = []
        Fe = []
        CBx = []
        CBe = []
        CBxx = []
        CBxe = []
        CBee = []
        for t in range(horizon):
            curr_x = state_traj[t]
            next_x = self.controlled_dyn_fn(t, curr_x, control_auxvar_value).full().flatten()
            curr_u = self.control_fn(t, curr_x, control_auxvar_value).full().flatten()
            curr_Fx = self.dfx_fn(t, curr_x, control_auxvar_value).full()
            curr_Fe = self.dfe_fn(t, curr_x, control_auxvar_value).full()
            curr_CBx = self.dCBx_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBe = self.dCBe_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBxx = self.ddCBxx_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBee = self.ddCBee_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBxe = self.ddCBxe_path_fn(t, curr_x, control_auxvar_value).full()
            cost_barrier_value += self.controlled_path_cb_fn(t, curr_x, control_auxvar_value).full()
            cost_value += self.path_cost_fn(curr_x, curr_u).full()
            state_traj += [next_x]
            control_traj += [curr_u]
            Fx += [curr_Fx]
            Fe += [curr_Fe]
            CBx += [curr_CBx]
            CBe += [curr_CBe]
            CBxx += [curr_CBxx]
            CBee += [curr_CBee]
            CBxe += [curr_CBxe]
        curr_x = state_traj[-1]
        CBxx += [self.ddCBxx_final_fn(curr_x).full()]
        CBx += [self.dCBx_final_fn(curr_x).full()]
        cost_barrier_value += self.controlled_final_cb_fn(curr_x)
        cost_value += self.final_cost_fn(curr_x)

        # compute the (second-order) gradient of the cost_barrier function using raccati-type equation
        P = [None] * (horizon + 1)
        p = [None] * (horizon + 1)
        K = [None] * (horizon + 1)
        W = [None] * (horizon + 1)
        w = [None] * (horizon + 1)
        P[-1] = CBxx[-1]
        p[-1] = CBx[-1]
        K[-1] = np.zeros((self.n_state, self.n_control_auxvar))
        W[-1] = np.zeros((self.n_control_auxvar, self.n_control_auxvar))
        w[-1] = np.zeros((self.n_control_auxvar, 1))
        for t in range(horizon, 0, -1):
            curr_P = P[t]
            curr_p = p[t]
            curr_K = K[t]
            curr_W = W[t]
            curr_w = w[t]
            P[t - 1] = CBxx[t - 1] + Fx[t - 1].T @ curr_P @ Fx[t - 1]
            p[t - 1] = CBx[t - 1] + Fx[t - 1].T @ curr_p
            K[t - 1] = CBxe[t - 1] + Fx[t - 1].T @ curr_P @ Fe[t - 1] + Fx[t - 1].T @ curr_K
            W[t - 1] = CBee[t - 1] + Fe[t - 1].T @ curr_P @ Fe[t - 1] + curr_W + Fe[t - 1].T @ curr_K + curr_K.T @ Fe[
                t - 1]
            w[t - 1] = CBe[t - 1] + curr_w + Fe[t - 1].T @ curr_p

        # solve for the (second-order) gradient where W[0] is the hessian matrix and w[0] is the gradient
        # to guarantee the henssian matrix is positive, we use the Levenberg-Marquardt e

        if damping_flag:
            u, s, vh = np.linalg.svd(W[0])
            s = s + damping_lambda
            W[0] = u @ np.diag(s) @ vh

        grad_control_auxvar = 0.5 * np.linalg.inv(W[0]) @ w[0]

        return cost_barrier_value, cost_value, grad_control_auxvar.flatten(), np.array(state_traj), np.array(
            control_traj),

    # integrating system given the initial condition and control function parameter (which is part of step method)
    def integrateSys(self, init_state, horizon, control_auxvar_value):
        # in case of not differentating the system
        if not hasattr(self, 'ddCBxx_final_fn'):
            self.diffSys()

        if type(init_state) == list:
            init_state = numpy.array(init_state)

        # integrate the system to obtain the state trajectory based on the current control_auxvar_value
        # at the same time obtain the differential matrices of the system
        state_traj = [init_state]
        control_traj = []
        cost_barrier_value = 0
        cost_value = 0
        Fx = []
        Fe = []
        CBx = []
        CBe = []
        CBxx = []
        CBxe = []
        CBee = []
        for t in range(horizon):
            curr_x = state_traj[t]
            next_x = self.controlled_dyn_fn(t, curr_x, control_auxvar_value).full().flatten()
            curr_u = self.control_fn(t, curr_x, control_auxvar_value).full().flatten()
            curr_Fx = self.dfx_fn(t, curr_x, control_auxvar_value).full()
            curr_Fe = self.dfe_fn(t, curr_x, control_auxvar_value).full()
            curr_CBx = self.dCBx_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBe = self.dCBe_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBxx = self.ddCBxx_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBee = self.ddCBee_path_fn(t, curr_x, control_auxvar_value).full()
            curr_CBxe = self.ddCBxe_path_fn(t, curr_x, control_auxvar_value).full()
            cost_barrier_value += self.controlled_path_cb_fn(t, curr_x, control_auxvar_value).full()
            cost_value += self.path_cost_fn(curr_x, curr_u).full()
            state_traj += [next_x]
            control_traj += [curr_u]
            Fx += [curr_Fx]
            Fe += [curr_Fe]
            CBx += [curr_CBx]
            CBe += [curr_CBe]
            CBxx += [curr_CBxx]
            CBee += [curr_CBee]
            CBxe += [curr_CBxe]
        curr_x = state_traj[-1]
        CBxx += [self.ddCBxx_final_fn(curr_x).full()]
        CBx += [self.dCBx_final_fn(curr_x).full()]
        cost_barrier_value += self.controlled_final_cb_fn(curr_x)
        cost_value += self.final_cost_fn(curr_x)

        return np.array(state_traj), np.array(control_traj), cost_barrier_value, cost_value,


# This is some utility functions

# compute the l2 trajectory loss and its gradient
def Traj_L2_Loss(demo_traj, traj, aux_sol):
    demo_state_traj = demo_traj['state_traj_opt']
    demo_control_traj = demo_traj['control_traj_opt']
    state_traj = traj['state_traj_opt']
    control_traj = traj['control_traj_opt']

    dldx_traj = state_traj - demo_state_traj
    dldu_traj = control_traj - demo_control_traj
    loss = numpy.linalg.norm(dldx_traj) ** 2 + numpy.linalg.norm(dldu_traj) ** 2

    dxdp_traj = aux_sol['state_traj_opt']
    dudp_traj = aux_sol['control_traj_opt']

    # use chain rule to compute the gradient
    dl = 0
    for t in range(len(demo_control_traj)):
        dl = dl + np.matmul(dldx_traj[t, :], dxdp_traj[t]) + np.matmul(dldu_traj[t, :], dudp_traj[t])
    dl = dl + numpy.dot(dldx_traj[-1, :], dxdp_traj[-1])

    return loss, dl


# check a matrix the symmetricity
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)


# supervised learning of a deep neural network where the nn function by default has the second argument be its parameter
def Traning_NN(nn_fn, in_data, out_data, max_iter=10000, lr=1e-3, display=False, seed=100):
    # get the function input argument's dimensions
    n_auxvar = nn_fn.size1_in(1)
    n_x = nn_fn.size1_in(0)
    # define the input and parameter varaible
    auxvar = SX.sym('auxvar', n_auxvar)

    # pre-process the data
    in_data = in_data.T  # convert to each column being a sample
    out_data = out_data.T  # convert to each column being a output
    in_data = in_data[:, 0:min(in_data.shape[1], out_data.shape[1])]
    out_data = out_data[:, 0:min(in_data.shape[1], out_data.shape[1])]

    # define the loss function and its gradient
    pred_out = nn_fn(in_data, auxvar)
    loss = dot((pred_out - out_data).reshape((-1, 1)), (pred_out - out_data).reshape((-1, 1)))
    loss_fn = Function('loss_fn', [auxvar], [loss])
    grad_loss = jacobian(loss, auxvar)
    grad_loss_fn = Function('grad_loss_fn', [auxvar], [grad_loss])

    # traning the loss function
    np.random.seed(seed)
    curr_auxvar = np.random.randn(n_auxvar)
    for k in range(max_iter):
        curr_loss = loss_fn(curr_auxvar)
        dloss = grad_loss_fn(curr_auxvar).full().flatten()
        curr_auxvar += -lr * dloss

        if display:
            if k % 1000 == 0: print('Iter #:', k, 'loss:', curr_loss)

    return curr_auxvar
