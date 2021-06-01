"""
# This module provides some useful tools and benchmark methods for solving control problems
# This module is separate from PDP module, which means that PDP modules can be  used independently of this module
# This module aims to include the most common control tools coded in Python.
# Some code block is overlapped with ones in the PDP module

# Do NOT distribute without written permission from Wanxin Jin
# Do NOT use it for any commercial purpose

# Contact email: wanxinjin@gmail.com
# Last update: Apr. 19, 2021
"""
import numpy as np
from casadi import *
import numpy
import time

'''
# =============================================================================================================
# This function is used to implement iterative LQR algorithm
'''


class iLQR:
    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        # assign the dynamics
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control], [self.dyn])

        # linearize the dynamics
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control], [jacobian(self.dyn, self.state)])
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control], [jacobian(self.dyn, self.control)])

    def setPathCost(self, path_cost):
        assert path_cost.numel() == 1, "path_cost must be a scalar function"
        # assign the path cost
        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control], [self.path_cost])

        # linearize the path cost
        dcx = jacobian(self.path_cost, self.state).T
        dcu = jacobian(self.path_cost, self.control).T
        self.dcx_fn = casadi.Function('dcx', [self.state, self.control], [dcx])
        self.dcu_fn = casadi.Function('dux', [self.state, self.control], [dcu])

        # hessian matrix of path cost
        ddcxx = jacobian(dcx, self.state)
        self.ddcxx_fn = casadi.Function('ddcxx', [self.state, self.control], [ddcxx])
        ddcuu = jacobian(dcu, self.control)
        self.ddcuu_fn = casadi.Function('ddcuu', [self.state, self.control], [ddcuu])
        ddcxu = jacobian(dcx, self.control)
        self.ddcxu_fn = casadi.Function('ddcxu', [self.state, self.control], [ddcxu])
        ddcux = jacobian(dcu, self.state)
        self.ddcux_fn = casadi.Function('ddcux', [self.state, self.control], [ddcux])

    def setFinalCost(self, final_cost):
        assert final_cost.numel() == 1, "final_cost must be a scalar function"
        # assign final cost function
        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.state], [self.final_cost])

        # linearize the final cost
        dhx = jacobian(self.final_cost, self.state).T
        self.dhx_fn = casadi.Function('dhx', [self.state], [dhx])

        # hessian matrix of final cost
        ddhxx = jacobian(dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx', [self.state], [ddhxx])

    def integrateSys(self, ini_state, control_traj):
        assert type(control_traj) is numpy.ndarray, "Control sequence must be of the numpy.ndarray type"

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        horizon = numpy.size(control_traj, 0)

        # initialization
        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        cost = 0
        for t in range(horizon):
            state_traj[t + 1, :] = self.dyn_fn(state_traj[t, :], control_traj[t, :]).full().flatten()
            cost += self.path_cost_fn(state_traj[t, :], control_traj[t, :]).full()
        cost += self.final_cost_fn(state_traj[-1, :]).full()

        sol = {'state_traj': state_traj,
               'control_traj': control_traj,
               'cost': cost.item()}
        return sol

    def getLQRApproximation(self, state_traj, control_traj):

        horizon = numpy.size(control_traj, 0)

        # initialization
        Fx, Fu = [], []
        Cxx, Cuu, Cxu, Cux, Cx, Cu = [], [], [], [], [], []
        Hxx, Hx = [], []

        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            Fx += [self.dfx_fn(curr_x, curr_u).full()]
            Fu += [self.dfu_fn(curr_x, curr_u).full()]
            Cx += [self.dcx_fn(curr_x, curr_u).full()]
            Cu += [self.dcu_fn(curr_x, curr_u).full()]
            Cxx += [self.ddcxx_fn(curr_x, curr_u).full()]
            Cuu += [self.ddcuu_fn(curr_x, curr_u).full()]
            Cxu += [self.ddcxu_fn(curr_x, curr_u).full()]
            Cux += [self.ddcux_fn(curr_x, curr_u).full()]

        Hxx += [self.ddhxx_fn(state_traj[-1, :]).full()]
        Hx += [self.dhx_fn(state_traj[-1, :]).full()]

        approx_lqr_dyn = {'Fx': Fx,
                          'Fu': Fu}
        approx_lqr_costs = {'Cxx': Cxx,
                            'Cuu': Cuu,
                            'Cux': Cux,
                            'Cxu': Cxu,
                            'Cx': Cx,
                            'Cu': Cu,
                            'Hxx': Hxx,
                            'Hx': Hx
                            }

        return approx_lqr_dyn, approx_lqr_costs

    def step(self, ini_state, control_traj, lqr_solver):

        # horizon
        horizon = control_traj.shape[0]

        # generate the system trajectory
        sol = self.integrateSys(ini_state=ini_state, control_traj=control_traj)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        loss = sol['cost']

        # generate the LQR approximation
        lqr_dyn, lqr_cost = self.getLQRApproximation(state_traj=state_traj, control_traj=control_traj)

        # solve the LQR
        lqr_solver.setDyn(dynF=lqr_dyn['Fx'], dynG=lqr_dyn['Fu'])
        lqr_solver.setPathCost(Hxx=lqr_cost['Cxx'], Huu=lqr_cost['Cuu'], Hxu=lqr_cost['Cxu'], Hux=lqr_cost['Cux'],
                               Hxe=lqr_cost['Cx'], Hue=lqr_cost['Cu'])
        lqr_solver.setFinalCost(hxx=lqr_cost['Hxx'], hxe=lqr_cost['Hx'])
        sol = lqr_solver.lqrSolver(ini_state=numpy.zeros(self.n_state), horizon=horizon)

        return loss, sol['control_traj_opt']


'''
# =============================================================================================================
# The LQR class is mainly for solving (time-varying or time-invariant) LQR problems.
# The standard form of the dynamics in the LQR system is
# X_k+1=dynF_k*X_k+dynG_k*U_k+dynE_k,
# where matrices dynF_k, dynG_k, and dynE_k are system dynamics matrices you need to specify (maybe time-varying)
# The standard form of cost function for the LQR system is
# J=sum_0^(horizon-1) path_cost + final cost, where
# path_cost  = trace (1/2*X'*Hxx*X +1/2*U'*Huu*U + 1/2*X'*Hxu*U + 1/2*U'*Hux*X + Hue'*U + Hxe'*X)
# final_cost = trace (1/2*X'*hxx*X +hxe'*X)
# Here, Hxx, Huu, Hux, Hxu, Heu, Hex, hxx, hex are cost matrices you need to specify (maybe time-varying).
# Some of the above dynamics and cost matrices, by default, are zero (none) matrices
# Note that the variable X and variable U can be matrix variables.
# The above defined standard form is consistent with the auxiliary control system defined in the PDP paper
'''


class LQR:

    def __init__(self, project_name="LQR system"):
        self.project_name = project_name

    def setDyn(self, dynF, dynG, dynE=None):
        if type(dynF) is numpy.ndarray:
            self.dynF = [dynF]
            self.n_state = numpy.size(dynF, 0)
        elif type(dynF[0]) is numpy.ndarray:
            self.dynF = dynF
            self.n_state = numpy.size(dynF[0], 0)
        else:
            assert False, "Type of dynF matrix should be numpy.ndarray  or list of numpy.ndarray"

        if type(dynG) is numpy.ndarray:
            self.dynG = [dynG]
            self.n_control = numpy.size(dynG, 1)
        elif type(dynG[0]) is numpy.ndarray:
            self.dynG = dynG
            self.n_control = numpy.size(self.dynG[0], 1)
        else:
            assert False, "Type of dynG matrix should be numpy.ndarray  or list of numpy.ndarray"

        if dynE is not None:
            if type(dynE) is numpy.ndarray:
                self.dynE = [dynE]
                self.n_batch = numpy.size(dynE, 1)
            elif type(dynE[0]) is numpy.ndarray:
                self.dynE = dynE
                self.n_batch = numpy.size(dynE[0], 1)
            else:
                assert False, "Type of dynE matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.dynE = None
            self.n_batch = None

    def setPathCost(self, Hxx, Huu, Hxu=None, Hux=None, Hxe=None, Hue=None):

        if type(Hxx) is numpy.ndarray:
            self.Hxx = [Hxx]
        elif type(Hxx[0]) is numpy.ndarray:
            self.Hxx = Hxx
        else:
            assert False, "Type of path cost Hxx matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if type(Huu) is numpy.ndarray:
            self.Huu = [Huu]
        elif type(Huu[0]) is numpy.ndarray:
            self.Huu = Huu
        else:
            assert False, "Type of path cost Huu matrix should be numpy.ndarray or list of numpy.ndarray, or None"

        if Hxu is not None:
            if type(Hxu) is numpy.ndarray:
                self.Hxu = [Hxu]
            elif type(Hxu[0]) is numpy.ndarray:
                self.Hxu = Hxu
            else:
                assert False, "Type of path cost Hxu matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxu = None

        if Hux is not None:
            if type(Hux) is numpy.ndarray:
                self.Hux = [Hux]
            elif type(Hux[0]) is numpy.ndarray:
                self.Hux = Hux
            else:
                assert False, "Type of path cost Hux matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hux = None

        if Hxe is not None:
            if type(Hxe) is numpy.ndarray:
                self.Hxe = [Hxe]
            elif type(Hxe[0]) is numpy.ndarray:
                self.Hxe = Hxe
            else:
                assert False, "Type of path cost Hxe matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hxe = None

        if Hue is not None:
            if type(Hue) is numpy.ndarray:
                self.Hue = [Hue]
            elif type(Hue[0]) is numpy.ndarray:
                self.Hue = Hue
            else:
                assert False, "Type of path cost Hue matrix should be numpy.ndarray or list of numpy.ndarray, or None"
        else:
            self.Hue = None

    def setFinalCost(self, hxx, hxe=None):

        if type(hxx) is numpy.ndarray:
            self.hxx = [hxx]
        elif type(hxx[0]) is numpy.ndarray:
            self.hxx = hxx
        else:
            assert False, "Type of final cost hxx matrix should be numpy.ndarray or list of numpy.ndarray"

        if hxe is not None:
            if type(hxe) is numpy.ndarray:
                self.hxe = [hxe]
            elif type(hxe[0]) is numpy.ndarray:
                self.hxe = hxe
            else:
                assert False, "Type of final cost hxe matrix should be numpy.ndarray, list of numpy.ndarray, or None"
        else:
            self.hxe = None

    def lqrSolver(self, ini_state, horizon):

        # Data pre-processing
        n_state = numpy.size(self.dynF[0], 1)
        if type(ini_state) is list:
            self.ini_x = numpy.array(ini_state, numpy.float64)
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        elif type(ini_state) is numpy.ndarray:
            self.ini_x = ini_state
            if self.ini_x.ndim == 2:
                self.n_batch = numpy.size(self.ini_x, 1)
            else:
                self.n_batch = 1
                self.ini_x = self.ini_x.reshape(n_state, -1)
        else:
            assert False, "Initial state should be of numpy.ndarray type or list!"

        self.horizon = horizon

        if self.dynE is not None:
            assert self.n_batch == numpy.size(self.dynE[0],
                                              1), "Number of data batch is not consistent with column of dynE"

        # Check the time horizon
        if len(self.dynF) > 1 and len(self.dynF) != self.horizon:
            assert False, "time-varying dynF is not consistent with given horizon"
        elif len(self.dynF) == 1:
            F = self.horizon * self.dynF
        else:
            F = self.dynF

        if len(self.dynG) > 1 and len(self.dynG) != self.horizon:
            assert False, "time-varying dynG is not consistent with given horizon"
        elif len(self.dynG) == 1:
            G = self.horizon * self.dynG
        else:
            G = self.dynG

        if self.dynE is not None:
            if len(self.dynE) > 1 and len(self.dynE) != self.horizon:
                assert False, "time-varying dynE is not consistent with given horizon"
            elif len(self.dynE) == 1:
                E = self.horizon * self.dynE
            else:
                E = self.dynE
        else:
            E = self.horizon * [numpy.zeros(self.ini_x.shape)]

        if len(self.Hxx) > 1 and len(self.Hxx) != self.horizon:
            assert False, "time-varying Hxx is not consistent with given horizon"
        elif len(self.Hxx) == 1:
            Hxx = self.horizon * self.Hxx
        else:
            Hxx = self.Hxx

        if len(self.Huu) > 1 and len(self.Huu) != self.horizon:
            assert False, "time-varying Huu is not consistent with given horizon"
        elif len(self.Huu) == 1:
            Huu = self.horizon * self.Huu
        else:
            Huu = self.Huu

        hxx = self.hxx

        if self.hxe is None:
            hxe = [numpy.zeros(self.ini_x.shape)]

        if self.Hxu is None:
            Hxu = self.horizon * [numpy.zeros((self.n_state, self.n_control))]
        else:
            if len(self.Hxu) > 1 and len(self.Hxu) != self.horizon:
                assert False, "time-varying Hxu is not consistent with given horizon"
            elif len(self.Hxu) == 1:
                Hxu = self.horizon * self.Hxu
            else:
                Hxu = self.Hxu

        if self.Hux is None:  # Hux is the transpose of Hxu
            Hux = self.horizon * [numpy.zeros((self.n_control, self.n_state))]
        else:
            if len(self.Hux) > 1 and len(self.Hux) != self.horizon:
                assert False, "time-varying Hux is not consistent with given horizon"
            elif len(self.Hux) == 1:
                Hux = self.horizon * self.Hux
            else:
                Hux = self.Hux

        if self.Hxe is None:
            Hxe = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        else:
            if len(self.Hxe) > 1 and len(self.Hxe) != self.horizon:
                assert False, "time-varying Hxe is not consistent with given horizon"
            elif len(self.Hxe) == 1:
                Hxe = self.horizon * self.Hxe
            else:
                Hxe = self.Hxe

        if self.Hue is None:
            Hue = self.horizon * [numpy.zeros((self.n_control, self.n_batch))]
        else:
            if len(self.Hue) > 1 and len(self.Hue) != self.horizon:
                assert False, "time-varying Hue is not consistent with given horizon"
            elif len(self.Hue) == 1:
                Hue = self.horizon * self.Hue
            else:
                Hue = self.Hue

        # Solve the Riccati equations: the notations used here are consistent with Lemma 4.2 in the PDP paper
        I = numpy.eye(self.n_state)
        PP = self.horizon * [numpy.zeros((self.n_state, self.n_state))]
        WW = self.horizon * [numpy.zeros((self.n_state, self.n_batch))]
        PP[-1] = self.hxx[0]
        WW[-1] = self.hxe[0]
        for t in range(self.horizon - 1, 0, -1):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            HxuinvHuu = numpy.matmul(Hxu[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            Q_t = Hxx[t] - numpy.matmul(HxuinvHuu, numpy.transpose(Hxu[t]))
            N_t = Hxe[t] - numpy.matmul(HxuinvHuu, Hue[t])

            temp_mat = numpy.matmul(numpy.transpose(A_t), numpy.linalg.inv(I + numpy.matmul(P_next, R_t)))
            P_curr = Q_t + numpy.matmul(temp_mat, numpy.matmul(P_next, A_t))
            W_curr = N_t + numpy.matmul(temp_mat, W_next + numpy.matmul(P_next, M_t))

            PP[t - 1] = P_curr
            WW[t - 1] = W_curr

        # Compute the trajectory using the Raccti matrices obtained from the above: the notations used here are
        # consistent with the PDP paper in Lemma 4.2
        state_traj_opt = (self.horizon + 1) * [numpy.zeros((self.n_state, self.n_batch))]
        control_traj_opt = (self.horizon) * [numpy.zeros((self.n_control, self.n_batch))]
        costate_traj_opt = (self.horizon) * [numpy.zeros((self.n_state, self.n_batch))]
        state_traj_opt[0] = self.ini_x
        for t in range(self.horizon):
            P_next = PP[t]
            W_next = WW[t]
            invHuu = numpy.linalg.inv(Huu[t])
            GinvHuu = numpy.matmul(G[t], invHuu)
            A_t = F[t] - numpy.matmul(GinvHuu, numpy.transpose(Hxu[t]))
            M_t = E[t] - numpy.matmul(GinvHuu, Hue[t])
            R_t = numpy.matmul(GinvHuu, numpy.transpose(G[t]))

            x_t = state_traj_opt[t]
            u_t = -numpy.matmul(invHuu, numpy.matmul(numpy.transpose(Hxu[t]), x_t) + Hue[t]) \
                  - numpy.linalg.multi_dot([invHuu, numpy.transpose(G[t]), numpy.linalg.inv(I + numpy.dot(P_next, R_t)),
                                            (numpy.matmul(numpy.matmul(P_next, A_t), x_t) + numpy.matmul(P_next,
                                                                                                         M_t) + W_next)])

            x_next = numpy.matmul(F[t], x_t) + numpy.matmul(G[t], u_t) + E[t]
            lambda_next = numpy.matmul(P_next, x_next) + W_next

            state_traj_opt[t + 1] = x_next
            control_traj_opt[t] = u_t
            costate_traj_opt[t] = lambda_next
        time = [k for k in range(self.horizon + 1)]

        opt_sol = {'state_traj_opt': state_traj_opt,
                   'control_traj_opt': control_traj_opt,
                   'costate_traj_opt': costate_traj_opt,
                   'time': time}
        return opt_sol


'''
# =============================================================================================================
# This class is used for system identification, where the target dynamics models are linear and of the form
# X_k+1=A*X_k+B*U_k
# where the matrix A and matrix B are unknown, which are to be determined.
# The input are states-inputs trajectory.
'''

class SysID_DMD:

    def __init__(self, project_name='my idenfication'):
        self.project_name = project_name

    def setDimensions(self, n_state, n_control):
        # generate state variable
        self.X = SX.sym('X', n_state)
        self.n_state = self.X.numel()
        self.U = SX.sym('U', n_control)
        self.n_control = self.U.numel()
        A = SX.sym('A', n_state, n_state)
        B = SX.sym('B', n_state, n_control)

        # the unknown parameter to be determined
        parameter = [A.reshape((A.numel(), 1)), B.reshape((B.numel(), 1))]
        self.auxvar = vcat(parameter)
        self.n_auxvar = self.auxvar.numel()

        # this is the learner dynamics
        self.dyn_linear_fn = Function('dyn_linear', [self.X, self.U, self.auxvar],
                                      [mtimes(A, self.X) + mtimes(B, self.U)])

    def setIOData(self, states, controls):

        n_batch = len(states)
        state_input_pairs = []
        observed_states = []
        for i in range(n_batch):
            input_traj = controls[i]
            state_traj = states[i]
            horizon = np.size(input_traj, 0)
            for t in range(horizon):
                curr_x = state_traj[t, :].tolist()
                curr_u = input_traj[t, :].tolist()
                state_input_pairs += [curr_x + curr_u]
                next_x = state_traj[t + 1, :].tolist()
                observed_states += [next_x]
        state_input_pairs = numpy.array(state_input_pairs)
        observed_states = numpy.array(observed_states)

        states_data = numpy.transpose(state_input_pairs[:, 0:self.n_state])
        inputs_date = numpy.transpose(state_input_pairs[:, self.n_state:])
        observed_states = np.transpose(observed_states)

        # predicted states
        predicted_states = self.dyn_linear_fn(states_data, inputs_date, self.auxvar)
        predicted_error = predicted_states - observed_states

        # define loss function
        loss = dot(predicted_error, predicted_error) / n_batch
        self.loss_fn = Function('loss', [self.auxvar], [loss])
        self.grad_loss_fn = Function('loss', [self.auxvar], [jacobian(loss, self.auxvar)])


'''
# =============================================================================================================
# This class is python implementation of the paper 'ALTRO: A Fast Solver for Constrained Trajectory Optimization'
# http://roboticexplorationlab.org/papers/altro-iros.pdf
'''

class ALTRO:

    def __init__(self, project_name="constrained optimal control system"):
        self.project_name = project_name

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        # assign the dynamics
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynamics', [self.state, self.control], [self.dyn])

        # linearize the dynamics
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control], [jacobian(self.dyn, self.state)])
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control], [jacobian(self.dyn, self.control)])

    def setPathCost(self, path_cost):
        assert path_cost.numel() == 1, "path_cost must be a scalar function"
        # assign the path cost
        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('path_cost', [self.state, self.control], [self.path_cost])

    def setFinalCost(self, final_cost):
        assert final_cost.numel() == 1, "final_cost must be a scalar function"
        # assign final cost function
        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('final_cost', [self.state], [self.final_cost])

    def setPathConstraint(self, path_constraint=None):
        if path_constraint is None:
            self.n_path_constraint = 0
            self.path_constraint = SX.sym('path_constraint', 0)
        elif path_constraint.numel() == 1:
            self.n_path_constraint = 1
            self.path_constraint = path_constraint
        else:
            self.n_path_constraint = path_constraint.numel()
            self.path_constraint = path_constraint

        self.path_constraint_fn = Function('path_constraint_fn', [self.state, self.control], [self.path_constraint])

        self.lam_t = SX.sym('lam_t', self.n_path_constraint)
        self.mu_t = SX.sym('mu_t', self.n_path_constraint)

    def setFinalConstraint(self, final_constraint=None):
        if final_constraint is None:
            self.n_final_constraint = 0
            self.final_constraint = SX.sym('final_constraint', 0)
        elif final_constraint.numel() == 1:
            self.n_final_constraint = 1
            self.final_constraint = final_constraint
        else:
            self.n_final_constraint = final_constraint.numel()
            self.final_constraint = final_constraint

        self.final_constraint_fn = Function('final_constraint_fn', [self.state], [self.final_constraint])
        self.lam_T = SX.sym('lam_T', self.n_final_constraint)
        self.mu_T = SX.sym('mu_T', self.n_final_constraint)

    def diffSys(self):

        if not hasattr(self, 'lam_t'): self.setPathConstraint()
        if not hasattr(self, 'lam_T'): self.setFinalConstraint()

        if self.n_path_constraint == 0:
            self.new_path_cost = self.path_cost
        else:
            self.new_path_cost = self.path_cost + dot(self.lam_t, self.path_constraint) + 0.5 * dot(
                self.mu_t * self.path_constraint, self.path_constraint)

        self.new_path_cost_fn = Function('new_path_cost_fn', [self.state, self.control, self.lam_t, self.mu_t],
                                         [self.new_path_cost])

        # linearize the new path cost
        dlx = jacobian(self.new_path_cost, self.state).T
        dlu = jacobian(self.new_path_cost, self.control).T
        self.dlx_fn = casadi.Function('dlx_fn', [self.state, self.control, self.lam_t, self.mu_t], [dlx])
        self.dlu_fn = casadi.Function('dlu_fn', [self.state, self.control, self.lam_t, self.mu_t], [dlu])

        # hessian matrix of path cost
        ddlxx = jacobian(dlx, self.state)
        self.ddlxx_fn = casadi.Function('ddlxx_fn', [self.state, self.control, self.lam_t, self.mu_t], [ddlxx])
        ddluu = jacobian(dlu, self.control)
        self.ddluu_fn = casadi.Function('ddluu_fn', [self.state, self.control, self.lam_t, self.mu_t], [ddluu])
        ddlxu = jacobian(dlx, self.control)
        self.ddlxu_fn = casadi.Function('ddlxu_fn', [self.state, self.control, self.lam_t, self.mu_t], [ddlxu])
        ddlux = jacobian(dlu, self.state)
        self.ddlux_fn = casadi.Function('ddlux_fn', [self.state, self.control, self.lam_t, self.mu_t], [ddlux])

        if self.n_final_constraint == 0:
            self.new_final_cost = self.final_cost
        else:
            self.new_final_cost = self.final_cost + dot(self.lam_T, self.final_constraint) + 0.5 * dot(
                self.mu_T * self.final_constraint, self.final_constraint)

        self.new_final_cost_fn = Function('new_final_cost_fn', [self.state, self.lam_t, self.mu_t],
                                          [self.new_final_cost])

        # linearize the final cost
        dhx = jacobian(self.new_final_cost, self.state).T
        self.dhx_fn = casadi.Function('dhx_fn', [self.state, self.lam_T, self.mu_T], [dhx])

        # hessian matrix of final cost
        ddhxx = jacobian(dhx, self.state)
        self.ddhxx_fn = casadi.Function('ddhxx_fn', [self.state, self.lam_T, self.mu_T], [ddhxx])

    def integrateSys(self, ini_state, control_traj):
        assert type(control_traj) is numpy.ndarray, "Control sequence must be of the numpy.ndarray type"

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        horizon = numpy.size(control_traj, 0)

        # initialization
        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        constraint_traj = []
        cost = 0
        for t in range(horizon):
            constraint_traj += [self.path_constraint_fn(state_traj[t, :], control_traj[t, :]).full().flatten()]
            state_traj[t + 1, :] = self.dyn_fn(state_traj[t, :], control_traj[t, :]).full().flatten()
            cost += self.path_cost_fn(state_traj[t, :], control_traj[t, :]).full()
        cost += self.final_cost_fn(state_traj[-1, :]).full()
        constraint_traj += [self.final_constraint_fn(state_traj[-1, :]).full().flatten()]

        sol = {'state_traj': state_traj,
               'control_traj': control_traj,
               'constraint_traj': constraint_traj,
               'cost': cost.item()}
        return sol

    def getLQRApproximation(self, state_traj, control_traj, lam_traj, mu_traj):

        horizon = numpy.size(control_traj, 0)

        # initialization
        Fx, Fu = [], []
        Cxx, Cuu, Cxu, Cux, Cx, Cu = [], [], [], [], [], []
        Hxx, Hx = [], []

        al_cost_value = 0

        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            curr_lam = lam_traj[t]
            curr_mu = mu_traj[t]
            Fx += [self.dfx_fn(curr_x, curr_u).full()]
            Fu += [self.dfu_fn(curr_x, curr_u).full()]
            Cx += [self.dlx_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            Cu += [self.dlu_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            Cxx += [self.ddlxx_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            Cuu += [self.ddluu_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            Cxu += [self.ddlxu_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            Cux += [self.ddlux_fn(curr_x, curr_u, curr_lam, curr_mu).full()]
            al_cost_value += self.new_path_cost_fn(curr_x, curr_u, curr_lam, curr_mu).full()

        Hxx += [self.ddhxx_fn(state_traj[-1, :], lam_traj[-1], mu_traj[-1]).full()]
        Hx += [self.dhx_fn(state_traj[-1, :], lam_traj[-1], mu_traj[-1]).full()]
        al_cost_value += self.new_final_cost_fn(state_traj[-1, :], lam_traj[-1], mu_traj[-1]).full()

        approx_lqr_dyn = {'Fx': Fx,
                          'Fu': Fu}
        approx_lqr_costs = {'Cxx': Cxx,
                            'Cuu': Cuu,
                            'Cux': Cux,
                            'Cxu': Cxu,
                            'Cx': Cx,
                            'Cu': Cu,
                            'Hxx': Hxx,
                            'Hx': Hx
                            }

        return approx_lqr_dyn, approx_lqr_costs, al_cost_value

    def stepILQR(self, ini_state, control_traj, lam_traj, mu_traj, lqr_solver):

        # horizon
        horizon = control_traj.shape[0]

        # generate the system trajectory
        sol = self.integrateSys(ini_state=ini_state, control_traj=control_traj)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        constraint_traj = sol['constraint_traj']
        loss = sol['cost']

        # generate the LQR approximation
        lqr_dyn, lqr_cost, al_cost_value = self.getLQRApproximation(state_traj=state_traj, control_traj=control_traj,
                                                                    lam_traj=lam_traj, mu_traj=mu_traj)

        # solve the LQR
        lqr_solver.setDyn(dynF=lqr_dyn['Fx'], dynG=lqr_dyn['Fu'])
        lqr_solver.setPathCost(Hxx=lqr_cost['Cxx'], Huu=lqr_cost['Cuu'], Hxu=lqr_cost['Cxu'], Hux=lqr_cost['Cux'],
                               Hxe=lqr_cost['Cx'], Hue=lqr_cost['Cu'])
        lqr_solver.setFinalCost(hxx=lqr_cost['Hxx'], hxe=lqr_cost['Hx'])
        sol = lqr_solver.lqrSolver(ini_state=numpy.zeros(self.n_state), horizon=horizon)


        return sol['control_traj_opt'], constraint_traj, loss, al_cost_value

    def updateDual(self, lam_traj, mu_traj, constraint_traj, base_mu=5):

        new_lam_traj = []
        new_mu_traj = []

        # for t in range(len(lam_traj)):
        #     new_lam_traj += [fmax(0, lam_traj[t] + mu_traj[t] * constraint_traj[t]).full().flatten()]
        #     new_mu_traj += [phi_rate * mu_traj[t]]

        for t in range(len(lam_traj)):
            new_lam_traj += [fmax(0, lam_traj[t] + mu_traj[t] * constraint_traj[t]).full().flatten()]
            new_mu_traj += [np.where(constraint_traj[t] < 0, 0, base_mu)]

        return new_lam_traj, new_mu_traj

    def initDual(self, ini_state, control_traj, base_mu=5 ):


        # horizon
        horizon = control_traj.shape[0]

        # generate the system trajectory
        sol = self.integrateSys(ini_state=ini_state, control_traj=control_traj)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        constraint_traj = sol['constraint_traj']
        loss = sol['cost']

        lam_traj = []
        mu_traj = []
        for t in range(horizon):
            lam_traj += [np.zeros(self.n_path_constraint)]
            mu_traj += [np.where(constraint_traj[t] < 0, 0, base_mu)]

        lam_traj += [np.zeros(self.n_final_constraint)]
        mu_traj += [np.where(constraint_traj[-1] < 0, 0, base_mu)]

        return lam_traj, mu_traj
