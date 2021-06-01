import numpy as np

from ControlTools import ControlTools

from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random


# --------------------------- load environment ----------------------------------------
env = JinEnv.RobotArm()
env.initDyn(m1=1, m2=1, l1=1, l2=1, g=0)
env.initCost(wq1=0.1, wq2=0.1, wdq1=0.1, wdq2=0.1, wu=0.01)
env.initConstraints(max_u=1., max_q=pi)
dt = 0.2
horizon = 25
init_state = [-pi / 2, 3 * pi / 4, 0, 0]
dyn = env.X + dt * env.f

# --------------------------- create COC object only for result comparison ----------------------------------------
coc = SafePDP.COCsys()
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(dyn)
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
coc.setPathInequCstr(env.path_inequ)
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
# env.play_animation(l1=1, l2=1, dt=dt, state_traj=coc_sol['state_traj_opt'])
# plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
# plt.show()


# --------------------------- create Altro object ----------------------------------------
altro = ControlTools.ALTRO()
altro.setStateVariable(env.X)
altro.setControlVariable(env.U)
altro.setDyn(dyn)
altro.setPathCost(env.path_cost)
altro.setFinalCost(env.final_cost)
altro.setPathConstraint(env.path_inequ)
altro.diffSys()

# create the LQR solver
lqrsolver = ControlTools.LQR()


# --------------------------- planning start here----------------------------------------
# penalty parameter
base_mu=1

# initialize the
control_traj = np.zeros((horizon, altro.n_control))

# initialize the dual variable and penalty
lam_traj, mu_traj = altro.initDual(ini_state=init_state, control_traj=control_traj, base_mu=base_mu)
lr = 1e-3
max_outer = 20
max_inner = 50


# initialize the storage
control_traj_trace = np.empty((max_outer*max_inner, control_traj.shape[0], control_traj.shape[1]))
loss_trace = []

# outer to update the dual
for j in range(max_outer):

    # inner to solve the iLQR
    cost = 0
    al_cost = 0
    constraint_traj = 0

    for i in range(max_inner):
        # one step
        du_traj, constraint_traj, cost, al_cost = altro.stepILQR(ini_state=init_state, control_traj=control_traj,
                                                                 lam_traj=lam_traj, mu_traj=mu_traj,
                                                                 lqr_solver=lqrsolver)
        # update control trajectory
        for t in range(horizon):
            control_traj[t, :] = control_traj[t, :] + lr * du_traj[t].flatten()

        # store the cost
        loss_trace += [cost]
        # store the trajectory
        control_traj_trace[j*max_inner+i, :, :] = control_traj

    # update the dual variable
    lam_traj, mu_traj = altro.updateDual(lam_traj=lam_traj, mu_traj=mu_traj, constraint_traj=constraint_traj, base_mu=base_mu)

    # print
    print('iter #', j * max_inner, 'al_cost:', al_cost, 'loss:', cost)


# save the results
if True:
    save_data = {'control_traj_trace': control_traj_trace,
                 'loss_trace': loss_trace,
                 'coc_sol': coc_sol,
                 'max_outer': max_outer,
                 'max_inner': max_inner,
                 'lr': lr}
    np.save('./Results/ALTRO_Robotarm_trial_1.npy', save_data)


# for u_traj in control_traj_trace:
#     plt.plot(u_traj)
#     sol=altro.integrateSys(ini_state=init_state,control_traj=u_traj)
#     plt.plot(sol['state_traj'][:,0])
# plt.show()
#
# plt.plot(loss_trace)
# plt.show()
