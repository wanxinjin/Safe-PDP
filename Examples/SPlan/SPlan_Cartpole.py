import numpy as np

from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# --------------------------- load environment ----------------------------------------
env = JinEnv.CartPole()
mc, mp, l = 0.5, 0.5, 1
env.initDyn(mc=mc, mp=mp, l=l)
wx, wq, wdx, wdq, wu = 0.1, 1, 0.1, 0.1, 0.1
env.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)
max_x = 1
max_u = 4
env.initConstraints(max_u=4, max_x=max_x)
dt = 0.12
horizon = 25
init_state = [0, 0, 0, 0]
dyn = env.X + dt * env.f

# --------------------------- create Safe PDP SPlan object ----------------------------------------
planner = SafePDP.CSysOPT()
planner.setStateVariable(env.X)
planner.setControlVariable(env.U)
planner.setDyn(dyn)
planner.setPathCost(env.path_cost)
planner.setFinalCost(env.final_cost)
planner.setPathInequCstr(env.path_inequ)
gamma = 1e-2
planner.convert2BarrierOC(gamma=gamma)

# --------------------------- create COC object only for result comparison ----------------------------------------
coc = SafePDP.COCsys()
coc.setStateVariable(planner.state)
coc.setControlVariable(planner.control)
coc.setDyn(planner.dyn)
coc.setPathCost(planner.path_cost)
coc.setFinalCost(planner.final_cost)
coc.setPathInequCstr(planner.path_inequ_cstr)
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
# env.play_animation(pole_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'])
# plt.plot(coc_sol['control_traj_opt'], label='ct_control')
# plt.plot(coc_sol['state_traj_opt'][:, 0], label='ct_cart_pos')
# plt.fill_between(np.arange(0, horizon), 1, -1, color='red', alpha=0.2)
# plt.fill_between(np.arange(0, horizon), max_u, -max_u, color='green', alpha=0.2)
# plt.legend()
# plt.show()

# --------------------------- Safe Motion Planning ----------------------------------------
# set the policy as polynomial
n_poly = 10
planner.setPolyTraj(horizon=horizon, n_poly=n_poly)
# set the initial condition
nn_seed=None
init_parameter = np.zeros(planner.n_control_auxvar)  # all zeros initial condition
# nn_seed = 200 # e.g. 200,300, 400, 500
# init_parameter = 0.1*np.random.randn(planner.n_control_auxvar)  # random initial condition

# planning parameter setting
max_iter = 3000
loss_barrier_trace, loss_trace = [], []
parameter_trace = np.empty((max_iter, init_parameter.size))
control_traj, state_traj = 0, 0
lr = 1e-1

# start safe motion planning
current_parameter = init_parameter
for k in range(int(max_iter)):
    # one iteration of PDP
    loss_barrier, loss, dp, state_traj, control_traj, = planner.step(init_state=init_state, horizon=horizon,
                                                                     control_auxvar_value=current_parameter)
    # storage
    loss_barrier_trace += [loss_barrier]
    loss_trace += [loss]
    parameter_trace[k, :] = current_parameter

    # update
    current_parameter -= lr * dp

    # print
    if k % 100 == 0:
        print('Iter #:', k, 'Loss_barrier:', loss_barrier, 'Loss:', loss)

# save the results
if True:
    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'loss_barrier_trace': loss_barrier_trace,
                 'gamma': gamma,
                 'coc_sol': coc_sol,
                 'lr': lr,
                 'init_parameter': init_parameter,
                 'n_poly': n_poly,
                 'nn_seed': nn_seed}
    np.save('./Results/SPlan_Cartpole_trial_2.npy', save_data)

plt.plot(control_traj, label='SPDP_control')
plt.plot(coc_sol['control_traj_opt'], label='ct_control')
plt.plot(state_traj[:, 0], label='SPDP_cart_pos')
plt.plot(coc_sol['state_traj_opt'][:, 0], label='ct_cart_pos')
plt.fill_between(np.arange(0, horizon), max_x, -max_x, color='red', alpha=0.2)
plt.fill_between(np.arange(0, horizon), max_u, -max_u, color='green', alpha=0.2)
plt.legend()
plt.show()
env.play_animation(pole_len=2, dt=dt, state_traj=state_traj)
