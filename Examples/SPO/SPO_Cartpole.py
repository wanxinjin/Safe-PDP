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
max_u=4
max_x=5
env.initConstraints(max_u=max_u, max_x=max_x)
dt = 0.15
horizon = 20
init_state = [0, 0, 0, 0]

# --------------------------- create Safe PDP OPT object ----------------------------------------
optimizer = SafePDP.CSysOPT()
optimizer.setStateVariable(env.X)
optimizer.setControlVariable(env.U)
dyn = env.X + dt * env.f
optimizer.setDyn(dyn)
optimizer.setPathCost(env.path_cost)
optimizer.setFinalCost(env.final_cost)
optimizer.setPathInequCstr(env.path_inequ)
gamma = 1e-2
optimizer.convert2BarrierOC(gamma=gamma)

# ----create constrained OC object for result comparison and neural policy initialization ---------------
coc = SafePDP.COCsys()
coc.setStateVariable(optimizer.state)
coc.setControlVariable(optimizer.control)
coc.setDyn(optimizer.dyn)
coc.setPathCost(optimizer.path_cost)
coc.setFinalCost(optimizer.final_cost)
coc.setPathInequCstr(optimizer.path_inequ_cstr)
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
# env.play_animation(pole_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'])
# plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
# plt.plot(coc_sol['state_traj_opt'][:,0], label='position by COC')
# # plt.fill_between(np.arange(horizon), max_u, -max_u, color='red', alpha=0.1)
# # plt.fill_between(np.arange(horizon), max_x, -max_x, color='blue', alpha=0.1)
# plt.show()

# --------------------------- Safe Policy Optimization ----------------------------------------
# set the neural policy
optimizer.setNeuralPolicy(hidden_layers=[4])
# initialize the policy by supervised learning from OC solution traj, a good initialization can avoid local minima
nn_seed = 100  # e.g. 100,200,300,600,800
init_parameter = SafePDP.Traning_NN(optimizer.neural_policy_fn, coc_sol['state_traj_opt'],
                                    0.85 * coc_sol['control_traj_opt'], display=False, max_iter=10000,
                                    seed=nn_seed)  # 0.85 is to make the control input small then the initial policy is more likely initially feasible,

current_parameter = init_parameter

# optimization parameter setting
max_iter = 3000
loss_barrier_trace, loss_trace = [], []
parameter_trace = np.empty((max_iter, init_parameter.size))
control_traj, state_traj = 0, 0
lr = 0.8e-2
# start policy optimization
for k in range(max_iter):

    # one iteration of safe policy optimization
    cost_barrier, cost, dp, state_traj, control_traj, = optimizer.step(init_state=init_state, horizon=horizon,
                                                                       control_auxvar_value=current_parameter,
                                                                       damping_flag=True, damping_lambda=1)
    # storage
    loss_barrier_trace += [cost_barrier]
    loss_trace += [cost]
    parameter_trace[k, :] = current_parameter

    # update
    current_parameter -= lr * dp

    # print
    if k % 100 == 0:
        print('Iter #:', k, 'Loss_barrier:', cost_barrier, 'Loss:', cost)

# save the results
if True:
    save_data = {'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'loss_barrier_trace': loss_barrier_trace,
                 'gamma': gamma,
                 'coc_sol': coc_sol,
                 'lr': lr,
                 'init_parameter': init_parameter,
                 'nn_seed': nn_seed}
    np.save('./Results/SPO_Cartpole_trial_2.npy', save_data)

# check and visualize the learning result
env.play_animation(pole_len=2, dt=dt, state_traj=state_traj)
plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
plt.plot(control_traj, label='Control by Neural Policy')
plt.legend()
plt.show()
