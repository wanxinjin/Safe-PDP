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
env = JinEnv.Rocket()
env.initDyn(Jx=0.5, Jy=1., Jz=1., mass=1., l=1.)
env.initCost(wr=10, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.4)
max_f_sq = 20**2
max_tilt_angle = 0.3
env.initConstraints(max_f_sq=max_f_sq, max_tilt_angle=max_tilt_angle)

dt = 0.1
horizon = 40
dyn = env.X + dt * env.f
# initial condition
init_r_I = [10, -8, 5.]
init_v_I = [-.1, 0.0, -0.0]
init_q = JinEnv.toQuaternion(0, [1, 0, 0])
init_w = [0, 0.0, 0.0]

init_state = init_r_I + init_v_I + init_q + init_w

# --------------------------- create Safe PDP SPlan object ----------------------------------------
planner = SafePDP.CSysOPT()
planner.setStateVariable(env.X)
planner.setControlVariable(env.U)
planner.setDyn(dyn)
planner.setPathCost(env.path_cost)
planner.setFinalCost(env.final_cost)
planner.setPathInequCstr(env.path_inequ)
gamma = 0  ###### THIS IS VERY IMPORTANT: gamma = 0  means that we do not add any barrier at all!
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
# env.play_animation(rocket_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'], control_traj=coc_sol['control_traj_opt'])
# plt.plot(np.linalg.norm(coc_sol['control_traj_opt'],2,1), label='ct_control')
# plt.legend()
# plt.show()
# # plt.plot(env.getTiltAngle(coc_sol['state_traj_opt']), label='ct_tilt_angle')
# # # plt.fill_between(np.arange(0, horizon), 1, -1, color='red', alpha=0.2)
# # # plt.fill_between(np.arange(0, horizon), max_u, -max_u, color='green', alpha=0.2)
# # plt.legend()
# # plt.show()

# --------------------------- Safe Motion Planing ----------------------------------------
# set the policy as polynomial
n_poly = 10
planner.setPolyTraj(horizon=horizon, n_poly=n_poly)
# set the initial condition
nn_seed = None
init_parameter = np.zeros(planner.n_control_auxvar)  # all zeros initial condition
# nn_seed = 500 # 200,300, 400, 500
# init_parameter = 0.3*np.random.randn(planner.n_control_auxvar)  # random initial condition

# planning parameter setting
max_iter = 2000
loss_barrier_trace, loss_trace = [], []
parameter_trace = np.empty((max_iter, init_parameter.size))
control_traj, state_traj = 0, 0
lr = 1e-2

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
    np.save('./Results/Plan_Rocket_trial_1.npy', save_data)

plt.plot(control_traj, label='SPDP_control')
plt.plot(coc_sol['control_traj_opt'], label='ct_control')
plt.legend()
plt.show()
plt.plot(env.getTiltAngle(coc_sol['state_traj_opt']), label='ct_tilt_angle')
plt.plot(env.getTiltAngle(state_traj), label='ct_tilt_angle')
plt.legend()
plt.show()
env.play_animation(rocket_len=2, dt=dt, state_traj=state_traj, control_traj=control_traj)
