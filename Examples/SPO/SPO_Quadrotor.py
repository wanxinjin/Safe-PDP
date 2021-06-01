from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# --------------------------- load environment ----------------------------------------
env = JinEnv.Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
env.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
env.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)
max_u = 12
max_r = 200
env.initConstraints(max_u=max_u, max_r=max_r)

dt = 0.15
horizon = 25
# set initial state
init_r_I = [-5, 5, 5.]
init_v_I = [0 - 5, -5., 0]
init_q = JinEnv.toQuaternion(0, [1, 0, 0])
init_w = [0.0, 0.0, 0.0]
init_state = init_r_I + init_v_I + init_q + init_w

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
# env.play_animation(wing_len=1.5, dt=dt, state_traj=coc_sol['state_traj_opt'])
# plt.plot(np.amax(np.abs(coc_sol['control_traj_opt']), axis=1), label='Control by COC')
# plt.show()

# --------------------------- Safe Policy Optimization ----------------------------------------
# set the neural policy
optimizer.setNeuralPolicy(hidden_layers=[15])
# initialize the policy by supervised learning from OC solution traj, a good initialization can avoid local minima
nn_seed = 200  # e.g. 200, 0, 5000, 3349, 937
init_parameter = SafePDP.Traning_NN(optimizer.neural_policy_fn, coc_sol['state_traj_opt'],
                                    0.5 * coc_sol['control_traj_opt'], display=False, max_iter=10000, lr=1e-3,
                                    seed=nn_seed)  # 0.5 is to make the control input small then the initial policy is more likely initially feasible,

current_parameter = init_parameter

# optimization parameter setting
max_iter = 2000
loss_barrier_trace, loss_trace = [], []
parameter_trace = np.empty((max_iter, init_parameter.size))
control_traj, state_traj = 0, 0
lr = 1e-2
# start policy optimization
for k in range(max_iter):

    # one iteration of safe policy optimization
    cost_barrier, cost, dp, state_traj, control_traj, = optimizer.step(init_state=init_state, horizon=horizon,
                                                                       control_auxvar_value=current_parameter,
                                                                       damping_flag=True, damping_lambda=100)
    # storage
    loss_barrier_trace += [cost_barrier]
    loss_trace += [cost]
    parameter_trace[k, :] = current_parameter

    # update
    current_parameter -= lr * dp

    # print
    if k % 5 == 0:
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
    np.save('./Results/SPO_Quadrotor_trial_2.npy', save_data)
#
# check and visualize the learning result
env.play_animation(wing_len=1.5, dt=dt, state_traj=state_traj)
plt.plot(np.amax(np.abs(coc_sol['control_traj_opt']), axis=1), label='Control by COC')
plt.plot(np.amax(np.abs(control_traj), axis=1), label='Control by Neural Policy')
plt.legend()
plt.show()
