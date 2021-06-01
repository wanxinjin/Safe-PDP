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
wx, wq, wdx, wdq, wu = 0.1, 1, 0.1, 0.1, 0.1
env.initCost(wq1=0.1, wq2=0.1, wdq1=0.1, wdq2=0.1, wu=0.01)
env.initConstraints(max_u=1., max_q=pi)
dt = 0.2
horizon = 25
init_state = [-pi / 2, 3 * pi / 4, 0, 0]

# --------------------------- create Safe PDP OPT object ----------------------------------------
optimizer = SafePDP.CSysOPT()
optimizer.setStateVariable(env.X)
optimizer.setControlVariable(env.U)
dyn = env.X + dt * env.f
optimizer.setDyn(dyn)
optimizer.setPathCost(env.path_cost)
optimizer.setFinalCost(env.final_cost)
optimizer.setPathInequCstr(env.path_inequ)
gamma = 0 ###### THIS IS VERY IMPORTANT: gamma = 0  means that we do not add any barrier at all!
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
# env.play_animation(l1=1, l2=1, dt=dt, state_traj=coc_sol['state_traj_opt'])
# plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
# plt.show()

# --------------------------- Safe Policy Optimization ----------------------------------------
# set the neural policy
optimizer.setNeuralPolicy(hidden_layers=[4])
# initialize the policy by supervised learning from OC solution traj, a good initialization can avoid local minima
nn_seed = 100
init_parameter = SafePDP.Traning_NN(optimizer.neural_policy_fn, coc_sol['state_traj_opt'],
                                    0.60 * coc_sol['control_traj_opt'], display=False, max_iter=10000,
                                    seed=nn_seed)  # make the initial condition is the same with the Safe PDP one
current_parameter = init_parameter

# optimization parameter setting
max_iter = 2000
loss_barrier_trace, loss_trace = [], []
parameter_trace = np.empty((max_iter, init_parameter.size))
control_traj, state_traj = 0, 0
lr = 0.5e-2
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
    np.save('./Results/PO_Robotarm_trial_1.npy', save_data)

# check and visualize the learning result
env.play_animation(l1=1, l2=1, dt=dt, state_traj=state_traj)
plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
plt.plot(control_traj, label='Control by Neural Policy')
plt.legend()
plt.show()
