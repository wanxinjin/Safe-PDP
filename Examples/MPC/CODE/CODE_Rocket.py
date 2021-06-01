import numpy as np

from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# --------------------------- load demonstration data ----------------------------------------
load = np.load('../Demos/Rocket_Demo.npy', allow_pickle=True).item()
dt = load['dt']
demo_storage = load['demos']

# -----------------------------  Load environment -----------------------------------------
env = JinEnv.Rocket()
env.initDyn()
env_dyn = env.X + dt * env.f
env.initCost(wr=load['wr'], wv=load['wv'], wtilt=load['wtilt'], ww=load['ww'], wsidethrust=load['wsidethrust'],
             wthrust=0.4)
true_parameter = [load['Jx'], load['Jy'], load['Jz'], load['mass'], load['l'], load['max_f_sq'], load['max_tilt_angle']]
env.initConstraints()

# ----------------------------create tunable coc object-----------------------
coc = SafePDP.COCsys()
# pass the system to coc
coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar, env.cost_auxvar))
print(coc.auxvar)
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(env_dyn)
# pass cost to coc
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
# pass constraints to coc
coc.setPathInequCstr(env.path_inequ)
# differentiating CPMP
coc.diffCPMP()
# convert to the unconstrained OC object
gamma = 1
coc.convert2BarrierOC(gamma=gamma)

# ----------------------------create the EQCLQR solver (if there is the need) --------------
clqr = SafePDP.EQCLQR()

# ----------------------------main learning procedure ----------------------
sigma = 0.2
nn_seed = 100
np.random.seed(nn_seed)
init_parameter = true_parameter + sigma * np.random.random(len(true_parameter))

# learning rate
lr = 1e-4
max_iter = 60

# initialize the storage
loss_trace_COC = []  # use COC solver to computer trajectory and use theorem 1 to compute the trajectory derivative
parameter_trace_COC = np.empty((max_iter, coc.n_auxvar))
loss_trace_barrierOC = []  # use theorem 2 to approximate both the system trajectory and its derivative
parameter_trace_barrierOC = np.empty((max_iter, coc.n_auxvar))
loss_trace_barrierOC2 = []  # use COC solver to computer trajectory and theorem 2 to approximate the trajectory derivative
parameter_trace_BarrierOC2 = np.empty((max_iter, coc.n_auxvar))

# To protect from the case where the trajectory is not differentiable, usually in such a case, our experience is that
# the output trajectory (i.e., the derivative of the trajectory) from auxiliary system would have spikes. This case
# is rarely happen, but when it happens, we simply let the current trajectory derivative equal to the one in previous
# iteration.

grad_protection_threshold = 1e5
previous_grad_COC = 0
previous_grad_barrierOC = 0
previous_grad_barrierOC2 = 0

current_parameter_COC = init_parameter
current_parameter_barrierOC = init_parameter
current_parameter_barrierOC2 = init_parameter

for k in range(max_iter):
    # batch for the constrained system
    batch_loss_COC = 0
    batch_grad_COC = 0
    # batch for the penalty system
    batch_loss_barrierOC = 0
    batch_grad_barrierOC = 0
    # batch for the penalty system (type 2)
    batch_loss_barrierOC2 = 0
    batch_grad_barrierOC2 = 0

    batch_size = len(demo_storage)
    for i in range(batch_size):
        # fetch the data sample
        demo = demo_storage[i]
        init_state = demo['state_traj_opt'][0, :]
        horizon = demo['control_traj_opt'].shape[0]

        # Strategy 1：
        # use COC solver to computer trajectory and use theorem 1 to compute the trajectory derivative
        traj_COC = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=current_parameter_COC)
        auxsys_COC = coc.getAuxSys(opt_sol=traj_COC, threshold=1e-5)
        clqr.auxsys2Eqctlqr(auxsys=auxsys_COC)
        aux_sol_COC = clqr.eqctlqrSolver(threshold=1e-5)
        # define the loss function and compute the loss function gradient using the chain rull
        loss_COC, grad_COC = SafePDP.Traj_L2_Loss(demo, traj_COC, aux_sol_COC)
        batch_loss_COC += loss_COC
        batch_grad_COC += grad_COC

        # Strategy 2：
        # use theorem 2 to approximate both the system trajectory and its derivative
        traj_barrierOC = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                            auxvar_value=current_parameter_barrierOC)
        aux_sol_barrierOC = coc.auxSysBarrierOC(opt_sol=traj_barrierOC)
        loss_barrierOC, grad_barrierOC = SafePDP.Traj_L2_Loss(demo, traj_barrierOC, aux_sol_barrierOC)
        batch_loss_barrierOC += loss_barrierOC
        batch_grad_barrierOC += grad_barrierOC

        # Strategy 3:
        # use COC solver to computer trajectory and theorem 2 to approximate the trajectory derivative
        traj_COC = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=current_parameter_barrierOC2)
        traj_barrierOC2 = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                             auxvar_value=current_parameter_barrierOC2)
        aux_sol_barrierOC2 = coc.auxSysBarrierOC(opt_sol=traj_barrierOC2)
        loss_barrierOC2, grad_barrierOC2 = SafePDP.Traj_L2_Loss(demo, traj_COC, aux_sol_barrierOC2)
        batch_loss_barrierOC2 += loss_barrierOC2
        batch_grad_barrierOC2 += grad_barrierOC2

    # protect the non-differentiable case for Strategy 1
    if norm_2(batch_grad_COC) > grad_protection_threshold:
        batch_grad_COC = previous_grad_COC
    else:
        previous_grad_COC = batch_grad_COC
    # protect the non-differentiable case for Strategy 2
    if norm_2(batch_grad_barrierOC) > grad_protection_threshold:
        batch_grad_barrierOC = previous_grad_barrierOC
    else:
        previous_grad_barrierOC = batch_grad_barrierOC
    # protect the non-differentiable case for Strategy 3
    if norm_2(batch_grad_barrierOC2) > grad_protection_threshold:
        batch_grad_barrierOC2 = previous_grad_barrierOC2
    else:
        previous_grad_barrierOC2 = batch_grad_barrierOC2

    # storage
    loss_trace_COC += [batch_loss_COC]
    parameter_trace_COC[k] = current_parameter_COC
    loss_trace_barrierOC += [batch_loss_barrierOC]
    parameter_trace_barrierOC[k] = current_parameter_barrierOC
    loss_trace_barrierOC2 += [batch_loss_barrierOC2]
    parameter_trace_BarrierOC2[k] = current_parameter_barrierOC2

    # print
    np.set_printoptions(suppress=True)
    print('iter #:', k, ' loss_COC:', batch_loss_COC, ' loss_barrierOC:', batch_loss_barrierOC,
          ' loss_barrierOC2:', batch_loss_barrierOC2)

    # update
    current_parameter_COC = current_parameter_COC - lr * batch_grad_COC
    current_parameter_barrierOC = current_parameter_barrierOC - lr * batch_grad_barrierOC
    current_parameter_barrierOC2 = current_parameter_barrierOC2 - lr * batch_grad_barrierOC2

# save
if True:
    save_data = {'parameter_trace_COC': parameter_trace_COC,
                 'loss_trace_COC': loss_trace_COC,
                 'parameter_trace_barrierOC': parameter_trace_barrierOC,
                 'loss_trace_barrierOC': loss_trace_barrierOC,
                 'parameter_trace_BarrierOC2': parameter_trace_BarrierOC2,
                 'loss_trace_barrierOC2': loss_trace_barrierOC2,
                 'gamma': gamma,
                 'nn_seed': nn_seed,
                 'lr': lr,
                 'init_parameter': init_parameter,
                 'true_parameter': true_parameter}
    np.save('./Results/CODE_Rocket_trial_1.npy', save_data)

