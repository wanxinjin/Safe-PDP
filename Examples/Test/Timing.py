import numpy as np

from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# -----------------------------  Load environment -----------------------------------------
env = JinEnv.CartPole()
env.initDyn()
dt = 0.1
dyn = env.X + dt * env.f
env.initCost(wu=0.1)
env.initConstraints()

# ----------------------------create tunable coc object-----------------------
coc = SafePDP.COCsys()
# pass the system to coc
coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar, env.cost_auxvar))
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(dyn)
# pass cost to coc
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
# pass constraints to coc
coc.setPathInequCstr(env.path_inequ)
# differentiating CPMP
coc.diffCPMP()
# convert to the unconstrained barrier OC object
gamma = 1e-1
coc.convert2BarrierOC(gamma=gamma)

# ----------------------------create the EQCLQR solver (if there is the need) --------------
clqr = SafePDP.EQCLQR()

# --------------------------- ----------------------
# print tunable parameter in the system
print(coc.auxvar)
# given a value to the parameter
parameter = [0.5, 0.5, 1, 5.0, 0.8, 0.1, 1, 0.1, 0.1]
horizon = 100
init_state = [0, 0, 0, 0]

# ------------------- start record the processing time for different time horizon -------------
horizons = [20, 40, 60, 80, 100]

strategy1_forward = []
strategy1_backward = []
strategy2_forward = []
strategy2_backward = []
for horizon in horizons:
    # Strategy 1:
    # use a COC solver to compute the trajectory and use Theorem 1 to compute trajectory derivative

    # timing for using a COC solver to computer trajectory
    start_time = time.time()
    traj_COC = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=parameter)
    strategy1_forward += [time.time() - start_time]

    # timing for establishing the auxiliary system to compute
    start_time = time.time()
    auxsys_COC = coc.getAuxSys(opt_sol=traj_COC, threshold=1e-5)
    clqr.auxsys2Eqctlqr(auxsys=auxsys_COC)
    aux_sol_COC = clqr.eqctlqrSolver(threshold=1e-5)
    strategy1_backward += [time.time() - start_time]

    # env.play_animation(pole_len=2, dt=dt, state_traj=traj_COC['state_traj_opt'])

    # Strategy 2:
    # use Theorem 2 to compute the trajectory  and also the trajectory derivative

    # timing for using a COC solver to computer trajectory
    start_time = time.time()
    traj_barrierOC = coc.solveBarrierOC(horizon=horizon, init_state=init_state,
                                        auxvar_value=parameter)
    strategy2_forward += [time.time() - start_time]

    # timing for establishing the auxiliary system to compute
    start_time = time.time()
    aux_sol_barrierOC = coc.auxSysBarrierOC(opt_sol=traj_barrierOC)
    strategy2_backward += [time.time() - start_time]

    # env.play_animation(pole_len=2, dt=dt, state_traj=traj_barrierOC['state_traj_opt'])

print(strategy1_forward)
print(strategy1_backward)
print(strategy2_forward)
print(strategy2_backward)


#  ------------------------ plot the result  ------------------------
# basic setting
params = {'axes.labelsize': 28,
          'axes.titlesize': 28,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# plot the loss
fig = plt.figure(0, figsize=(5.5, 5.5))
ax = fig.subplots(1, 1)

labels = ['20', '40', '60', '80', '100']
x = np.arange(len(labels))
width = 0.3
bar1 = ax.bar(x - width/2, strategy1_forward, width, label='Theorem 1 forward', color='tab:blue')
bar2 = ax.bar(x - width/2, strategy1_backward, width, label='Theorem 1 backward', color='tab:cyan')
bar3 = ax.bar(x+ width/2, strategy2_forward, width, label='Theorem 2 forward', color='tab:red')
bar4 = ax.bar(x + width/2, strategy2_backward, width, label='Theorem 2 backward', color='tab:orange')
ax.legend()

ax.set_ylabel('Time per iteration [s]')
ax.set_xlabel('Time horizon T')
ax.set_facecolor('#E6E6E6')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid()

ax.set_position([0.23, 0.14, 0.76, 0.78])
plt.show()
