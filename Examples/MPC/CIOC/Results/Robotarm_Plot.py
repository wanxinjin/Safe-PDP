import numpy as np

from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random

# ------------------------ load the learning results ------------------------
load = np.load('./CIOC_Robotarm_trial_1.npy', allow_pickle=True).item()
gamma = load['gamma']
loss_trace_COC = load['loss_trace_COC']
loss_trace_barrierOC = load['loss_trace_barrierOC']
loss_trace_barrierOC2 = load['loss_trace_barrierOC2']

#  ------------------------ load demonstration data ------------------------
data = np.load('../../Demos/Robotarm_Demo.npy', allow_pickle=True).item()
dt = data['dt']
demo_storage = data['demos']

# -----------------------------  Load environment -----------------------------------------
env = JinEnv.RobotArm()
env.initDyn(g=0)
env_dyn = env.X + dt * env.f
env.initCost(wu=data['wu'])
true_parameter = [data['m1'], data['m2'], data['l1'], data['l1'], data['max_u'], data['max_q'], data['wq1'], data['wq2'], data['wdq1'],
                  data['wdq2'], ]
env.initConstraints()

#  ------------------------ create coc object to regenerate the trajectory based on the learned parameter
coc = SafePDP.COCsys()
coc.setAuxvarVariable(vertcat(env.dyn_auxvar, env.constraint_auxvar, env.cost_auxvar))
print(coc.auxvar)
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(env_dyn)
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
coc.setPathInequCstr(env.path_inequ)
coc.diffCPMP()
coc.convert2BarrierOC(gamma=gamma)

#  ------------------------  create the EQCLQR solver (if there is the need) ------------------------
clqr = SafePDP.EQCLQR()

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
ax1 = fig.subplots(1, 1)


line_COC, = ax1.plot(loss_trace_COC[0:], color='tab:green', linewidth=4)
line_barrierOC, = ax1.plot(loss_trace_barrierOC, color='tab:blue', linewidth=5,)
line_barrierOC2, = ax1.plot(loss_trace_barrierOC2, color='tab:red', linewidth=4, )
ax1.legend([line_COC, line_barrierOC, line_barrierOC2],
           ['Thm 1', r'Thm 2 for $\xi_{\theta}$ and  $\frac{\partial \xi_{\theta}}{\partial \theta}$',
            r'Thm 2 only for $\frac{\partial \xi_{\theta}}{\partial \theta}$'], ncol=1, prop={'size': 18},
           columnspacing=0.5,
           handlelength=1,
           loc='upper right').set_zorder(-102)

ax1.set_xlabel('Iteration', labelpad=0)
ax1.set_ylabel('Reproducing loss', labelpad=0)
ax1.set_facecolor('#E6E6E6')
ax1.grid()
ax1.set_xlim([-2, 60])
# ax1.set_ylim([-0.5,600])
ax1.set_xticks(np.arange(0, 61, 20))
ax1.set_position([0.16, 0.13, 0.80, 0.78])
plt.show()

