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
env = JinEnv.RobotArm()
m1, m2, l1, l2 = 1, 1, 1, 1
env.initDyn(m1=m1, m2=m2, l1=l1, l2=l2, g=0)
wq1, wq2, wdq1, wdq2, wu = 0.1, 0.1, 0.1, 0.1, 0.01
env.initCost(wq1=wq1, wq2=wq2, wdq1=wdq1, wdq2=wdq2, wu=wu)
max_u = 1
max_q = pi
env.initConstraints(max_u=max_u, max_q=max_q)

dt = 0.2
dyn = env.X + dt * env.f

# --------create COC object for trajectory demo generation ---------------
coc = SafePDP.COCsys()
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
coc.setDyn(dyn)
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
coc.setPathInequCstr(env.path_inequ)

# -------- generating the demos ---------------
horizon = 25
init_state = [-pi / 2, 3 * pi / 4, 0, 0]
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
env.play_animation(l1=1, l2=1, dt=dt, state_traj=coc_sol['state_traj_opt'])
plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
plt.fill_between(np.arange(0, horizon), max_u, -max_u, color='red', alpha=0.2)
plt.show()

demos = [
    {'state_traj_opt': coc_sol['state_traj_opt'], 'control_traj_opt': coc_sol['control_traj_opt'],
     'cost': coc_sol['cost']}]

if True:
    save_data = {'m1': m1,
                 'm2': m2,
                 'l1': l1,
                 'l2': l2,
                 'wq1': wq1,
                 'wq2': wq2,
                 'wdq1': wdq1,
                 'wdq2': wdq2,
                 'wu': wu,
                 'max_u': max_u,
                 'max_q': max_q,
                 'dt': dt,
                 'demos': demos}
    np.save('./Robotarm_Demo.npy', save_data)
