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
max_x = 0.8
max_u = 5.0
env.initConstraints(max_u=max_u, max_x=max_x)

dt = 0.1
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
horizon = 35
init_state = [0, 0, 0, 0]
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
env.play_animation(pole_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'])
plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
plt.plot(coc_sol['state_traj_opt'][:, 0])
plt.fill_between(np.arange(0, horizon), max_u, -max_u, color='red', alpha=0.2)
plt.fill_between(np.arange(0, horizon), 1, -1, color='green', alpha=0.2)
plt.show()

demos = [
    {'state_traj_opt': coc_sol['state_traj_opt'], 'control_traj_opt': coc_sol['control_traj_opt'], 'cost': coc_sol['cost']}]

if True:
    save_data = {'mc': mc,
                 'mp': mp,
                 'l': l,
                 'wx': wx,
                 'wq': wq,
                 'wdx': wdx,
                 'wdq': wdq,
                 'wu': wu,
                 'max_x': max_x,
                 'max_u': max_u,
                 'dt': dt,
                 'demos': demos}
    np.save('./Cartpole_Demo.npy', save_data)
