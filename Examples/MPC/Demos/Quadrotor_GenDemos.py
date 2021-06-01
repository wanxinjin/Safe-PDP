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
env = JinEnv.Quadrotor()
Jx, Jy, Jz, mass, win_len = 1, 1, 1, 1, 0.4
env.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=win_len, c=0.01)
wr, wv, wq, ww = 1, 1, 5, 1
env.initCost(wr=wr, wv=wv, wq=wq, ww=ww, wthrust=0.1)
max_u = 12
max_r = 200
env.initConstraints(max_u=max_u, max_r=max_r)

dt = 0.15
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
# set initial state
init_r_I = [-5, 5, 5.]
init_v_I = [0 - 5, -5., 0]
init_q = JinEnv.toQuaternion(0, [1, 0, 0])
init_w = [0.0, 0.0, 0.0]
init_state = init_r_I + init_v_I + init_q + init_w
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
env.play_animation(wing_len=1.5, dt=dt, state_traj=coc_sol['state_traj_opt'])
plt.plot(np.amax(np.abs(coc_sol['control_traj_opt']), axis=1), label='Control by COC')
plt.show()

demos = [
    {'state_traj_opt': coc_sol['state_traj_opt'], 'control_traj_opt': coc_sol['control_traj_opt'],
     'cost': coc_sol['cost']}]

if True:
    save_data = {'Jx': Jx,
                 'Jy': Jy,
                 'Jz': Jz,
                 'mass': mass,
                 'win_len': win_len,
                 'wr': wr,
                 'wv': wv,
                 'wq': wq,
                 'ww': ww,
                 'max_u': max_u,
                 'max_r': max_r,
                 'dt': dt,
                 'demos': demos}
    np.save('Quadrotor_Demo', save_data)
