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
Jx, Jy, Jz, mass, l = 0.5, 1., 1., 1., 1.
env.initDyn(Jx=Jx, Jy=Jy, Jz=Jz, mass=mass, l=l)
wr, wv, wtilt, ww, wsidethrust, wthrust = 10, 1, 50, 1, 1, 0.4
env.initCost(wr=wr, wv=wv, wtilt=wtilt, ww=ww, wsidethrust=wsidethrust, wthrust=wthrust)
max_f_sq = 20**2
max_tilt_angle = 0.3
env.initConstraints(max_f_sq=max_f_sq, max_tilt_angle=max_tilt_angle)

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
horizon = 40
# initial condition
init_r_I = [10, -8, 5.]
init_v_I = [-.1, 0.0, -0.0]
init_q = JinEnv.toQuaternion(0, [1, 0, 0])
init_w = [0, 0.0, 0.0]
init_state = init_r_I + init_v_I + init_q + init_w
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon)
print('constrained cost', coc_sol['cost'])
env.play_animation(rocket_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'], control_traj=coc_sol['control_traj_opt'])
plt.plot(np.linalg.norm(coc_sol['control_traj_opt'],2,1), label='ct_control')
plt.legend()
plt.show()
plt.plot(env.getTiltAngle(coc_sol['state_traj_opt']), label='ct_tilt_angle')
plt.legend()
plt.show()

demos = [
    {'state_traj_opt': coc_sol['state_traj_opt'], 'control_traj_opt': coc_sol['control_traj_opt'],
     'cost': coc_sol['cost']}]

if True:
    save_data = {'Jx': Jx,
                 'Jy': Jy,
                 'Jz': Jz,
                 'mass': mass,
                 'l': l,
                 'wr': wr,
                 'wv': wv,
                 'wtilt': wtilt,
                 'ww': ww,
                 'wsidethrust': wsidethrust,
                 'wthrust': wthrust,
                 'max_f_sq': max_f_sq,
                 'max_tilt_angle': max_tilt_angle,
                 'dt': dt,
                 'demos': demos}
    np.save('Rocket_Demo', save_data)
