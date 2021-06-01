import numpy as np
from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
from colour import Color
import time
import random
from matplotlib import cm
from ControlTools import ControlTools

# --------------------------- load environment ----------------------------------------
env = JinEnv.Rocket()
env.initDyn(Jx=0.5, Jy=1., Jz=1., mass=1., l=1.)
env.initCost(wr=10, wv=1, wtilt=50, ww=1, wsidethrust=1, wthrust=0.4)
max_f_sq = 20 ** 2
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
time_grid = np.arange(0, horizon+1)
# --------------------------- basic plot setting ----------------------------------------
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# ----------- Plot the comparison between  the Safe MP and ground truth results from OC -------------
if True:
    # load safe motion planning results
    safeload = np.load('./SPlan_Rocket_trial_2.npy', allow_pickle=True).item()
    safe_loss_trace = safeload['loss_trace']
    safe_parameter_trace = safeload['parameter_trace']
    safe_gamma = safeload['gamma']
    safe_max_iter = safe_parameter_trace.shape[0]
    safe_init_parameter = safeload['init_parameter']
    # create safe policy optimization object
    safe_planner = SafePDP.CSysOPT()
    safe_planner.setStateVariable(env.X)
    safe_planner.setControlVariable(env.U)
    safe_planner.setDyn(dyn)
    safe_planner.setPathCost(env.path_cost)
    safe_planner.setFinalCost(env.final_cost)
    safe_planner.setPathInequCstr(env.path_inequ)
    safe_planner.convert2BarrierOC(gamma=safe_gamma)
    # set the poly policy
    safe_planner.setPolyTraj(horizon=horizon, n_poly=safeload['n_poly'])

    # load altro motion planning results
    altro_load = np.load('./ALTRO_Rocket_trial_1.npy', allow_pickle=True).item()
    altro_loss_trace = altro_load['loss_trace']
    altro_control_traj_trace = altro_load['control_traj_trace']
    altro_max_outer = altro_load['max_outer']
    altro_max_inner = altro_load['max_inner']

    # create  PDP policy optimization object
    altro = ControlTools.ALTRO()
    altro.setStateVariable(env.X)
    altro.setControlVariable(env.U)
    altro.setDyn(dyn)
    altro.setPathCost(env.path_cost)
    altro.setFinalCost(env.final_cost)
    altro.setPathConstraint(env.path_inequ)
    altro.diffSys()

    # --------------------------- plot comparison result ----------------------------------------
    # plot the results
    fig = plt.figure(0, figsize=(8.5, 5.0))
    ax = fig.subplots(2, 2)

    iter_index_vec = [k for k in range(0, 800, 20)]
    iter_index_vec += [k for k in range(800, 1500, 200)]
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    colors2 = list(Color("navajowhite").range_to(Color("darkorange"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        # safe policy
        safe_state_traj, safe_control_traj, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                                              control_auxvar_value=safe_parameter_trace[
                                                                                                   iter_k, :])

        ax[0, 0].plot(time_grid[0:-1], np.linalg.norm(safe_control_traj, 2, axis=1), color=colors[i].hex, linewidth=2)
        ax[1, 0].plot(time_grid, env.getTiltAngle(safe_state_traj), color=colors2[i].hex, linewidth=2)

    # legend
    safe_state_trajs, safe_control_trajs, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                                            control_auxvar_value=safe_parameter_trace[
                                                                                                 0, :])
    safe_state_trajf, safe_control_trajf, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                                            control_auxvar_value=safe_parameter_trace[
                                                                                                 -1, :])

    line_safe_control_s, = ax[0, 0].plot(time_grid[0:-1], np.linalg.norm(safe_control_trajs, 2, axis=1),
                                         color=colors[0].hex, zorder=-100, linewidth=3)
    line_safe_control_f, = ax[0, 0].plot(time_grid[0:-1], np.linalg.norm(safe_control_trajs, 2, axis=1),
                                         color=colors[-1].hex, zorder=100, linewidth=3)
    line_safe_state_s, = ax[1, 0].plot(time_grid, env.getTiltAngle(safe_state_trajs), color=colors2[0].hex, zorder=-100,
                                       linewidth=3)
    line_safe_state_f, = ax[1, 0].plot(time_grid, env.getTiltAngle(safe_state_trajf), color=colors2[-1].hex, zorder=100,
                                       linewidth=3)
    ax[0, 0].legend([line_safe_control_s, line_safe_control_f ],
                    ['Iter. #0',   'Iter. #1500'  ], ncol=2, prop={'size': 15},
                    columnspacing=1, handlelength=1, framealpha=0.2).set_zorder(-102)

    ax[1, 0].legend([line_safe_state_s, line_safe_state_f],
                    ['Iter. #0', 'Iter. #1500' ], ncol=2, prop={'size': 15},
                    columnspacing=1, handlelength=1, framealpha=0.2).set_zorder(-102)

    # plot the ALTRO results
    iter_index_vec = [k for k in range(0, 1500, 60)]
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    colors2 = list(Color("navajowhite").range_to(Color("darkorange"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        altro_control_traj = altro_control_traj_trace[iter_k]
        sol = altro.integrateSys(init_state, altro_control_traj)
        altro_state_traj = sol['state_traj']
        ax[0, 1].plot(time_grid[0:-1], np.linalg.norm(altro_control_traj, 2, axis=1), color=colors[i].hex, linewidth=2)
        ax[1, 1].plot(time_grid, env.getTiltAngle(altro_state_traj), color=colors2[i].hex, linewidth=2)



    altro_sols= altro.integrateSys(init_state, altro_control_traj_trace[0])
    altro_solf = altro.integrateSys(init_state, altro_control_traj_trace[-1])
    #
    line_altro_control_s, = ax[0, 1].plot(time_grid[0:-1], np.linalg.norm(altro_sols['control_traj'], 2, axis=1), color=colors[0].hex,
                                    zorder=-100, linewidth=3)
    line_altro_control_f, = ax[0, 1].plot(time_grid[0:-1], np.linalg.norm(altro_solf['control_traj'], 2, axis=1), color=colors[-1].hex,
                                    zorder=100, linewidth=3)
    line_altro_state_s, = ax[1, 1].plot(time_grid, env.getTiltAngle(altro_sols['state_traj']), color=colors2[0].hex, zorder=-100,
                                  linewidth=3)
    line_altro_state_f, = ax[1, 1].plot(time_grid, env.getTiltAngle(altro_solf['state_traj']), color=colors2[-1].hex, zorder=100,
                                  linewidth=3)

    ax[0, 1].legend([line_altro_control_s, line_altro_control_f],
                    ['Iter. #0', 'Iter. #1500', ], ncol=2, prop={'size': 15},
                    columnspacing=1, handlelength=1, framealpha=0.2).set_zorder(-102)

    ax[1, 1].legend([line_altro_state_s, line_altro_state_f],
                    ['Iter. #0', 'Iter. #1500', ], ncol=2, prop={'size': 15},
                    columnspacing=1, handlelength=1, framealpha=0.2).set_zorder(-102)


    ax[0, 0].plot(time_grid, sqrt(max_f_sq) * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0, 0].fill_between(time_grid, sqrt(max_f_sq), 0, color='#EFEFEF', alpha=1)
    ax[1, 0].plot(time_grid, max_tilt_angle * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1, 0].fill_between(time_grid, max_tilt_angle, 0, color='#EFEFEF', alpha=1)

    ax[0, 1].plot(time_grid, sqrt(max_f_sq) * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0, 1].fill_between(time_grid, sqrt(max_f_sq), 0, color='#EFEFEF', alpha=1)
    ax[1, 1].plot(time_grid, max_tilt_angle * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1, 1].fill_between(time_grid, max_tilt_angle, 0, color='#EFEFEF', alpha=1)

    ax[0, 0].set_ylabel(r'Thrust $||u||_2$', labelpad=5)
    ax[1, 0].set_ylabel('Tilt angle', labelpad=5)

    ax[0, 0].set_ylim([0, 30])
    ax[0, 1].set_ylim([0, 30])
    ax[1, 0].set_ylim([0, 0.55])
    ax[1, 1].set_ylim([0, 0.55])
    ax[0, 0].set_xlim([0, horizon])
    ax[0, 1].set_xlim([0, horizon])
    ax[1, 0].set_xlim([0, horizon])
    ax[1, 1].set_xlim([0, horizon])

    ax[0, 0].set_xticks(np.arange(0,horizon+1,10))
    ax[0, 1].set_xticks(np.arange(0,horizon+1,10))
    plt.setp(ax[0, 1].get_yticklabels(), visible=False)
    plt.setp(ax[1, 1].get_yticklabels(), visible=False)
    plt.setp(ax[0, 0].get_xticklabels(), visible=False)
    plt.setp(ax[0, 1].get_xticklabels(), visible=False)

    plt.text(-13.28, 1.00, r'$||u||_{max}$', fontsize=20, fontweight="bold", color='black')
    plt.text(-12.68, 0.27, r'$tilt_{max}$', fontsize=20, fontweight="bold", color='black')

    ax[1,0].set_xticks(np.arange(0,horizon+1,10))
    ax[1,1].set_xticks(np.arange(0,horizon+1,10))

    ax[1, 0].set_xlabel(r'Time $t$')
    ax[1, 1].set_xlabel(r'Time $t$')
    # ax[1,0 ].tick_params(axis='y', which='major', pad=5)
    # ax[0,0 ].tick_params(axis='y', which='major', pad=5)
    # ax[1,0].set_ylim([-2, 3])
    # ax[1,1].set_ylim([-2, 3])
    # ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    # ax[1].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    ax[0, 0].set_title(r'Safe PDP, $\epsilon=10^{-2}$', pad=15)
    ax[0, 1].set_title('ALTRO', pad=15)


    ax[0, 0].grid(alpha=0.5)
    ax[0, 1].grid(alpha=0.5)
    ax[1, 0].grid(alpha=0.5)
    ax[1, 1].grid(alpha=0.5)

    #
    plt.subplots_adjust(left=0.115, right=0.98, bottom=0.14, top=0.89, wspace=0.33, hspace=0.2)
    plt.show()

# ------------Plot the results of the PDP under different gamma (barrier paramter)-----------------
if True:
    # load safe motion planning results

    params = {'axes.labelsize': 28,
              'axes.titlesize': 28,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'legend.fontsize': 16}
    plt.rcParams.update(params)

    loss_trace_list = []
    for j in range(0, 3):
        load = np.load('./SPlan_Rocket_trial_' + str(j) + '.npy', allow_pickle=True).item()
        safe_loss_trace = load['loss_trace']
        loss_trace_list += [safe_loss_trace]

    # plot
    fig = plt.figure(0, figsize=(5.5, 5.5))
    ax = fig.subplots(1, 1)
    gamma_0, = ax.plot(loss_trace_list[0], color='tab:green', linewidth=4, )
    gamma_1, = ax.plot(loss_trace_list[1], color='tab:brown', linewidth=4, )
    gamma_2, = ax.plot(loss_trace_list[2], color='tab:red', linewidth=4, )

    ax.legend([gamma_0, gamma_1, gamma_2],
              [r'$\epsilon=1$', r'$\epsilon=10^{-1}$', r'$\epsilon=10^{-2}$', ], ncol=1, prop={'size': 25},
              columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax.set_xlim(0, 1500)
    # ax.set_ylim(100, 300)
    # ax.tick_params(axis='x', which='major', pad=10)
    # ax.tick_params(axis='y', which='major', pad=10)
    ax.set_xlabel('Iteration', labelpad=0)
    ax.set_ylabel('Loss (planning loss)', labelpad=0)
    ax.set_facecolor('#E6E6E6')
    ax.grid()
    ax.set_position([0.13, 0.13, 0.78, 0.81])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # ax.set_title('Convergence of Safe PDP', pad=25)
    ax.set_xticks(np.arange(0, 1501, 500))

    plt.show()
