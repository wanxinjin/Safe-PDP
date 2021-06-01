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
env = JinEnv.CartPole()
mc, mp, l = 0.5, 0.5, 1
env.initDyn(mc=mc, mp=mp, l=l)
wx, wq, wdx, wdq, wu = 0.1, 1, 0.1, 0.1, 0.1
env.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)
max_x = 1
max_u = 4
env.initConstraints(max_u=4, max_x=max_x)
dt = 0.12
horizon = 25
init_state = [0, 0, 0, 0]
dyn = env.X + dt * env.f
time_grid = np.arange(0, horizon+1)
# --------------------------- basic plot setting ----------------------------------------
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# ----------- Plot the comparison between  the Safe PDP and ALTRO  results -------------
if True:
    # load safe motion planning results
    load = np.load('./SPlan_Cartpole_trial_2.npy', allow_pickle=True).item()
    safe_loss_trace = load['loss_trace']
    safe_parameter_trace = load['parameter_trace']
    safe_gamma = load['gamma']
    safe_max_iter = safe_parameter_trace.shape[0]
    safe_init_parameter = load['init_parameter']
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
    safe_planner.setPolyTraj(horizon=horizon, n_poly=load['n_poly'])

    # load altro motion planning results
    altro_load = np.load('./ALTRO_Cartpole_trial_1.npy', allow_pickle=True).item()
    altro_loss_trace = altro_load['loss_trace']
    altro_control_traj_trace = altro_load['control_traj_trace']
    altro_max_outer = altro_load['max_outer']
    altro_max_inner=altro_load['max_inner']

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
    fig = plt.figure(0, figsize=(8.5, 5.0))
    ax = fig.subplots(2, 2)

    # plot the safe PDP results
    iter_index_vec = [k for k in range(0, 1000, 30)]
    iter_index_vec += [k for k in range(1000,3000, 1000)]
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    colors2=list(Color("navajowhite").range_to(Color("darkorange"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        # safe policy
        safe_state_traj, safe_control_traj, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                                              control_auxvar_value=safe_parameter_trace[
                                                                                                   iter_k, :])
        ax[0,0].plot(time_grid[0:-1], safe_control_traj,  color=colors[i].hex, )
        ax[1,0].plot(time_grid, safe_state_traj[:,0],  color=colors2[i].hex, )

    # legend
    safe_state_trajs, safe_control_trajs, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    0, :])
    safe_state_trajf, safe_control_trajf, _, _, = safe_planner.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    -1, :])
    line_safe_control_s, = ax[0,0].plot(time_grid[0:-1], safe_control_trajs, color=colors[0].hex, zorder=-100, linewidth=3)
    line_safe_control_f, = ax[0,0].plot(time_grid[0:-1], safe_control_trajf, color=colors[-1].hex, zorder=100, linewidth=3)
    line_safe_state_s, = ax[1,0].plot(time_grid, safe_state_trajs[:,0], color=colors2[0].hex, zorder=-100, linewidth=3)
    line_safe_state_f, = ax[1,0].plot(time_grid, safe_state_trajf[:,0], color=colors2[-1].hex, zorder=100, linewidth=3)
    ax[0,0].legend([line_safe_control_s, line_safe_control_f],
                          ['Iter. #0', 'Iter. #3000', ], ncol=2, prop={'size': 15},
                          columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax[1, 0].legend([line_safe_state_s, line_safe_state_f],
                    ['Iter. #0', 'Iter. #3000', ], ncol=2, prop={'size': 15},
                    columnspacing=0.5, handlelength=1).set_zorder(-102)


    # plot the ALTRO results
    iter_index_vec = [k for k in range(0, 300, 8)]
    iter_index_vec+=[k for k in range(300, 3000, 1000)]
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    colors2=list(Color("navajowhite").range_to(Color("darkorange"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        altro_control_traj=altro_control_traj_trace[iter_k]
        sol = altro.integrateSys(init_state, altro_control_traj)
        altro_state_traj=sol['state_traj']
        ax[0, 1].plot(time_grid[0:-1], altro_control_traj, color=colors[i].hex, )
        ax[1, 1].plot(time_grid, altro_state_traj[:, 0], color=colors2[i].hex)

    # legend
    altro_sols = altro.integrateSys(init_state, altro_control_traj_trace[0])
    altro_solf = altro.integrateSys(init_state, altro_control_traj_trace[-1])

    line_altro_control_s, = ax[0,1].plot(time_grid[0:-1],  altro_control_traj_trace[0], color=colors[0].hex, zorder=-100, linewidth=3)
    line_altro_control_f, = ax[0,1].plot(time_grid[0:-1],  altro_control_traj_trace[-1], color=colors[-1].hex, zorder=100, linewidth=3)
    line_altro_state_s, = ax[1,1].plot(time_grid, altro_sols['state_traj'][:,0], color=colors2[0].hex, zorder=-100, linewidth=3)
    line_altro_state_f, = ax[1,1].plot(time_grid, altro_solf['state_traj'][:,0], color=colors2[-1].hex, zorder=100, linewidth=3)
    ax[0,1].legend([line_altro_control_s, line_altro_control_f],
                          ['Iter. #0', 'Iter. #3000', ], ncol=2, prop={'size': 15},
                          columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax[1, 1].legend([line_altro_state_s, line_altro_state_f],
                    ['Iter. #0', 'Iter. #3000', ], ncol=2, prop={'size': 15},
                    columnspacing=0.5, handlelength=1).set_zorder(-102)


    ax[0,0].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0,0].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0,1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0,1].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[0,0].fill_between(time_grid, max_u, -max_u, color='#EFEFEF', alpha=1)
    ax[0,1].fill_between(time_grid, max_u, -max_u, color='#EFEFEF', alpha=1)

    ax[1,0].fill_between(time_grid, max_x, -max_x, color='#EFEFEF', alpha=1)
    ax[1,1].fill_between(time_grid, max_x, -max_x, color='#EFEFEF', alpha=1)
    ax[1,0].plot(time_grid, max_x * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1,0].plot(time_grid, -max_x * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1,1].plot(time_grid, max_x * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1,1].plot(time_grid, -max_x * np.ones_like(time_grid), '--', linewidth=4, color='black')


    ax[0,0].set_ylabel('Control', labelpad=0)
    ax[1,0].set_ylabel('Cart pos.', labelpad=0)
    # ax[0].set_xlabel('Time')
    # ax[0].tick_params(axis='x', which='major', pad=10)
    # ax[0].tick_params(axis='y', which='major', pad=10)
    # ax[0].set_xlim([0, 3])
    ax[0,0].set_ylim([-7, 9])
    ax[0,1].set_ylim([-7, 9])
    ax[1,0].set_ylim([-2, 2.5])
    ax[1,1].set_ylim([-2, 2.5])
    ax[0, 0].set_xlim([0, horizon])
    ax[0, 1].set_xlim([0, horizon])
    ax[1, 0].set_xlim([0, horizon])
    ax[1, 1].set_xlim([0, horizon])

    ax[0, 0].set_xticks(np.arange(0,horizon+1,5))
    ax[0, 1].set_xticks(np.arange(0,horizon+1,5))
    plt.setp(ax[0,1].get_yticklabels(), visible=False)
    plt.setp(ax[1,1].get_yticklabels(), visible=False)
    plt.setp(ax[0,0].get_xticklabels(), visible=False)
    plt.setp(ax[0,1].get_xticklabels(), visible=False)

    plt.text(-6.81, 6.2, r'$u_{max}$', fontsize=25, fontweight="bold", color='black')
    plt.text(-6.81, 4.0, r'$u_{min}$', fontsize=25, fontweight="bold", color='black')
    plt.text(-6.81, 0.8, r'$x_{max}$', fontsize=25, fontweight="bold", color='black')
    plt.text(-6.81, -1.3, r'$x_{min}$', fontsize=25, fontweight="bold", color='black')

    ax[1,0].set_xticks(np.arange(0,horizon+1,5))
    ax[1,1].set_xticks(np.arange(0,horizon+1,5))


    ax[1,0].set_xlabel(r'Time $t$')
    ax[1,1].set_xlabel(r'Time $t$')
    # ax[1].tick_params(axis='x', which='major', pad=10)
    # ax[1,0].set_ylim([-2, 3])
    # ax[1,1].set_ylim([-2, 3])
    # ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    # ax[1].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    ax[0,0].set_title(r'Safe PDP, $\epsilon=10^{-2}$', pad=15)
    ax[0,1].set_title('ALTRO', pad=15)


    ax[0,0].grid(alpha=0.5)
    ax[0,1].grid(alpha=0.5)
    ax[1, 0].grid(alpha=0.5)
    ax[1, 1].grid(alpha=0.5)

    #
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.15, top=0.89, wspace=0.30, hspace=0.2)
    plt.show()

# ------------Plot multiple trials of the safe PO results-----------------------------
if False:
    # load safe motion planning results

    params = {'axes.labelsize': 28,
              'axes.titlesize': 28,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'legend.fontsize': 16}
    plt.rcParams.update(params)

    loss_trace_list = []
    for j in range(1, 2):
        load = np.load('./SPlan_Cartpole_trial_' + str(j) + '.npy', allow_pickle=True).item()
        safe_loss_trace = load['loss_trace']
        loss_trace_list += [safe_loss_trace]



    # plot
    fig = plt.figure(0, figsize=(5.5, 5.5))
    ax = fig.subplots(1, 1)
    for loss_trace in loss_trace_list:
        ax.plot(loss_trace, color=[0.6350, 0.0780, 0.1840], linewidth=4, )

    ax.set_xlim(0, 2000)
    # ax.set_ylim(100, 300)
    # ax.tick_params(axis='x', which='major', pad=10)
    # ax.tick_params(axis='y', which='major', pad=10)
    ax.set_xlabel('Iteration', labelpad=0)
    ax.set_ylabel('Planning loss', labelpad=0)
    ax.set_facecolor('#E6E6E6')
    ax.grid()
    ax.set_position([0.19, 0.13, 0.73, 0.81])
    # ax.set_title('Convergence of Safe PDP', pad=25)
    ax.set_xticks(np.arange(0, 2001, 500))

    plt.show()


# ------------Plot the results of the PDP under different gamma (barrier paramter)-----------------
if True:
    # load safe policy optimization results

    params = {'axes.labelsize': 28,
              'axes.titlesize': 28,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'legend.fontsize': 16}
    plt.rcParams.update(params)

    loss_trace_list = []
    for j in range(0, 3):
        load = np.load('./SPlan_Cartpole_trial_' + str(j) + '.npy', allow_pickle=True).item()
        safe_loss_trace = load['loss_trace']
        loss_trace_list += [safe_loss_trace]
        print(load['gamma'])



    # plot
    fig = plt.figure(0, figsize=(5.5, 5.5))
    ax = fig.subplots(1, 1)
    gamma_0,= ax.plot(loss_trace_list[0], color='tab:green', linewidth=4, )
    gamma_1,= ax.plot(loss_trace_list[1], color='tab:brown', linewidth=4, )
    gamma_2,= ax.plot(loss_trace_list[2], color='tab:red', linewidth=4, )

    ax.legend([gamma_0, gamma_1, gamma_2],
                          [r'$\epsilon=1$', r'$\epsilon=10^{-1}$', r'$\epsilon=10^{-2}$', ], ncol=1, prop={'size': 25}, columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax.set_xlim(0, 3000)
    # ax.set_ylim(100, 300)
    ax.set_xlabel('Iteration', labelpad=0)
    ax.set_ylabel('Loss (planning loss)', labelpad=0)
    ax.set_facecolor('#E6E6E6')
    ax.grid()
    ax.set_position([0.21, 0.13, 0.72, 0.78])
    # ax.set_title('Convergence of Safe PDP', pad=25)
    ax.set_xticks(np.arange(0, 3001, 1000))

    plt.show()

