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
horizon = 25
dyn = env.X + dt * env.f
# set initial state
init_r_I = [-5, 5, 5.]
init_v_I = [0 - 5, -5., 0]
init_q = JinEnv.toQuaternion(0, [1, 0, 0])
init_w = [0.0, 0.0, 0.0]
init_state = init_r_I + init_v_I + init_q + init_w
time_grid = np.arange(0, horizon)
# --------------------------- basic plot setting ----------------------------------------
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# ----------- Plot the comparison between  the Safe PO and unconstrained PO results -------------
if True:
    # load safe policy optimization results
    load = np.load('./SPO_Quadrotor_trial_2.npy', allow_pickle=True).item()
    safe_loss_trace = load['loss_trace']
    safe_parameter_trace = load['parameter_trace']
    safe_gamma = load['gamma']
    safe_max_iter = safe_parameter_trace.shape[0]
    safe_init_parameter = load['init_parameter']
    # create safe policy optimization object
    safe_optimizer = SafePDP.CSysOPT()
    safe_optimizer.setStateVariable(env.X)
    safe_optimizer.setControlVariable(env.U)
    safe_optimizer.setDyn(dyn)
    safe_optimizer.setPathCost(env.path_cost)
    safe_optimizer.setFinalCost(env.final_cost)
    safe_optimizer.setPathInequCstr(env.path_inequ)
    safe_optimizer.convert2BarrierOC(gamma=safe_gamma)
    # set the neural policy
    safe_optimizer.setNeuralPolicy(hidden_layers=[15])

    # load unconstrained policy optimization results
    load = np.load('./PO_Quadrotor_trial_1.npy', allow_pickle=True).item()
    loss_trace = load['loss_trace']
    parameter_trace = load['parameter_trace']
    max_iter = parameter_trace.shape[0]
    init_parameter = load['init_parameter']
    # create  PDP policy optimization object
    optimizer = SafePDP.CSysOPT()
    optimizer.setStateVariable(env.X)
    optimizer.setControlVariable(env.U)
    dyn = env.X + dt * env.f
    optimizer.setDyn(dyn)
    optimizer.setPathCost(env.path_cost)
    optimizer.setFinalCost(env.final_cost)
    optimizer.setPathInequCstr(env.path_inequ)
    optimizer.convert2BarrierOC(gamma=0)
    # set the neural policy
    optimizer.setNeuralPolicy(hidden_layers=[15])

    # --------------------------- plot comparison result ----------------------------------------

    # plot the results
    fig = plt.figure(0, figsize=(8.5, 5))
    ax = fig.subplots(1, 2)

    # storage of the safe
    safe_state_traj_trace = []
    safe_control_traj_trace = []
    # storage of the unconstrained one
    state_traj_trace = []
    control_traj_trace = []

    iter_index_vec = [k for k in range(0, 2000, 60)]
    # iter_index_vec = [-1]
    # color_light_vec = np.linspace(0.6, 0.1, len(iter_index_vec))
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        # safe policy
        safe_state_traj, safe_control_traj, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                                                control_auxvar_value=safe_parameter_trace[
                                                                                                     iter_k, :])
        safe_state_traj_trace += [safe_state_traj]
        safe_control_traj_trace += [safe_control_traj]
        # unconstrained policy
        state_traj, control_traj, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                                 control_auxvar_value=parameter_trace[iter_k, :])
        state_traj_trace += [state_traj]
        control_traj_trace += [control_traj]

        # ax[0].plot(time_grid, safe_control_traj, color=color_light_vec[i]*np.array([1.0,1,1]))
        # ax[1].plot(time_grid, control_traj, color=color_light_vec[i]*np.array([1.0,0,0]))

        ax[0].plot(time_grid, np.linalg.norm(safe_control_traj,inf,axis=1), color=colors[i].hex, )
        ax[1].plot(time_grid, np.linalg.norm(control_traj,inf,axis=1), color=colors[i].hex, )


    # legend
    _, safe_control_trajs, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    0, :])
    _, safe_control_trajf, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    -1, :])

    line_safe_s, = ax[0].plot(time_grid, np.linalg.norm(safe_control_trajs,inf,axis=1), color=colors[0].hex,label='Iter. #0', zorder=-100, linewidth=4)
    line_safe_f, = ax[0].plot(time_grid, np.linalg.norm(safe_control_trajf,inf,axis=1), color=colors[-1].hex, label='Iter. #2000', zorder=100, linewidth=4)
    ax[0].legend(ncol=1, prop={'size': 18}, columnspacing=0.5, handlelength=1).set_zorder(-102)

    # legend
    _, control_trajs, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                     control_auxvar_value=parameter_trace[
                                                                          0, :])
    _, control_trajf, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                     control_auxvar_value=parameter_trace[
                                                                          -1, :])

    ax[1].plot(time_grid, np.linalg.norm(control_trajs,inf,axis=1), color=colors[0].hex, zorder=-100, label='Iter. #0', linewidth=4)
    ax[1].plot(time_grid, np.linalg.norm(control_trajf,inf,axis=1), color=colors[-1].hex, zorder=100, label='Iter. #2000', linewidth=4)
    ax[1].legend(ncol=1, prop={'size': 18}, columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax[0].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')

    ax[0].fill_between(time_grid, max_u, 0, color='#EFEFEF', alpha=1)
    ax[1].fill_between(time_grid, max_u, 0, color='#EFEFEF', alpha=1)
    ax[0].set_ylabel(r'Generated $||u_t||_\infty$', labelpad=0)
    ax[0].set_xlabel(r'Time $t$')
    ax[0].set_xlim([0, horizon])
    ax[0].set_ylim([0, 20])
    ax[1].set_xlim([0, horizon])
    ax[1].set_xlabel(r'Time $t$')
    ax[1].set_ylim([0, 20])
    # ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    # ax[1].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    ax[0].set_title(r'Safe PDP, $\epsilon=10^{-2}$', pad=15)
    ax[1].set_title('Unconstrained', pad=15)
    ax[0].set_xticks(np.arange(0,horizon+1,5))
    ax[1].set_xticks(np.arange(0,horizon+1,5))

    ax[0].grid(alpha=0.5)
    ax[1].grid(alpha=0.5)

    # # ax[0].grid()
    # # ax[1].grid()
    plt.text(-7.2, max_u-0.4, r'$u_{max}$', fontsize=30, fontweight="bold", color='black')
    #
    plt.setp(ax[1].get_yticklabels(), visible=False)
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.89, wspace=0.30)
    plt.show()

# ------------Plot the results of the safe PO under different gamma (barrier paramter)-----------------
if False:
    # load safe policy optimization results

    params = {'axes.labelsize': 28,
              'axes.titlesize': 28,
              'xtick.labelsize': 22,
              'ytick.labelsize': 22,
              'legend.fontsize': 16}
    plt.rcParams.update(params)

    loss_trace_list = []
    for j in range(0, 3):
        load = np.load('./SPO_Quadrotor_trial_' + str(j) + '.npy', allow_pickle=True).item()
        safe_loss_trace = load['loss_trace']
        loss_trace_list += [safe_loss_trace]
        print(load['gamma'])

    # plot
    fig = plt.figure(0, figsize=(5.5, 5.5))
    ax = fig.subplots(1, 1)
    gamma_0,= ax.plot(loss_trace_list[0], color='tab:green', linewidth=4, )
    gamma_1,= ax.plot(loss_trace_list[1], color='tab:brown', linewidth=4, )
    gamma_2,= ax.plot(loss_trace_list[2], color='tab:red', linewidth=4, )
    # gamma_4,= ax.plot(loss_trace_list[3], color='tab:red', linewidth=4, )
    ax.legend([gamma_0, gamma_1, gamma_2],
                          [r'$\epsilon=1$', r'$\epsilon=10^{-1}$',r'$\epsilon=10^{-2}$' ], ncol=1, prop={'size': 25}, columnspacing=0.5, handlelength=1).set_zorder(-102)


    ax.set_xlim(0, 2000)
    ax.set_xlabel('Iteration', labelpad=0)
    ax.set_ylabel('Loss (control cost)', labelpad=0)
    ax.set_facecolor('#E6E6E6')
    ax.grid()
    ax.set_position([0.24, 0.13, 0.70, 0.78])
    # ax.set_title('Convergence of Safe PDP', pad=25)
    ax.set_xticks(np.arange(0, 2001, 500))

    plt.show()


# ------------Plot all lines together in the same figure-----------------
if True:
    # --------------------------- basic plot setting ----------------------------------------
    params = {'axes.labelsize': 25,
              'axes.titlesize': 25,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.fontsize': 20}
    plt.rcParams.update(params)

    # plot
    fig = plt.figure(0, figsize=(15, 5))
    ax = fig.subplots(1, 3)

    loss_trace_list = []
    for j in range(0, 3):
        load = np.load('./SPO_Quadrotor_trial_' + str(j) + '.npy', allow_pickle=True).item()
        safe_loss_trace = load['loss_trace']
        loss_trace_list += [safe_loss_trace]
        print(load['gamma'])

    # plot
    gamma_0,= ax[0].plot(loss_trace_list[0], color='tab:green', linewidth=4, )
    gamma_1,= ax[0].plot(loss_trace_list[1], color='tab:brown', linewidth=4, )
    gamma_2,= ax[0].plot(loss_trace_list[2], color='tab:red', linewidth=4, )
    # gamma_4,= ax.plot(loss_trace_list[3], color='tab:red', linewidth=4, )
    ax[0].legend([gamma_0, gamma_1, gamma_2],
                          [r'$\epsilon=1$', r'$\epsilon=10^{-1}$',r'$\epsilon=10^{-2}$' ], ncol=1, prop={'size': 25}, columnspacing=0.5, handlelength=1).set_zorder(-102)


    ax[0].set_xlim(0, 2000)
    ax[0].set_xlabel('Iteration', labelpad=0)
    ax[0].set_ylabel('Loss (control cost)', labelpad=0)
    ax[0].set_facecolor('#E6E6E6')
    ax[0].grid()



   # load safe policy optimization results
    load = np.load('./SPO_Quadrotor_trial_2.npy', allow_pickle=True).item()
    safe_loss_trace = load['loss_trace']
    safe_parameter_trace = load['parameter_trace']
    safe_gamma = load['gamma']
    safe_max_iter = safe_parameter_trace.shape[0]
    safe_init_parameter = load['init_parameter']
    # create safe policy optimization object
    safe_optimizer = SafePDP.CSysOPT()
    safe_optimizer.setStateVariable(env.X)
    safe_optimizer.setControlVariable(env.U)
    safe_optimizer.setDyn(dyn)
    safe_optimizer.setPathCost(env.path_cost)
    safe_optimizer.setFinalCost(env.final_cost)
    safe_optimizer.setPathInequCstr(env.path_inequ)
    safe_optimizer.convert2BarrierOC(gamma=safe_gamma)
    # set the neural policy
    safe_optimizer.setNeuralPolicy(hidden_layers=[15])

    # load unconstrained policy optimization results
    load = np.load('./PO_Quadrotor_trial_1.npy', allow_pickle=True).item()
    loss_trace = load['loss_trace']
    parameter_trace = load['parameter_trace']
    max_iter = parameter_trace.shape[0]
    init_parameter = load['init_parameter']
    # create  PDP policy optimization object
    optimizer = SafePDP.CSysOPT()
    optimizer.setStateVariable(env.X)
    optimizer.setControlVariable(env.U)
    dyn = env.X + dt * env.f
    optimizer.setDyn(dyn)
    optimizer.setPathCost(env.path_cost)
    optimizer.setFinalCost(env.final_cost)
    optimizer.setPathInequCstr(env.path_inequ)
    optimizer.convert2BarrierOC(gamma=0)
    # set the neural policy
    optimizer.setNeuralPolicy(hidden_layers=[15])

    # --------------------------- plot comparison result ----------------------------------------
    # storage of the safe
    safe_state_traj_trace = []
    safe_control_traj_trace = []
    # storage of the unconstrained one
    state_traj_trace = []
    control_traj_trace = []

    iter_index_vec = [k for k in range(0, 2000, 60)]
    # iter_index_vec = [-1]
    # color_light_vec = np.linspace(0.6, 0.1, len(iter_index_vec))
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(iter_index_vec)))
    for i, iter_k in enumerate(iter_index_vec):
        # safe policy
        safe_state_traj, safe_control_traj, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                                                control_auxvar_value=safe_parameter_trace[
                                                                                                     iter_k, :])
        safe_state_traj_trace += [safe_state_traj]
        safe_control_traj_trace += [safe_control_traj]
        # unconstrained policy
        state_traj, control_traj, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                                 control_auxvar_value=parameter_trace[iter_k, :])
        state_traj_trace += [state_traj]
        control_traj_trace += [control_traj]

        # ax[0].plot(time_grid, safe_control_traj, color=color_light_vec[i]*np.array([1.0,1,1]))
        # ax[1].plot(time_grid, control_traj, color=color_light_vec[i]*np.array([1.0,0,0]))

        ax[1].plot(time_grid, np.linalg.norm(safe_control_traj,inf,axis=1), color=colors[i].hex, )
        ax[2].plot(time_grid, np.linalg.norm(control_traj,inf,axis=1), color=colors[i].hex, )


    # legend
    _, safe_control_trajs, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    0, :])
    _, safe_control_trajf, _, _, = safe_optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                               control_auxvar_value=safe_parameter_trace[
                                                                                    -1, :])

    line_safe_s, = ax[1].plot(time_grid, np.linalg.norm(safe_control_trajs,inf,axis=1), color=colors[0].hex,label='Iter. #0', zorder=-100, linewidth=4)
    line_safe_f, = ax[1].plot(time_grid, np.linalg.norm(safe_control_trajf,inf,axis=1), color=colors[-1].hex, label='Iter. #2000', zorder=100, linewidth=4)
    ax[1].legend(ncol=1, prop={'size': 18}, columnspacing=0.5, handlelength=1).set_zorder(-102)

    # legend
    _, control_trajs, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                     control_auxvar_value=parameter_trace[
                                                                          0, :])
    _, control_trajf, _, _, = optimizer.integrateSys(init_state=init_state, horizon=horizon,
                                                     control_auxvar_value=parameter_trace[
                                                                          -1, :])

    ax[2].plot(time_grid, np.linalg.norm(control_trajs,inf,axis=1), color=colors[0].hex, zorder=-100, label='Iter. #0', linewidth=4)
    ax[2].plot(time_grid, np.linalg.norm(control_trajf,inf,axis=1), color=colors[-1].hex, zorder=100, label='Iter. #2000', linewidth=4)
    ax[2].legend(ncol=1, prop={'size': 18}, columnspacing=0.5, handlelength=1).set_zorder(-102)

    ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')
    ax[2].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=4, color='black')

    ax[1].fill_between(time_grid, max_u, 0, color='#EFEFEF', alpha=1)
    ax[2].fill_between(time_grid, max_u, 0, color='#EFEFEF', alpha=1)
    ax[1].set_ylabel(r'Generated $||u_t||_\infty$', labelpad=0)
    ax[1].set_xlabel(r'Time $t$')
    ax[1].set_xlim([0, horizon])
    ax[1].set_ylim([0, 20])
    ax[2].set_xlim([0, horizon])
    ax[2].set_xlabel(r'Time $t$')
    ax[2].set_ylim([0, 20])
    # ax[1].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    # ax[1].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='red')
    ax[1].set_title(r'Safe PDP, $\epsilon=10^{-2}$', pad=15)
    ax[2].set_title('Unconstrained', pad=15)
    ax[1].set_xticks(np.arange(0,horizon+1,5))
    ax[2].set_xticks(np.arange(0,horizon+1,5))

    ax[1].grid(alpha=0.5)
    ax[2].grid(alpha=0.5)

    # # ax[0].grid()
    # # ax[1].grid()
    plt.text(-10.2, max_u-0.4, r'$u_{max}$', fontsize=30, fontweight="bold", color='black')

    plt.setp(ax[2].get_yticklabels(), visible=False)
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.18, top=0.85, wspace=0.50)
    plt.show()