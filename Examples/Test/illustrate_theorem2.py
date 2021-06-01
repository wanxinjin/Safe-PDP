import numpy as np
from SafePDP import SafePDP
from SafePDP import PDP
from JinEnv import JinEnv
from casadi import *
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import random
from colour import Color

# --------------------------- load environment ----------------------------------------
env = JinEnv.CartPole()
mc, mp, l = 0.5, 0.5, 1
env.initDyn()
wx, wq, wdx, wdq, wu = 0.1, 1, 0.1, 0.1, 0.1
env.initCost(wu=wu)
max_u = 4
max_x = 0.8
env.initConstraints()
true_theta = [mc, mp, l, wx, wq, wdx, wdq, max_u, max_x]

dt = 0.12
horizon = 25
init_state = [0, 0, 0, 0]
time_grid = np.arange(0, horizon + 1)

# --------------------------- create constrained COC object  --------------------------
coc = SafePDP.COCsys()
coc.setAuxvarVariable(vcat([env.dyn_auxvar, env.cost_auxvar, env.constraint_auxvar]))
coc.setStateVariable(env.X)
coc.setControlVariable(env.U)
dyn = env.X + dt * env.f
coc.setDyn(dyn)
coc.setPathCost(env.path_cost)
coc.setFinalCost(env.final_cost)
coc.setPathInequCstr(env.path_inequ)
coc_sol = coc.ocSolver(init_state=init_state, horizon=horizon, auxvar_value=true_theta)
# env.play_animation(pole_len=2, dt=dt, state_traj=coc_sol['state_traj_opt'])

# plt.plot(coc_sol['control_traj_opt'], label='Control by COC')
# plt.plot(coc_sol['state_traj_opt'][:,0],label='Position by COC')
# plt.fill_between(time_grid, max_u, -max_u, color='b', alpha=0.1)
# plt.fill_between(time_grid, max_x, -max_x, color='c', alpha=0.1)
# plt.show()


# # --------------------------- plot the case of differnt gramma  -----------------------------
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 20}
plt.rcParams.update(params)

if False:

    # gamma_list = []
    gamma_list = np.linspace(0.5, 1e-2, 15).tolist()
    gamma_list += np.linspace(1e-2, 1e-4, 10).tolist()
    # gamma_list = [1e-0, 1e-1, 1e-2, 1e-3]
    barrier_oc_sol_list = []
    for gamma in gamma_list:
        coc.convert2BarrierOC(gamma=gamma)
        barrier_oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
        barrier_oc_sol_list += [barrier_oc_sol]

    # --------------------------control input plot ---------------

    fig1 = plt.figure(0, figsize=(5.5, 4.5))
    ax1 = fig1.subplots(1, 1)
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(barrier_oc_sol_list)))
    true_coc_control_traj, = ax1.plot(coc_sol['control_traj_opt'], linewidth=4, color='tab:red', zorder=10,
                                      linestyle=':', marker='o', markersize=0)

    boundu, = ax1.plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax1.plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax1.fill_between(time_grid, max_u, -max_u, color='c', alpha=0.1)

    for count, barrier_oc_sol in enumerate(barrier_oc_sol_list):
        ax1.plot(barrier_oc_sol['control_traj_opt'], linewidth=4, color=colors[count].hex, )

    # legend
    sgamma_control_traj, = ax1.plot(barrier_oc_sol_list[0]['control_traj_opt'], linewidth=4, color=colors[0].hex,
                                    zorder=-100)

    fgamma_control_traj, = ax1.plot(barrier_oc_sol_list[-1]['control_traj_opt'], linewidth=4, color=colors[-1].hex,
                                    zorder=-100)

    ax1.legend([sgamma_control_traj, true_coc_control_traj, fgamma_control_traj, boundu],
               [r'$\gamma=0.5$', 'true sol', r'$\gamma=10^{-4}$', 'constraints'], ncol=2, prop={'size': 18},
               columnspacing=1.5, handlelength=1).set_zorder(-102)

    ax1.set_xlabel('Time')
    ax1.set_xticks(np.arange(0, horizon + 1, 5))
    ax1.set_ylabel('Cart force input', labelpad=0)
    ax1.set_ylim([-6, 8.5])
    ax1.set_position([0.21, 0.16, 0.78, 0.81])

    # ------------------- cart position plot

    fig2 = plt.figure(1, figsize=(5.5, 4.5))
    ax2 = fig2.subplots(1, 1)
    colors2 = list(Color("navajowhite").range_to(Color("darkorange"), len(barrier_oc_sol_list)))
    true_coc_position_traj, = ax2.plot(coc_sol['state_traj_opt'][:, 0], linewidth=4, color='tab:red', zorder=10,
                                       linestyle=':', marker='o', markersize=0)
    boundx, = ax2.plot(time_grid, max_x * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax2.plot(time_grid, -max_x * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax2.fill_between(time_grid, max_x, -max_x, color='c', alpha=0.1)

    for count, barrier_oc_sol in enumerate(barrier_oc_sol_list):
        ax2.plot(barrier_oc_sol['state_traj_opt'][:, 0], linewidth=4, color=colors2[count].hex)

    # legend
    sgamma_position_traj, = ax2.plot(barrier_oc_sol_list[0]['state_traj_opt'][:, 0], linewidth=4,
                                     color=colors2[0].hex, zorder=-100)
    fgamma_position_traj, = ax2.plot(barrier_oc_sol_list[-1]['state_traj_opt'][:, 0], linewidth=4,
                                     color=colors2[-1].hex, zorder=-100)

    ax2.legend([sgamma_position_traj, true_coc_control_traj, fgamma_position_traj, boundx],
               [r'$\gamma=0.5$', 'true sol', r'$\gamma=10^{-4}$', 'counstraints'], ncol=2, prop={'size': 18},
               columnspacing=1.5, handlelength=1, ).set_zorder(-102)

    # plt.legend([true_coc_control_traj,oc_control_traj],['constrained sol', 'unconstrained sol'])

    ax2.set_xlabel('Time')
    ax2.set_xticks(np.arange(0, horizon + 1, 5))
    ax2.set_ylabel('Cart position', labelpad=0)
    ax2.set_ylim([-1.5, 2])
    ax2.set_position([0.16, 0.16, 0.78, 0.81])

    plt.show()

if False:

    true_coc_control_traj = coc_sol['control_traj_opt']
    true_coc_state_traj = coc_sol['state_traj_opt']

    gamma_list = np.linspace(1e-1, 1e-4, 50)
    traj_error_list = []
    for gamma in gamma_list:
        coc.convert2BarrierOC(gamma=gamma)
        barrier_oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
        barrier_control_traj = barrier_oc_sol['control_traj_opt']
        barrier_state_traj = barrier_oc_sol['state_traj_opt']

        traj_error = np.linalg.norm(true_coc_control_traj.flatten() - barrier_control_traj.flatten(),
                                    2) ** 2 + np.linalg.norm(
            true_coc_state_traj.flatten() - barrier_state_traj.flatten(), 2) ** 2

        traj_error_list += [traj_error]

    # ------------------- plot cart cost function value

    fig3 = plt.figure(1, figsize=(5.5, 4.5))
    ax3 = fig3.subplots(1, 1)

    ax3.plot(gamma_list, traj_error_list, linewidth=5, color='tab:red', marker='o', markersize=0)

    ax3.set_xlabel(r'$\gamma$')
    ax3.set_ylabel(r'$||\xi_{(\theta,\gamma)}-\xi_{\theta}||^2$')
    ax3.set_xscale('log')
    ax3.set_xlim([1e-1, 1e-4])
    ax3.set_facecolor('#E6E6E6')
    ax3.grid()
    ax3.set_position([0.19, 0.18, 0.74, 0.79])

    plt.show()

if False:

    # compute the true trajectory derivative using a very very small gamma because Theorem is very unstable when identifying the active constraints
    gamma = 1e-5
    coc.convert2BarrierOC(gamma=gamma)
    oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
    true_grad_traj = coc.auxSysBarrierOC(opt_sol=oc_sol)
    true_grad_state = true_grad_traj['state_traj_opt']
    true_grad_control = true_grad_traj['control_traj_opt']

    gamma_list = np.linspace(1e-1, 1e-4, 50)
    grad_traj_error_list = []
    for gamma in gamma_list:
        coc.convert2BarrierOC(gamma=gamma)
        barrier_oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
        barrier_grad_traj = coc.auxSysBarrierOC(opt_sol=barrier_oc_sol)
        barrier_grad_state = barrier_grad_traj['state_traj_opt']
        barrier_grad_control = barrier_grad_traj['control_traj_opt']

        # compute the gradient
        grad_traj_error = 0
        for t in range(len(barrier_grad_state)):
            grad_traj_error += np.linalg.norm(barrier_grad_state[t].flatten() - true_grad_state[t].flatten(),2)**2
        for t in range(len(barrier_grad_control)):
            grad_traj_error+=np.linalg.norm(barrier_grad_control[t].flatten()-true_grad_control[t].flatten(),2)**2



        grad_traj_error_list += [grad_traj_error]

    # ------------------- plot cart cost function value

    fig4 = plt.figure(1, figsize=(5.5, 4.5))
    ax4 = fig4.subplots(1, 1)

    ax4.plot(gamma_list, grad_traj_error_list, linewidth=5, color='tab:red', marker='o', markersize=0)

    ax4.set_xlabel(r'$\gamma$')
    ax4.set_ylabel(r'$||\frac{\partial \xi_{(\theta,\gamma)}}{\partial \theta}-\frac{\partial \xi_{\theta}}{\partial \theta}||^2$')
    ax4.set_xscale('log')
    ax4.set_xlim([1e-1, 1e-4])
    ax4.set_facecolor('#E6E6E6')
    ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax4.grid()
    ax4.set_position([0.17, 0.18, 0.77, 0.75])

    plt.show()

if True:

    # true constrained optimal control solution
    true_coc_control_traj = coc_sol['control_traj_opt']
    true_coc_state_traj = coc_sol['state_traj_opt']

    # compute the true trajectory derivative using a very very small gamma because Theorem is very unstable when identifying the active constraints
    gamma = 1e-5
    coc.convert2BarrierOC(gamma=gamma)
    oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
    true_grad_traj = coc.auxSysBarrierOC(opt_sol=oc_sol)
    true_grad_state = true_grad_traj['state_traj_opt']
    true_grad_control = true_grad_traj['control_traj_opt']

    gamma_list = np.linspace(1e-1, 1e-4, 50)
    traj_error_list = []
    grad_traj_error_list = []
    for gamma in gamma_list:
        coc.convert2BarrierOC(gamma=gamma)
        barrier_oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)

        barrier_control_traj = barrier_oc_sol['control_traj_opt']
        barrier_state_traj = barrier_oc_sol['state_traj_opt']

        barrier_grad_traj = coc.auxSysBarrierOC(opt_sol=barrier_oc_sol)

        barrier_grad_state = barrier_grad_traj['state_traj_opt']
        barrier_grad_control = barrier_grad_traj['control_traj_opt']

        traj_error = np.linalg.norm(true_coc_control_traj.flatten() - barrier_control_traj.flatten(),
                                    2) ** 2 + np.linalg.norm(
            true_coc_state_traj.flatten() - barrier_state_traj.flatten(), 2) ** 2

        traj_error_list += [traj_error]

        # compute the gradient error
        grad_traj_error = 0
        for t in range(len(barrier_grad_state)):
            grad_traj_error += np.linalg.norm(barrier_grad_state[t].flatten() - true_grad_state[t].flatten(),2)**2
        for t in range(len(barrier_grad_control)):
            grad_traj_error+=np.linalg.norm(barrier_grad_control[t].flatten()-true_grad_control[t].flatten(),2)**2

        grad_traj_error_list += [grad_traj_error]

    fig = plt.figure(1, figsize=(9, 8))
    ax = fig.subplots(2, 2)

    ax[0,0].plot(gamma_list, traj_error_list, linewidth=5, color='tab:brown', marker='o', markersize=0)

    ax[0,0].set_xlabel(r'$\gamma$')
    ax[0,0].set_ylabel(r'$||\xi_{(\theta,\gamma)}-\xi_{\theta}||^2$')
    ax[0,0].set_xscale('log')
    ax[0,0].set_xlim([1e-1, 1e-4])
    ax[0,0].set_facecolor('#EFEFEF')
    ax[0,0].grid()


    ax[0,1].plot(gamma_list, grad_traj_error_list, linewidth=5, color='tab:brown', marker='o', markersize=0)

    ax[0,1].set_xlabel(r'$\gamma$')
    ax[0,1].set_ylabel(r'$||\frac{\partial \xi_{(\theta,\gamma)}}{\partial \theta}-\frac{\partial \xi_{\theta}}{\partial \theta}||^2$')
    ax[0,1].set_xscale('log')
    ax[0,1].set_xlim([1e-1, 1e-4])
    ax[0,1].set_facecolor('#ECECEC')
    ax[0,1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax[0,1].grid()


    # -----------------------plot trajecotry -----------------------
    # gamma_list = []
    gamma_list = np.linspace(0.5, 1e-1, 25).tolist()
    gamma_list += np.linspace(1e-1, 1e-4, 5).tolist()
    # gamma_list = [0.5,  0.3, 0.2, 0.15, 0.119, 0.1, 0.01, 0.001, ]
    barrier_oc_sol_list = []
    for gamma in gamma_list:
        coc.convert2BarrierOC(gamma=gamma)
        barrier_oc_sol = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=true_theta)
        barrier_oc_sol_list += [barrier_oc_sol]

    # --------------------------control input plot ---------------
    colors = list(Color("lightblue").range_to(Color("royalblue"), len(barrier_oc_sol_list)))
    true_coc_control_traj, = ax[1,0].plot(coc_sol['control_traj_opt'], linewidth=4, color='tab:red', zorder=10,
                                      linestyle=':', marker='o', markersize=0)

    boundu, = ax[1,0].plot(time_grid, max_u * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax[1,0].plot(time_grid, -max_u * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax[1,0].fill_between(time_grid, max_u, -max_u, color='#EFEFEF', alpha=1)

    for count, barrier_oc_sol in enumerate(barrier_oc_sol_list):
        ax[1,0].plot(barrier_oc_sol['control_traj_opt'], linewidth=4, color=colors[count].hex, )

    # legend
    sgamma_control_traj, = ax[1,0].plot(barrier_oc_sol_list[0]['control_traj_opt'], linewidth=4, color=colors[0].hex,
                                    zorder=-100)

    fgamma_control_traj, = ax[1,0].plot(barrier_oc_sol_list[-1]['control_traj_opt'], linewidth=4, color=colors[-1].hex,
                                    zorder=-100)

    ax[1,0].legend([sgamma_control_traj, true_coc_control_traj, fgamma_control_traj, boundu],
               [r'$\gamma=0.5$', 'true sol.', r'$\gamma=10^{-4}$', 'constraint'], ncol=2, prop={'size': 16},
               columnspacing=1.0, handlelength=0.8, bbox_to_anchor=(0.535, 0.54, 0.5, 0.5)).set_zorder(-102)

    ax[1,0].set_xlabel('Time')
    ax[1,0].grid(alpha=0.5)
    ax[1,0].set_xticks(np.arange(0, horizon + 1, 5))
    ax[1,0].set_ylabel('Cart force input', labelpad=0)
    ax[1,0].set_ylim([-6, 8.5])

    # ------------------- cart position plot

    colors2 = list(Color("navajowhite").range_to(Color("darkorange"), len(barrier_oc_sol_list)))
    true_coc_position_traj, = ax[1,1].plot(coc_sol['state_traj_opt'][:, 0], linewidth=4, color='tab:red', zorder=10,
                                       linestyle=':', marker='o', markersize=0)
    boundx, = ax[1,1].plot(time_grid, max_x * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax[1,1].plot(time_grid, -max_x * np.ones_like(time_grid), '--', linewidth=3, color='k', zorder=100)
    ax[1,1].fill_between(time_grid, max_x, -max_x, color='#EFEFEF', alpha=1)

    for count, barrier_oc_sol in enumerate(barrier_oc_sol_list):
        ax[1,1].plot(barrier_oc_sol['state_traj_opt'][:, 0], linewidth=4, color=colors2[count].hex)

    # legend
    sgamma_position_traj, = ax[1,1].plot(barrier_oc_sol_list[0]['state_traj_opt'][:, 0], linewidth=4,
                                     color=colors2[0].hex, zorder=-100)
    fgamma_position_traj, = ax[1,1].plot(barrier_oc_sol_list[-1]['state_traj_opt'][:, 0], linewidth=4,
                                     color=colors2[-1].hex, zorder=-100)

    ax[1,1].legend([sgamma_position_traj, true_coc_control_traj, fgamma_position_traj, boundx],
               [r'$\gamma=0.5$', 'true sol.', r'$\gamma=10^{-4}$', 'constraint'], ncol=2, prop={'size': 16},
               columnspacing=1.0, handlelength=0.8, bbox_to_anchor=(0.535, 0.54, 0.5, 0.5)).set_zorder(-102)

    # plt.legend([true_coc_control_traj,oc_control_traj],['constrained sol', 'unconstrained sol'])

    ax[1,1].set_xlabel('Time')
    ax[1,1].grid(alpha=0.5)
    ax[1,1].set_xticks(np.arange(0, horizon + 1, 5))
    ax[1,1].set_ylabel('Cart position', labelpad=0)
    ax[1,1].set_ylim([-1.5, 2])



    plt.subplots_adjust(left=0.12, right=0.96, bottom=0.1, top=0.95, wspace=0.36, hspace=0.35)
    plt.show()