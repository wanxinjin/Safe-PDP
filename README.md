# Safe Pontryagin Differentiable Programming

This code project is an implementation of Safe Pontryagin Differentiable Programming (Safe PDP) methodology. Safe PDP is
a new theoretical and algorithmic safe differentiable framework that can solve a broad class of safety-critical
learning and control tasks---problems that require the guarantee of both immediate and long-term constraint satisfaction
at any stage of the learning and control progress. Safe PDP has the following key features:

- Safe PDP provides a systematic way for handling __different types of constraints__, including state and inputs (or
  mixed), immediate, and long-term constraints in a  learning and control system.

- Safe PDP attains the __provable safety- and optimality- guarantees__ throughout the learning and control process,
  guaranteeing strict constraint satisification throughout the entire learning and control process.

- Safe PDP establishes a __unified differentiable programming__ framework to efficiently solve a board class of
  safety-critical learning and control tasks, such as safe policy optimization, safe motion planning, and safe learning
  from demonstrations, etc.

Please find the technical details of Safe PDP in our paper:
https://arxiv.org/abs/2105.14937,
by [Wanxin Jin](https://wanxinjin.github.io/) (Purdue University),
[Shaoshuai Mou](https://engineering.purdue.edu/AAE/people/ptProfile?resource_id=124981) (Purdue University),
[George J. Pappas](https://www.georgejpappas.org/) (University of Pennsylvania)

## 1. Project Overview

The current version of the code project includes the following folders.

- **SafePDP**:  an independent package for core implementation of Safe PDP. This package includes two modules:
    - `SafePDP.py` the core module that implements the Safe PDP for constrained learning and control tasks
    - `PDP.py` the core module that implements the PDP (NeurIPS
      2020, [link](https://proceedings.neurips.cc/paper/2020/file/5a7b238ba0f6502e5d6be14424b20ded-Paper.pdf))
      for unconstrained learning and control tasks


- **Examples**:  a folder containing different applications of Safe PDP, including
    - *SPO*:  examples of applying Safe PDP for safe policy optimization.
    - _LQR_: examples of applying Safe PDP for safe motion planning.
    - *MPC*: examples of applying Safe PDP for
        - *CODE*: learning constrained ODEs from demonstrations
        - *CIOC*: jointly learning dynamics, state and input constraints, control cost from demonstrations.

  Note that all the above examples are presented in the Safe PDP paper. Please find the experiment details in the Safe PDP paper.



- **JinEnv**: an independent package, containing a single module, `JinEnv.py`, which implements different control
  environments and its animation.  More information about how to use `JinEnv.py`, please refer to 
  [link](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming/tree/master/JinEnv). 
  Compared to the previous version of `JinEnv.py`,  we here have added some interfaces to each environment to allow the user add the constraints to the environment.


- **ControlTools**: an independent package, containing a single module `ControlTools.py`, which implements some popular
  control algorithms, including LQR, iLQR, system identification (DMD algorithms), ALTRO trajectory optimization, etc.
  Note that this package is irrelevant to Safe PDP, and It is included here mainly for convenience of users
  (who may use these algorithms for benchmark comparison).

## 2. Dependency

* CasADi: version >= 3.5.1. Info: https://web.casadi.org/
* Numpy: version >= 1.18.1. Info: https://numpy.org/
* Matplotlib: version >= 3.3.3. Info: https://matplotlib.org/
* Python: version >= 3.7. Info: https://www.python.org/downloads/

Note: before you try Safe PDP and JinEnv Packages, we strongly recommend you to familiarize yourself with the CasADi
programming language, e.g., how to define a symbolic expression/function. Reading Sections 2, 3, 4 on the
page  https://web.casadi.org/docs/ is enough (around 30 mins)!
Because this really helps you to debug and understand the codes here.

The codes have been tested and run smoothly with Python 3.7. on macOS Big Sur (11.2.3) machine.

## 3. How to Use the SafePDP Package

First, be relaxed :). We have optimized the interface of the Safe PDP package, which hopefully minimizes your burden to
understand and use them. All variables/functions have pretty straightforward names, and most of the lines are carefully
commented! In most of the cases, all you need to do is to specify/connect your control system from `JinEnv.py` environment
to `SafePDP.py`, then Safe PDP will take care of the rest.

The quickest way to start is to read and run each example in the **Example** folder, e.g.,

* Read and run Examples/SPO/`SPO_Cartpole.py` --- you will understand how to use SafePDP to solve safe policy
  optimization.
* Read and run Examples/SPO/`PO_Cartpole.py` --- you will understand how to use SafePDP to solve unconstrained policy
  optimization.


* Read and run Examples/SPlan/`SPlan_Cartpole.py` --- you will understand how to use SafePDP to solve safe motion
  planning.
* Read and run Examples/Plan/`Plan_Cartpole.py` --- you will understand how to use SafePDP to solve unconstrained motion
  planning.


* Read and run Examples/MPC/CIOC/`CIOC_Cartpole.py` --- you will understand how to use SafePDP to jointly learn
  dynamics, constraints, and control cost from demonstrations.
* Read and run Examples/MPC/CODE/`CODE_Cartpole.py` --- you will understand how to use SafePDP to learning dynamics and
  constraints from demonstrations.

In SafePDP package, the instructions for the interfaces within `PDP.py` module can be found
at [here](https://github.com/wanxinjin/Pontryagin-Differentiable-Programming). The following briefly describes the
interfaces within `SafePDP.py` module.

* `Class COCsys`: this class has the following functionalities (interfaces):
    * construct a (parameterized) constrained optimal control agent, i.e., Equ. (1) in the Safe PDP paper,
    * solving a constrained optimal control problem by an OC solver,
    * differentiate the C-PMP and establish the auxiliary control system, i.e.,Theorem 1 in Safe PDP paper,
    * convert the constrained optimal control agent into an unconstrained one, i.e., Equ. (5) in Safe PDP paper,
    * Approximate both trajectory and its derivative using safe unconstrained couterparts, i.e., Theorem 2 in Safe PDP paper.


* `Class EQCLQR`: this class is an implementation of an equality-constrained LQR solver, which is based on the paper of Efficient Computation of Feedback
  Control for Constrained Systems by F. Laine and C. Tomlin.


* `Class CSysOPT`: this class particularly focuses on a special case of Safe PDP: for a constrained optimal control agent,
  the control cost and the constraints are given, only the agent dynamics model and control policy are parameterized (unknown).
  This class has the following functionalities (interfaces):
    * specify a (parameterized) dynamics model;
    * specify the control cost function;
    * specify the constraints;
    * specify the parametrized control policy: use the neural policy if you are solving the safe policy optimization, 
      and use polynomial policy if you are solving the safe motion planning;
    * convert the constrained optimal control agent into an unconstrained one;
    * optimization via differential C-PMP.
  

_Note that: all interfaces within `SafePDP.py` module are corresponding to the technical details of the Safe PDP paper. Please read the Safe PDP paper 
carefully before understanding these interfaces._


## 4. Citation

If you have difficulty to understand the code, please contact me by email: wanxinjin@gmail.com

If you find this work helpful in your research, please consider citing our paper: https://arxiv.org/abs/2105.14937.



