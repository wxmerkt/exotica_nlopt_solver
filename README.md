# EXOTica NLopt Motion Solvers

This repository contains motion solvers for the EXOTica framework based on NLopt.

As NLopt is a dense nonlinear programming solver, we only provide solvers for optimization-based inverse kinematics (end-pose) problems at this stage.

## Available solvers and options

We provide three base solvers for the main types of EXOTica EndPose problems:

- `NLoptUnconstrainedEndPoseSolver` to solve unconstrained end-pose problems (`UnconstrainedEndPoseProblem`)
- `NLoptBoundedEndPoseSolver` to solve end-pose problems with costs and bounds on the configuration variables (`BoundedEndPoseProblem`). Note, that a special solver setting exists to also restrict the joint velocities for use in interactive inverse kinematics (`dt`, `BoundVelocities`)
- `NLoptEndPoseSolver` to solve end-pose problems with costs, variable bounds, and general equality and inequality constraints (`EndPoseProblem`). This solver also supports joint velocity limits for interactive inverse kinematics as in the bounded solver. Please note that not all NLopt algorithms support equality and inequality constraints - users are recommended to consult the manual to select the appropriate algorithm. The SLSQP algorithm is generally a good starting point.

A range of options of the base NLopt solvers have been exposed to the initializers - these can be found in `init/NLoptMotionSolver.in`. Notably, all algorithms from NLopt can be specified and used. Users are recommended to consult [NLopt's official documentation](https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/) for choosing the appropriate solver for their problems.

## Examples

We provide examples for each of the solvers following a figure-eight trajectory, if applicable with inequality constraints:

```bash
roslaunch exotica_nlopt_solver python_nlopt_ik_unconstrained.launch
roslaunch exotica_nlopt_solver python_nlopt_ik_bounded.launch
roslaunch exotica_nlopt_solver python_nlopt_ik_constrained.launch
```

and an interactive example to demonstrate velocity limits in interactive IK:

```bash
roslaunch exotica_nlopt_solver python_nlopt_ik_velocity_constrained.launch
```

## Citation

If you use the solvers provided in this repository for academic work, please cite our work using the following reference for which these solvers were originally created:

```bibtex
@phdthesis{merkt2019phd,
  author      = {Wolfgang Merkt},
  school      = {School of Informatics, The University of Edinburgh},
  title       = {Experience-driven optimal motion synthesis in complex and shared environments},
  year        = {2019},
  doi         = {10.7488/era/358}
}
```

as well as [NLopt](https://nlopt.readthedocs.io/en/latest/Citing_NLopt/):

```bibtex
@misc{nlopt,
  title        = {The NLopt nonlinear-optimization package},
  author       = {Steven G. Johnson},
  publisher    = {GitHub},
  howpublished = {\url{http://github.com/stevengj/nlopt}}
}
```
