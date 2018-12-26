/*
 *      Author: Wolfgang Merkt
 *
 * Copyright (c) 2018, Wolfgang Merkt
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of  nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef EXOTICA_NLOPT_SOLVER_NLOPT_SOLVER_H_
#define EXOTICA_NLOPT_SOLVER_NLOPT_SOLVER_H_

#include <iostream>

#include <exotica/MotionSolver.h>
#include <exotica/Problems/BoundedEndPoseProblem.h>
#include <exotica/Problems/EndPoseProblem.h>
#include <exotica/Problems/UnconstrainedEndPoseProblem.h>

#include <exotica_nlopt_solver/NLoptBoundedEndPoseSolverInitializer.h>
#include <exotica_nlopt_solver/NLoptEndPoseSolverInitializer.h>
#include <exotica_nlopt_solver/NLoptUnconstrainedEndPoseSolverInitializer.h>

// Note: We are using the C and not the C++ API as the latter requires copy
// operations for the evaluation functions (in order to use std::vector)
#include <nlopt.h>

namespace exotica
{
template <typename Problem>
double end_pose_problem_objective_func(unsigned n, const double *x,
                                       double *gradient, /* nullptr if not needed */
                                       void *func_data)
{
    Problem *prob = reinterpret_cast<Problem *>(func_data);
    Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(x, n);
    prob->Update(q);
    if (gradient != nullptr)
    {
        auto grad_eigen = Eigen::Map<Eigen::VectorXd>(gradient, n);
        grad_eigen = prob->getScalarJacobian();
    }
    return prob->getScalarCost();
}

template <typename Problem>
void end_pose_problem_inequality_constraint_mfunc(unsigned m, double *result, unsigned n, const double *x, double *gradient, void *func_data)
{
    Problem *prob = reinterpret_cast<Problem *>(func_data);
    Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::VectorXd neq = Eigen::Map<const Eigen::VectorXd>(result, m);
    prob->Update(q);

    neq = prob->getInequality();
    if (gradient != nullptr)
    {
        auto grad_eigen = Eigen::Map<Eigen::MatrixXd>(gradient, n, m);
        grad_eigen = prob->getInequalityJacobian();
    }
}

template <typename Problem>
void end_pose_problem_equality_constraint_mfunc(unsigned m, double *result, unsigned n, const double *x, double *gradient, void *func_data)
{
    Problem *prob = reinterpret_cast<Problem *>(func_data);
    Eigen::VectorXd q = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::VectorXd neq = Eigen::Map<const Eigen::VectorXd>(result, m);
    prob->Update(q);

    neq = prob->getEquality();
    if (gradient != nullptr)
    {
        auto grad_eigen = Eigen::Map<Eigen::MatrixXd>(gradient, n, m);
        grad_eigen = prob->getEqualityJacobian();
    }
}

template <typename Problem, typename ProblemInitializer>
class NLoptGenericEndPoseSolver : public MotionSolver, public Instantiable<ProblemInitializer>
{
public:
    NLoptGenericEndPoseSolver() = default;
    virtual ~NLoptGenericEndPoseSolver() = default;

    void Instantiate(ProblemInitializer &init) override
    {
        HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Instantiating");

        // TODO: Select algorithm
        // From https://github.com/stevengj/nlopt/blob/master/src/api/nlopt.h#L72
        //
        // Naming conventions:
        // NLOPT_{G/L}{D/N}_*
        // = global/local derivative/no-derivative optimization,
        // respectively
        // *_RAND algorithms involve some randomization.
        // *_NOSCAL algorithms are *not* scaled to a unit hypercube
        // (i.e. they are sensitive to the units of x)

        if (init.Algorithm == "NLOPT_GN_DIRECT")
        {
            // From: https://github.com/stevengj/nlopt/blob/1226c1276dacf3687464c65eb165932281493a35/src/algs/direct/README
            //
            // The DIRECT algorithm (DIviding RECTangles) is a derivative-free global
            // optimization algorithm invented by Jones et al.:

            // 	D. R. Jones, C. D. Perttunen, and B. E. Stuckmann,
            // 	"Lipschitzian optimization without the lipschitz constant,"
            // 	J. Optimization Theory and Applications, vol. 79, p. 157 (1993).

            // This is a deterministic-search algorithm based on systematic division
            // of the search domain into smaller and smaller hyperrectangles.

            // The implementation is based on the 1998-2001 Fortran version by
            // J. M. Gablonsky at North Carolina State University, converted to C by
            // Steven G. Johnson.  The Fortran source was downloaded from:

            // 	http://www4.ncsu.edu/~ctk/SOFTWARE/DIRECTv204.tar.gz
            algorithm_ = nlopt_algorithm::NLOPT_GN_DIRECT;
        }
        else if (init.Algorithm == "NLOPT_GN_DIRECT_L")
        {
            // From: https://github.com/stevengj/nlopt/blob/1226c1276dacf3687464c65eb165932281493a35/src/algs/direct/README
            //
            // Gablonsky et al implemented a modified version of the original DIRECT
            // algorithm, as described in:

            // 	J. M. Gablonsky and C. T. Kelley, "A locally-biased form
            // 	of the DIRECT algorithm," J. Global Optimization 21 (1),
            // 	p. 27-37 (2001).

            // Both the original Jones algorithm (NLOPT_GN_DIRECT) and the
            // Gablonsky modified version (NLOPT_GN_DIRECT_L) are implemented
            // and available from the NLopt interface.  The Gablonsky version
            // makes the algorithm "more biased towards local search" so that it
            // is more efficient for functions without too many local minima.

            // Also, Gablonsky et al. extended the algorithm to handle "hidden
            // constraints", i.e. arbitrary nonlinear constraints.  In NLopt, a
            // hidden constraint is represented by returning NaN (or Inf, or
            // HUGE_VAL) from the objective function at any points violating the
            // constraint.
            algorithm_ = nlopt_algorithm::NLOPT_GN_DIRECT_L;
        }
        // NLOPT_GN_DIRECT_L_RAND,
        // NLOPT_GN_DIRECT_NOSCAL,
        // NLOPT_GN_DIRECT_L_NOSCAL,
        // NLOPT_GN_DIRECT_L_RAND_NOSCAL,

        // NLOPT_GN_ORIG_DIRECT,
        // NLOPT_GN_ORIG_DIRECT_L,

        // NLOPT_GD_STOGO,
        else if (init.Algorithm == "NLOPT_GD_STOGO")
        {
            // From: https://github.com/stevengj/nlopt/blob/master/src/algs/stogo/README
            //
            // StoGO uses a gradient-based direct-search branch-and-bound algorithm,
            // described in:

            // S. Gudmundsson, "Parallel Global Optimization," M.Sc. Thesis, IMM,
            // 	Technical University of Denmark, 1998.

            // K. Madsen, S. Zertchaninov, and A. Zilinskas, "Global Optimization
            // 	using Branch-and-Bound," Submitted to the Journal of Global
            // 	Optimization, 1998.
            // 	[ never published, but preprint is included as paper.pdf ]

            // S. Zertchaninov and K. Madsen, "A C++ Programme for Global Optimization,"
            // 	IMM-REP-1998-04, Department of Mathematical Modelling,
            // 	Technical University of Denmark, DK-2800 Lyngby, Denmark, 1998.
            // 	[ included as techreport.pdf ]
            algorithm_ = nlopt_algorithm::NLOPT_GD_STOGO;
        }
        // NLOPT_GD_STOGO_RAND,

        // NLOPT_LD_LBFGS_NOCEDAL,

        // NLOPT_LD_LBFGS,

        // NLOPT_LN_PRAXIS,

        // NLOPT_LD_VAR1,
        // NLOPT_LD_VAR2,

        // NLOPT_LD_TNEWTON,
        else if (init.Algorithm == "NLOPT_LD_TNEWTON")
        {
            algorithm_ = nlopt_algorithm::NLOPT_LD_TNEWTON;
        }
        // NLOPT_LD_TNEWTON_RESTART,
        // NLOPT_LD_TNEWTON_PRECOND,
        // NLOPT_LD_TNEWTON_PRECOND_RESTART,

        // NLOPT_GN_CRS2_LM,

        // NLOPT_GN_MLSL,
        // NLOPT_GD_MLSL,
        // NLOPT_GN_MLSL_LDS,
        // NLOPT_GD_MLSL_LDS,

        // NLOPT_LD_MMA,

        // NLOPT_LN_COBYLA,

        // NLOPT_LN_NEWUOA,
        // NLOPT_LN_NEWUOA_BOUND,

        // NLOPT_LN_NELDERMEAD,
        // NLOPT_LN_SBPLX,

        // NLOPT_LN_AUGLAG,
        // NLOPT_LD_AUGLAG,
        // NLOPT_LN_AUGLAG_EQ,
        // NLOPT_LD_AUGLAG_EQ,
        else if (init.Algorithm == "NLOPT_LD_AUGLAG")
        {
            algorithm_ = nlopt_algorithm::NLOPT_LD_AUGLAG;
            local_optimizer_ = nlopt_algorithm::NLOPT_LD_MMA; //NLOPT_LD_TNEWTON;
        }

        // NLOPT_LN_BOBYQA,

        // NLOPT_GN_ISRES,

        // // new variants that require local_optimizer to be set,
        //     // not with older constants for backwards compatibility
        //     NLOPT_AUGLAG,
        // NLOPT_AUGLAG_EQ,
        // NLOPT_G_MLSL,
        // NLOPT_G_MLSL_LDS,

        // NLOPT_LD_SLSQP,

        // NLOPT_LD_CCSAQ,

        // NLOPT_GN_ESCH,

        else
        {
            throw_pretty("Selected algorithm " << init.Algorithm << " is not supported.");
        }
    }

    void specifyProblem(PlanningProblem_ptr pointer) override
    {
        if (pointer->type().find("EndPoseProblem") == std::string::npos)
        {
            throw_named("NLoptGenericEndPoseSolver can't solve problem of type '"
                        << pointer->type() << "'!");
        }
        MotionSolver::specifyProblem(pointer);
        prob_ = std::static_pointer_cast<Problem>(pointer);
    }

    void Solve(Eigen::MatrixXd &solution) override
    {
        Timer timer;
        planning_time_ = -1;
        prob_->preupdate();

        if (!prob_)
            throw_named("Solver has not been initialized!");
        Eigen::VectorXd q0 = prob_->applyStartState();
        if (prob_->N != q0.rows())
            throw_named("Wrong size q0 size=" << q0.rows()
                                              << ", required size=" << prob_->N);
        solution.resize(1, prob_->N);
        // prob_->resetCostEvolution(parameters.iterations + 1);
        // prob_->setCostEvolution(0, f(q0));

        // Create optimisation solver
        nlopt_opt opt = nlopt_create(algorithm_, prob_->N);

        // If problem supports bounds, set bounds
        set_bounds(opt);

        // If problem supports constraints, set constraints
        set_constraints(opt);

        // Set minimization objective
        {
            nlopt_result info = nlopt_set_min_objective(opt, &end_pose_problem_objective_func<Problem>, (void *)prob_.get());
            if (info != 1) WARNING("Error while setting objective function: " << (int)info << ": " << nlopt_get_errmsg(opt));
        }

        // Set tolerances
        set_tolerances(opt);

        // Create and assign local optimizer, if required
        nlopt_opt local_opt = nullptr;
        if (local_optimizer_ != nlopt_algorithm::NLOPT_NUM_ALGORITHMS)
        {
            local_opt = nlopt_create(local_optimizer_, prob_->N);
            nlopt_result info = nlopt_set_local_optimizer(opt, local_opt);
            set_bounds(local_opt);
            set_tolerances(local_opt);
            if (info != 1) WARNING("Error while setting local optimizer: " << (int)info << ": " << nlopt_get_errmsg(opt));
        }

        // Record the cost value for the initial solution
        double initial_cost_value = end_pose_problem_objective_func<Problem>(prob_->N, q0.data(), nullptr, (void *)prob_.get());

        // Scale the initial step size (only for derivative-free algorithms)
        constexpr double initial_step_scale = 1.0;
        Eigen::VectorXd initial_step = Eigen::VectorXd::Zero(prob_->N);
        nlopt_get_initial_step(opt, q0.data(), initial_step.data());
        initial_step *= initial_step_scale;
        nlopt_set_initial_step(opt, initial_step.data());

        // Run the optimization
        double final_cost_value;
        nlopt_result info = nlopt_optimize(opt, q0.data(), &final_cost_value);
        if (info < 0)
            WARNING("Optimization did not exit cleanly! " << (int)info);

        solution.row(0) = q0;
        planning_time_ = timer.getDuration();

        // Show statistics if "verbose" is set as true
        if (debug_)
        {
            HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Info: " << (int)info);
            std::cout << "------ nlopt ------" << std::endl;
            std::cout << "Dimensions     : " << prob_->N << std::endl;
            std::cout << "Function value : " << initial_cost_value << " => " << final_cost_value << std::endl;
            std::cout << "Elapsed time   : " << planning_time_ << " [s]" << std::endl;
            std::cout << "--------------------" << std::endl;
        }

        // Destroy/clean-up
        nlopt_destroy(opt);
        if (local_opt) nlopt_destroy(local_opt);
    }

protected:
    std::shared_ptr<Problem> prob_;  // Shared pointer to the planning problem.

    nlopt_algorithm algorithm_ = nlopt_algorithm::NLOPT_NUM_ALGORITHMS;        ///< Selected optimization algorithm.
    nlopt_algorithm local_optimizer_ = nlopt_algorithm::NLOPT_NUM_ALGORITHMS;  ///< Local optimization, if required by the selected algorithm.

    virtual void set_bounds(nlopt_opt my_opt) {}       // To be reimplemented in bounded problems
    virtual void set_constraints(nlopt_opt my_opt) {}  // To be reimplemented in constrained problems

    void set_tolerances(nlopt_opt my_opt)
    {
        // nlopt_set_maxeval(opt, getNumberOfMaxIterations());  // Note: Not strictly true - this is function evaluations and not iterations...
        // nlopt_set_ftol_rel(my_opt, 1e-6);  // TODO: Make parameters
        // nlopt_set_xtol_rel(my_opt, 1e-6);  // TODO: Make parameters
        // nlopt_set_ftol_abs(my_opt, 1e-9);
    }
};

// TODO: This could likely be made much neater without copy-paste with some
// combination of std::enable_if and std::is_same.
template <>
void NLoptGenericEndPoseSolver<BoundedEndPoseProblem, NLoptBoundedEndPoseSolverInitializer>::set_bounds(nlopt_opt my_opt)
{
    const Eigen::VectorXd lower_bounds = prob_->getBounds().col(0);
    const Eigen::VectorXd upper_bounds = prob_->getBounds().col(1);
    {
        nlopt_result info = nlopt_set_lower_bounds(my_opt, lower_bounds.data());
        if (info != 1) WARNING("Error while setting lower bounds: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
    }
    {
        nlopt_result info = nlopt_set_upper_bounds(my_opt, upper_bounds.data());
        if (info != 1) WARNING("Error while setting upper bounds: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
    }
}

template <>
void NLoptGenericEndPoseSolver<EndPoseProblem, NLoptEndPoseSolverInitializer>::set_bounds(nlopt_opt my_opt)
{
    const Eigen::VectorXd lower_bounds = prob_->getBounds().col(0);
    const Eigen::VectorXd upper_bounds = prob_->getBounds().col(1);
    {
        nlopt_result info = nlopt_set_lower_bounds(my_opt, lower_bounds.data());
        if (info != 1) WARNING("Error while setting lower bounds: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
    }
    {
        nlopt_result info = nlopt_set_upper_bounds(my_opt, upper_bounds.data());
        if (info != 1) WARNING("Error while setting upper bounds: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
    }
}

template <>
void NLoptGenericEndPoseSolver<EndPoseProblem, NLoptEndPoseSolverInitializer>::set_constraints(nlopt_opt my_opt)
{
    {
        if (!my_opt) throw_pretty("opt is dead");
        const unsigned int &m_neq = prob_->Inequality.PhiN;
        if (m_neq > 0)
        {
            const Eigen::VectorXd tol_neq = prob_->init_.InequalityFeasibilityTolerance * Eigen::VectorXd::Ones(m_neq);
            nlopt_result info = nlopt_add_inequality_mconstraint(my_opt, m_neq, &end_pose_problem_inequality_constraint_mfunc<EndPoseProblem>, (void *)prob_.get(), tol_neq.data());
            if (info != 1) WARNING("Error while setting inequality constraints: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
        }
    }

    {
        const unsigned int &m_eq = prob_->Equality.PhiN;
        if (m_eq > 0)
        {
            const Eigen::VectorXd tol_eq = prob_->init_.EqualityFeasibilityTolerance * Eigen::VectorXd::Ones(m_eq);
            nlopt_result info = nlopt_add_equality_mconstraint(my_opt, m_eq, &end_pose_problem_equality_constraint_mfunc<EndPoseProblem>, (void *)prob_.get(), tol_eq.data());
            if (info != 1) WARNING("Error while setting equality constraints: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
        }
    }
}

typedef NLoptGenericEndPoseSolver<BoundedEndPoseProblem, NLoptBoundedEndPoseSolverInitializer> NLoptBoundedEndPoseSolver;
typedef NLoptGenericEndPoseSolver<EndPoseProblem, NLoptEndPoseSolverInitializer> NLoptEndPoseSolver;
typedef NLoptGenericEndPoseSolver<UnconstrainedEndPoseProblem, NLoptUnconstrainedEndPoseSolverInitializer> NLoptUnconstrainedEndPoseSolver;
}  // namespace exotica

#endif  // EXOTICA_NLOPT_SOLVER_NLOPT_SOLVER_H_