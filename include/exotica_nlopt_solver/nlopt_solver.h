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
#include <memory>
#include <set>
#include <unordered_map>

#include <exotica_core/motion_solver.h>
#include <exotica_core/problems/bounded_end_pose_problem.h>
#include <exotica_core/problems/end_pose_problem.h>
#include <exotica_core/problems/unconstrained_end_pose_problem.h>

#include <exotica_nlopt_solver/NLoptBoundedEndPoseSolver_initializer.h>
#include <exotica_nlopt_solver/NLoptEndPoseSolver_initializer.h>
#include <exotica_nlopt_solver/NLoptUnconstrainedEndPoseSolver_initializer.h>

// Note: We are using the C and not the C++ API as the latter requires copy
// operations for the evaluation functions (in order to use std::vector)
#include <nlopt.h>

namespace exotica
{
template <typename Problem>
struct ProblemWrapperData
{
    ProblemWrapperData(Problem *problem_in) : problem(problem_in), q(Eigen::VectorXd::Zero(problem->N)) {}
    int objective_function_evaluations = 0;
    int inequality_function_evaluations = 0;
    int equality_function_evaluations = 0;
    Problem *problem = nullptr;
    Eigen::VectorXd q;

    void reset()
    {
        objective_function_evaluations = 0;
        inequality_function_evaluations = 0;
        equality_function_evaluations = 0;
    }
};

template <typename Problem>
double end_pose_problem_objective_func(unsigned n, const double *x,
                                       double *gradient, /* nullptr if not needed */
                                       void *func_data)
{
    ProblemWrapperData<Problem> *data = reinterpret_cast<ProblemWrapperData<Problem> *>(func_data);
    Problem *prob = reinterpret_cast<Problem *>(data->problem);
    Eigen::Map<const Eigen::VectorXd> q = Eigen::Map<const Eigen::VectorXd>(x, n);
    prob->Update(q);
    if (gradient != nullptr)
    {
        Eigen::Map<Eigen::RowVectorXd> grad_eigen = Eigen::Map<Eigen::RowVectorXd>(gradient, n);
        grad_eigen = prob->GetScalarJacobian();
    }
    ++data->objective_function_evaluations;
    return prob->GetScalarCost();
}

template <typename Problem>
void end_pose_problem_inequality_constraint_mfunc(unsigned m, double *result, unsigned n, const double *x, double *gradient, void *func_data)
{
    ProblemWrapperData<Problem> *data = reinterpret_cast<ProblemWrapperData<Problem> *>(func_data);
    Problem *prob = reinterpret_cast<Problem *>(data->problem);
    Eigen::Map<const Eigen::VectorXd> q = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::Map<Eigen::VectorXd> neq = Eigen::Map<Eigen::VectorXd>(result, m);
    prob->Update(q);

    neq = prob->GetInequality();
    if (gradient != nullptr)
    {
        Eigen::Map<Eigen::MatrixXd> grad_eigen = Eigen::Map<Eigen::MatrixXd>(gradient, m, n);
        grad_eigen = prob->GetInequalityJacobian();
    }
    ++data->inequality_function_evaluations;
}

template <typename Problem>
void end_pose_problem_equality_constraint_mfunc(unsigned m, double *result, unsigned n, const double *x, double *gradient, void *func_data)
{
    ProblemWrapperData<Problem> *data = reinterpret_cast<ProblemWrapperData<Problem> *>(func_data);
    Problem *prob = reinterpret_cast<Problem *>(data->problem);
    Eigen::Map<const Eigen::VectorXd> q = Eigen::Map<const Eigen::VectorXd>(x, n);
    Eigen::Map<Eigen::VectorXd> eq = Eigen::Map<Eigen::VectorXd>(result, m);
    prob->Update(q);

    eq = prob->GetEquality();
    if (gradient != nullptr)
    {
        Eigen::Map<Eigen::MatrixXd> grad_eigen = Eigen::Map<Eigen::MatrixXd>(gradient, m, n);
        grad_eigen = prob->GetEqualityJacobian();
    }
    ++data->equality_function_evaluations;
}

static inline std::string get_result_info(const nlopt_result &info)
{
    // or use nlopt_result_to_string
    switch (info)
    {
        case nlopt_result::NLOPT_FAILURE:
            return "generic failure";
        case nlopt_result::NLOPT_INVALID_ARGS:
            return "invalid arguments";
        // case nlopt_result::NLOPT_OUT_OF_MEMROY:
        //     return "out of memory";
        case nlopt_result::NLOPT_ROUNDOFF_LIMITED:
            return "round off limited";
        case nlopt_result::NLOPT_FORCED_STOP:
            return "forced stop";
        case nlopt_result::NLOPT_SUCCESS:
            return "success";
        case nlopt_result::NLOPT_STOPVAL_REACHED:
            return "stop val reached";
        case nlopt_result::NLOPT_FTOL_REACHED:
            return "ftol reached";
        case nlopt_result::NLOPT_XTOL_REACHED:
            return "xtol reached";
        case nlopt_result::NLOPT_MAXEVAL_REACHED:
            return "max eval reached";
        case nlopt_result::NLOPT_MAXTIME_REACHED:
            return "max time reached";
        default:
            return "unknown?!";
    }
}

template <typename Problem, typename ProblemInitializer>
class NLoptGenericEndPoseSolver : public MotionSolver, public Instantiable<ProblemInitializer>
{
public:
    NLoptGenericEndPoseSolver() = default;
    virtual ~NLoptGenericEndPoseSolver() = default;

    void Instantiate(const ProblemInitializer &init) override
    {
        this->parameters_ = init;

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
        const std::unordered_map<std::string, nlopt_algorithm> algorithm_map = {
            {"NLOPT_GN_DIRECT", NLOPT_GN_DIRECT},
            {"NLOPT_GN_DIRECT_L", NLOPT_GN_DIRECT_L},
            {"NLOPT_GN_DIRECT_L_RAND", NLOPT_GN_DIRECT_L_RAND},
            {"NLOPT_GN_DIRECT_NOSCAL", NLOPT_GN_DIRECT_NOSCAL},
            {"NLOPT_GN_DIRECT_L_NOSCAL", NLOPT_GN_DIRECT_L_NOSCAL},
            {"NLOPT_GN_DIRECT_L_RAND_NOSCAL", NLOPT_GN_DIRECT_L_RAND_NOSCAL},

            {"NLOPT_GN_ORIG_DIRECT", NLOPT_GN_ORIG_DIRECT},
            {"NLOPT_GN_ORIG_DIRECT_L", NLOPT_GN_ORIG_DIRECT_L},

            {"NLOPT_GD_STOGO", NLOPT_GD_STOGO},
            {"NLOPT_GD_STOGO_RAND", NLOPT_GD_STOGO_RAND},

            {"NLOPT_LD_LBFGS_NOCEDAL", NLOPT_LD_LBFGS_NOCEDAL},

            {"NLOPT_LD_LBFGS", NLOPT_LD_LBFGS},

            {"NLOPT_LN_PRAXIS", NLOPT_LN_PRAXIS},

            {"NLOPT_LD_VAR1", NLOPT_LD_VAR1},
            {"NLOPT_LD_VAR2", NLOPT_LD_VAR2},

            {"NLOPT_LD_TNEWTON", NLOPT_LD_TNEWTON},
            {"NLOPT_LD_TNEWTON_RESTART", NLOPT_LD_TNEWTON_RESTART},
            {"NLOPT_LD_TNEWTON_PRECOND", NLOPT_LD_TNEWTON_PRECOND},
            {"NLOPT_LD_TNEWTON_PRECOND_RESTART", NLOPT_LD_TNEWTON_PRECOND_RESTART},

            {"NLOPT_GN_CRS2_LM", NLOPT_GN_CRS2_LM},

            {"NLOPT_GN_MLSL", NLOPT_GN_MLSL},
            {"NLOPT_GD_MLSL", NLOPT_GD_MLSL},
            {"NLOPT_GN_MLSL_LDS", NLOPT_GN_MLSL_LDS},
            {"NLOPT_GD_MLSL_LDS", NLOPT_GD_MLSL_LDS},

            {"NLOPT_LD_MMA", NLOPT_LD_MMA},

            {"NLOPT_LN_COBYLA", NLOPT_LN_COBYLA},

            {"NLOPT_LN_NEWUOA", NLOPT_LN_NEWUOA},
            {"NLOPT_LN_NEWUOA_BOUND", NLOPT_LN_NEWUOA_BOUND},

            {"NLOPT_LN_NELDERMEAD", NLOPT_LN_NELDERMEAD},
            {"NLOPT_LN_SBPLX", NLOPT_LN_SBPLX},

            {"NLOPT_LN_AUGLAG", NLOPT_LN_AUGLAG},
            {"NLOPT_LD_AUGLAG", NLOPT_LD_AUGLAG},
            {"NLOPT_LN_AUGLAG_EQ", NLOPT_LN_AUGLAG_EQ},
            {"NLOPT_LD_AUGLAG_EQ", NLOPT_LD_AUGLAG_EQ},

            {"NLOPT_LN_BOBYQA", NLOPT_LN_BOBYQA},

            {"NLOPT_GN_ISRES", NLOPT_GN_ISRES},

            /* new variants that require local_optimizer to be set, not with older constants for backwards compatibility */
            {"NLOPT_AUGLAG", NLOPT_AUGLAG},
            {"NLOPT_AUGLAG_EQ", NLOPT_AUGLAG_EQ},
            {"NLOPT_G_MLSL", NLOPT_G_MLSL},
            {"NLOPT_G_MLSL_LDS", NLOPT_G_MLSL_LDS},

            {"NLOPT_LD_SLSQP", NLOPT_LD_SLSQP},

            {"NLOPT_LD_CCSAQ", NLOPT_LD_CCSAQ},

            {"NLOPT_GN_ESCH", NLOPT_GN_ESCH},

            {"NLOPT_GN_AGS", NLOPT_GN_AGS}};

        const std::set<std::string> requires_local_optimizer = {"NLOPT_AUGLAG", "NLOPT_AUGLAG_EQ", "NLOPT_G_MLSL", "NLOPT_G_MLSL_LDS"};

        auto it = algorithm_map.find(init.Algorithm);
        if (it != algorithm_map.end())
        {
            if (debug_) HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Initialising " << it->first);
            algorithm_ = it->second;

            if (requires_local_optimizer.count(it->first) != 0)
            {
                if (init.LocalOptimizer != "")
                {
                    auto local_optimizer_it = algorithm_map.find(init.LocalOptimizer);
                    if (local_optimizer_it != algorithm_map.end())
                    {
                        local_optimizer_ = local_optimizer_it->second;
                        if (debug_) HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Setting local optimizer to " << local_optimizer_it->first);
                    }
                    else
                    {
                        ThrowPretty("Selected local optimizer '" << init.LocalOptimizer << "' does not exist.");
                    }
                }
                else
                {
                    // Default local optimizer
                    if (debug_) HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Selecting default local optimizer");
                    local_optimizer_ = nlopt_algorithm::NLOPT_LD_MMA;  //NLOPT_LD_TNEWTON;
                }
            }
            else
            {
                // Warn if a local optimizer is specified but not required.
                if (init.LocalOptimizer != "") WARNING("The selected algorithm does not require a local optimizer, ignoring.");
            }
        }
        else
        {
            ThrowPretty("Selected algorithm " << init.Algorithm << " is not supported.");
        }
    }

    void SpecifyProblem(PlanningProblemPtr pointer) override
    {
        if (pointer->type().find("EndPoseProblem") == std::string::npos)
        {
            ThrowNamed("NLoptGenericEndPoseSolver can't solve problem of type '"
                       << pointer->type() << "'!");
        }
        MotionSolver::SpecifyProblem(pointer);
        prob_ = std::static_pointer_cast<Problem>(pointer);

        // Create wrapper
        data_.reset(new ProblemWrapperData<Problem>(prob_.get()));
    }

    void Solve(Eigen::MatrixXd &solution) override
    {
        Timer timer;
        planning_time_ = -1;
        prob_->PreUpdate();

        if (!prob_)
            ThrowNamed("Solver has not been initialized!");

        // Reset problem wrapper data
        data_->reset();

        // Get start state and set up solution
        Eigen::VectorXd q0 = prob_->ApplyStartState();
        if (prob_->N != q0.rows())
            ThrowNamed("Wrong size q0 size=" << q0.rows()
                                             << ", required size=" << prob_->N);
        solution.resize(1, prob_->N);
        // prob_->ResetCostEvolution(parameters_.iterations + 1);
        // prob_->SetCostEvolution(0, f(q0));

        // TODO: tmp hack
        data_->q = q0;

        // Create optimisation solver
        nlopt_opt opt = nlopt_create(algorithm_, prob_->N);

        // If problem supports bounds, set bounds
        set_bounds(opt);

        // If problem supports constraints, set constraints
        set_constraints(opt);

        // Set minimization objective
        {
            nlopt_result info = nlopt_set_min_objective(opt, &end_pose_problem_objective_func<Problem>, (void *)data_.get());
            if (info != 1) WARNING("Error while setting objective function: " << (int)info << ": " << nlopt_get_errmsg(opt));
        }

        // Set tolerances
        set_tolerances(opt);

        // Create and assign local optimizer, if required
        nlopt_opt local_opt = nullptr;
        if (local_optimizer_ != nlopt_algorithm::NLOPT_NUM_ALGORITHMS)
        {
            local_opt = nlopt_create(local_optimizer_, prob_->N);
            set_bounds(local_opt);
            set_tolerances(local_opt);
            nlopt_result info = nlopt_set_local_optimizer(opt, local_opt);
            if (info != 1) WARNING("Error while setting local optimizer: " << (int)info << ": " << nlopt_get_errmsg(opt));
        }

        // Record the cost value for the initial solution
        double initial_cost_value = end_pose_problem_objective_func<Problem>(prob_->N, q0.data(), nullptr, (void *)data_.get());

        // Scale the initial step size (only for derivative-free algorithms)
        constexpr double initial_step_scale = 1.0;
        Eigen::VectorXd initial_step = Eigen::VectorXd::Zero(prob_->N);
        nlopt_get_initial_step(opt, q0.data(), initial_step.data());
        initial_step *= initial_step_scale;
        nlopt_set_initial_step(opt, initial_step.data());

        Eigen::VectorXd lb(prob_->N), ub(prob_->N);
        nlopt_get_lower_bounds(opt, lb.data());
        nlopt_get_upper_bounds(opt, ub.data());
        q0 = q0.cwiseMin(ub).cwiseMax(lb);  // Project to feasible bounds

        // Run the optimization
        double final_cost_value;
        nlopt_result info = nlopt_optimize(opt, q0.data(), &final_cost_value);
        if (info < 0)
        {
            auto msg = nlopt_get_errmsg(opt);
            std::string err_msg{};
            if (msg != nullptr) err_msg = std::string(msg);
            WARNING("Optimization did not exit cleanly (code: " << (int)info << ")! " << get_result_info(info) << ": " << err_msg);
            // Do not throw an exception since e.g. in iterative IK the result tends to go away on the following iteration
        }

        solution.row(0) = q0;
        planning_time_ = timer.GetDuration();

        // Show statistics if "verbose" is set as true
        if (debug_)
        {
            HIGHLIGHT_NAMED("NLoptGenericEndPoseSolver", "Info: " << get_result_info(info));
            std::cout << "------ nlopt ------" << std::endl;
            std::cout << "Dimensions     : " << prob_->N << std::endl;
            std::cout << "  f(obj/eq/neq): " << data_->objective_function_evaluations << " / " << data_->equality_function_evaluations << " / " << data_->inequality_function_evaluations << std::endl;
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
    std::shared_ptr<ProblemWrapperData<Problem>> data_ = nullptr;

    nlopt_algorithm algorithm_ = nlopt_algorithm::NLOPT_NUM_ALGORITHMS;        ///< Selected optimization algorithm.
    nlopt_algorithm local_optimizer_ = nlopt_algorithm::NLOPT_NUM_ALGORITHMS;  ///< Local optimization, if required by the selected algorithm.

    virtual void set_bounds(nlopt_opt /*my_opt*/) {}       // To be reimplemented in bounded problems
    virtual void set_constraints(nlopt_opt /*my_opt*/) {}  // To be reimplemented in constrained problems
    void set_tolerances(nlopt_opt my_opt)
    {
        nlopt_set_maxeval(my_opt, this->parameters_.MaxFunctionEvaluations);
        nlopt_set_ftol_rel(my_opt, this->parameters_.RelativeFunctionTolerance);
        nlopt_set_xtol_rel(my_opt, this->parameters_.RelativeVariableTolerance);
        nlopt_set_ftol_abs(my_opt, this->parameters_.AbsoluteFunctionTolerance);
    }
};

// TODO: This could likely be made much neater without copy-paste with some
// combination of std::enable_if and std::is_same.
template <>
void NLoptGenericEndPoseSolver<BoundedEndPoseProblem, NLoptBoundedEndPoseSolverInitializer>::set_bounds(nlopt_opt my_opt)
{
    // Creating a copy since we are modifying it below - TODO: Do not create copies
    Eigen::VectorXd lower_bounds = prob_->GetBounds().col(0);
    Eigen::VectorXd upper_bounds = prob_->GetBounds().col(1);

    // Check if we need to modify the bounds based on velocity limits
    if (this->parameters_.BoundVelocities)
    {
        const Eigen::VectorXd &jvl = prob_->GetScene()->GetKinematicTree().GetVelocityLimits();
        const Eigen::VectorXd incremental_motion = this->parameters_.dt * jvl;

        // Update the bounds based on the allowed incremental motion given the velocity limits and the timestep
        lower_bounds = lower_bounds.cwiseMax(data_->q - incremental_motion);
        upper_bounds = upper_bounds.cwiseMin(data_->q + incremental_motion);
    }

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
    // Creating a copy since we are modifying it below - TODO: Do not create copies
    Eigen::VectorXd lower_bounds = prob_->GetBounds().col(0);
    Eigen::VectorXd upper_bounds = prob_->GetBounds().col(1);

    // Check if we need to modify the bounds based on velocity limits
    if (this->parameters_.BoundVelocities)
    {
        const Eigen::VectorXd &jvl = prob_->GetScene()->GetKinematicTree().GetVelocityLimits();
        const Eigen::VectorXd incremental_motion = this->parameters_.dt * jvl;

        // Update the bounds based on the allowed incremental motion given the velocity limits and the timestep
        lower_bounds = lower_bounds.cwiseMax(data_->q - incremental_motion);
        upper_bounds = upper_bounds.cwiseMin(data_->q + incremental_motion);
    }

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
        if (!my_opt) ThrowPretty("opt is dead");
        const unsigned int &m_neq = prob_->inequality.length_Phi;
        if (m_neq > 0)
        {
            const Eigen::VectorXd tol_neq = 1e-6 * Eigen::VectorXd::Ones(m_neq);  // prob_->parameters.InequalityFeasibilityTolerance * Eigen::VectorXd::Ones(m_neq);
            nlopt_result info = nlopt_add_inequality_mconstraint(my_opt, m_neq, &end_pose_problem_inequality_constraint_mfunc<EndPoseProblem>, (void *)data_.get(), tol_neq.data());
            if (info != 1) WARNING("Error while setting inequality constraints: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
        }
    }

    {
        const unsigned int &m_eq = prob_->equality.length_Phi;
        if (m_eq > 0)
        {
            const Eigen::VectorXd tol_eq = 1e-6 * Eigen::VectorXd::Ones(m_eq);  // prob_->parameters.EqualityFeasibilityTolerance * Eigen::VectorXd::Ones(m_eq);
            nlopt_result info = nlopt_add_equality_mconstraint(my_opt, m_eq, &end_pose_problem_equality_constraint_mfunc<EndPoseProblem>, (void *)data_.get(), tol_eq.data());
            if (info != 1) WARNING("Error while setting equality constraints: " << (int)info << ": " << nlopt_get_errmsg(my_opt));
        }
    }
}

typedef NLoptGenericEndPoseSolver<BoundedEndPoseProblem, NLoptBoundedEndPoseSolverInitializer> NLoptBoundedEndPoseSolver;
typedef NLoptGenericEndPoseSolver<EndPoseProblem, NLoptEndPoseSolverInitializer> NLoptEndPoseSolver;
typedef NLoptGenericEndPoseSolver<UnconstrainedEndPoseProblem, NLoptUnconstrainedEndPoseSolverInitializer> NLoptUnconstrainedEndPoseSolver;
}  // namespace exotica

#endif  // EXOTICA_NLOPT_SOLVER_NLOPT_SOLVER_H_
