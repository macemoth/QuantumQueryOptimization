import math
import random

import numpy as np
from qiskit import IBMQ, Aer, execute, ClassicalRegister
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA, ADAM, NELDER_MEAD, NFT, SLSQP, TNC, POWELL
from qiskit.aqua import aqua_globals
from qiskit.optimization.applications.ising.common import sample_most_likely
from qiskit.providers.ibmq import least_busy
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import QuadraticProgram


# What device to use #############################################################
def use_local_simulator():
    aqua_globals.random_seed = 69069
    seed = 29
    backend = Aer.get_backend('qasm_simulator')
    return QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed), backend


def use_online_qasm_simulator():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = provider.get_backend('ibmq_qasm_simulator')
    return QuantumInstance(backend, skip_qobj_validation=False), backend


def use_least_busy_real_device():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q')
    backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= 4 and
                                                             not x.configuration().simulator and x.status().operational == True))
    print("Running on current least busy device: ", backend)
    return QuantumInstance(backend, skip_qobj_validation=False), backend


####################################################################################

# Save data created by the optimizer during QAOA runs ##############################
class Steps:
    def __init__(self):
        self.steps = []

    def next_step(self, eval_count, params, eval_mean, eval_sd):
        self.steps.append([eval_count, params, eval_mean, eval_sd])


def callback(eval_count, params, eval_mean, eval_sd):
    saved_data.next_step(eval_count, params, eval_mean, eval_sd)


####################################################################################

# Different Optimizers to choose from ##############################################
def build_optimizers(maxiter):
    optimizers = [ADAM(maxiter=maxiter),
                  COBYLA(maxiter=maxiter),
                  NELDER_MEAD(maxiter=maxiter),
                  NFT(maxiter=maxiter),
                  POWELL(maxiter=maxiter),
                  SLSQP(maxiter=maxiter),
                  TNC(maxiter=maxiter)]
    return optimizers


####################################################################################

# Create the QAOA instance with qiskit #############################################
def create_qaoa(instance, p, maxiter=1000, params=None, algo_nr=1):
    optimizers = build_optimizers(maxiter)
    qaoa_instance = QAOA(optimizer=optimizers[algo_nr], initial_point=params,
                         reps=p, quantum_instance=instance,
                         callback=callback)
    return qaoa_instance


####################################################################################


# construct the circuit corresponding to the current QAOA instance and parameters ##
def construct_circuit(qaoa_results, operator, nr_of_qb):
    q_circuit = qaoa_results.construct_circuit(qaoa_results.optimal_params, operator)
    q_circuit = q_circuit[0]
    cr = ClassicalRegister(nr_of_qb, 'c')
    q_circuit.add_register(cr)
    q_circuit.measure(range(nr_of_qb), range(nr_of_qb))
    return q_circuit


####################################################################################


# run the specified quantum circuit ################################################
def run_circuit(qc):
    shots = 4096
    job = execute(qc, backend, shots=shots)
    result = job.result()
    return result.get_counts()


####################################################################################

# map the initial MQO-Problem to a problem matrix for use in QuadraticProgram ######
def create_problem_matrix_and_dict(problem_tuple):
    nr_of_queries = problem_tuple[0]
    plan_costs = problem_tuple[1]
    savings = problem_tuple[2]

    nr_of_plans = len(plan_costs)
    nr_of_plans_each = nr_of_plans / nr_of_queries

    eps = 1
    w_min = max(plan_costs) + eps
    w_max = w_min
    if savings:
        sum_savings = sum(savings.values())
        w_max = w_min - sum_savings

    linear_terms = []
    quadratic_terms = {}

    for i in range(nr_of_plans):
        for j in range(i, nr_of_plans):
            query_i = math.floor(i / nr_of_plans_each)
            query_j = math.floor(j / nr_of_plans_each)
            plan_1 = 'p' + str(i + 1)
            plan_2 = 'p' + str(j + 1)
            if i == j:
                linear_terms.append(plan_costs[i] - w_min)
            elif query_i == query_j:
                quadratic_terms[plan_1, plan_2] = w_max
            else:
                tuple_forward = (i, j)
                tuple_backward = (j, i)
                if tuple_forward in savings:
                    quadratic_terms[plan_1, plan_2] = savings[tuple_forward]
                elif tuple_backward in savings:
                    quadratic_terms[plan_1, plan_2] = savings[tuple_backward]

    return linear_terms, quadratic_terms


#####################################################################################


# create the QuadraticProgram (QUBO) from the problem matrix ########################
def create_problem_operator(linear_terms, quadratic_terms):
    # create a QUBO
    qubo = QuadraticProgram()
    for i in range(len(linear_terms)):
        qubo.binary_var('p' + str(i + 1))

    qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

    qubit_op, offset = qubo.to_ising()
    return qubit_op, qubo


#####################################################################################


# class to hold all generated data ##################################################
class Solutions:
    def __init__(self):
        self.problems = {}

    def next_solution(self, index, problem, solution_state, classical_solution_state, opt_params, expectation_value,
                      q_costs, classical_costs,
                      counts, saved_data):
        self.problems[index] = {"Problem": problem,
                                "Solution_state": solution_state,
                                "Classical_solution_state": classical_solution_state,
                                "Opt_params": opt_params,
                                "Expectation_value": expectation_value,
                                "QCosts": q_costs,
                                "CCosts": classical_costs,
                                "Counts": counts,
                                "Saved_data": saved_data}


######################################################################################


# create random savings for random problem generation ################################
def create_savings(nr_of_queries, nr_of_plans):
    savings = {}
    nr_of_plans_each = int(nr_of_plans / nr_of_queries)
    for j in range(nr_of_plans):
        current_query = math.floor(j / nr_of_plans_each)
        first_plan_next_query = (current_query + 1) * nr_of_plans_each
        for i in range(first_plan_next_query, nr_of_plans):
            savings[j, i] = random.randint(-20, 0)

    return savings


#######################################################################################


# print the problem ###################################################################
def print_problem(problem):
    print("Costs:")
    print(problem[1])
    print("Savings:")
    print(problem[2])


#######################################################################################

if __name__ == '__main__':

    # Which instance to use -------------------------
    quantum_instance, backend = use_local_simulator()
    # quantum_instance, backend = use_online_qasm_simulator()
    # quantum_instance, backend = use_least_busy_real_device()
    # -----------------------------------------------

    # Create Problems -------------------------------
    # (nr_of_queries, cost_vector, savings_dict)
    nr_of_problems = 1
    problems = []

    nr_of_qbs = 4
    nr_of_queries = 2

    for i in range(nr_of_problems):
        problems.append((nr_of_queries, np.random.randint(0, 50, nr_of_qbs), create_savings(nr_of_queries, nr_of_qbs)))
    # -----------------------------------------------

    solutions = Solutions()

    repetitions_of_F = 5
    algo_nr = 0
    optimizers_names = ["COBYLA", "ADAM", "NELDER_MEAD", "NFT", "POWELL", "SLSQP", "TNC"]

    start_params = [np.random.uniform(low=-math.pi, high=math.pi),
                    np.random.uniform(low=-math.pi, high=math.pi)] * repetitions_of_F

    for index, problem in enumerate(problems):
        print(f"next problem running {index}")
        nr_of_qubits = len(problem[1])
        saved_data = Steps()

        # create QUBO-Operator from problem ----------------------
        linear, quadratic = create_problem_matrix_and_dict(problem)
        problem_operator, qubo = create_problem_operator(linear, quadratic)
        # --------------------------------------------------------

        # create QAOA and solve problem --------------------------

        reps = repetitions_of_F  # p

        # params = start_params[nr_of_outputs]
        params = start_params

        qaoa = create_qaoa(quantum_instance, reps, params=params, algo_nr=algo_nr)

        ####### solving the MQO Problem
        result = qaoa.compute_minimum_eigenvalue(problem_operator)
        solution_state = sample_most_likely(result.eigenstate)

        #################

        opt_params = qaoa.optimal_params
        expectation_value = qaoa.get_optimal_cost()
        q_costs = qaoa.get_optimal_cost()
        # --------------------------------------------------------

        # create best optimized circuit from QAOA instance -------
        qc = construct_circuit(qaoa, problem_operator, nr_of_qubits)
        # run circuit
        counts = run_circuit(qc)
        # --------------------------------------------------------

        # calculate with classical eigensolver -------------------
        npme = NumPyMinimumEigensolver()
        exact = MinimumEigenOptimizer(npme)
        classical_result = exact.solve(qubo)
        classical_solution_state = classical_result.x
        classical_costs = classical_result.fval
        # --------------------------------------------------------

        print("\nProblem:")
        print_problem(problem)
        print(f"Quadratic Objective: {qubo.objective.quadratic_program}")
        print(f"reps: {repetitions_of_F}")
        print(f"State with highest probability from statevector (eigenstate): {solution_state}")
        print(f"Parameters which lead to lowest expectation value: {opt_params}")
        print(f"Lowest Expectation value: {expectation_value}")
        print(f"Classical solution: {classical_solution_state}")
        print(f"Classical costs: {classical_costs}")
        # print(qc.draw()) # draws the quantum circuit

        solution_index = (repetitions_of_F, index)
        solutions.next_solution(solution_index, problem, solution_state, classical_solution_state,
                                opt_params, expectation_value, q_costs,
                                classical_costs, counts, saved_data)
