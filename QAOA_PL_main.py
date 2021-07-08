######
# A similar code can be found in more detail on:
# https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html
######
import math

import pandas as pd
import pennylane as qml
from matplotlib import pyplot as plt
from pennylane import qaoa
from qiskit_optimization.problems import QuadraticProgram


def use_simulator(wires):
    return qml.device('qiskit.aer', wires=wires, backend='aer_simulator_statevector')


def create_cost_hamiltonian():
    coeffs_cost = [1.5, -5.0, 9.0, -1.0, -3.5, 0.5, 9.0]
    obs_cost = [qml.PauliZ(0),
                qml.PauliZ(1),
                qml.PauliZ(0) @ qml.PauliZ(1),
                qml.PauliZ(2),
                qml.PauliZ(1) @ qml.PauliZ(2),
                qml.PauliZ(3),
                qml.PauliZ(2) @ qml.PauliZ(3)]
    cost_h = qml.Hamiltonian(coeffs_cost, obs_cost)

    coeffs_mix = [1 for _ in range(4)]
    obs_mix = [qml.PauliX(i) for i in range(4)]
    mix_h = qml.Hamiltonian(coeffs_mix, obs_mix)

    return cost_h, mix_h


def create_cost_hamiltonian_2(operator, wires):
    cost_matrix = operator.to_matrix_op().to_matrix()
    cost_obs = qml.Hermitian(cost_matrix, wires=wires)
    cost_h = qml.Hamiltonian((1, ), (cost_obs, ))

    coeffs_mix = [1 for _ in range(len(wires))]
    obs_mix = [qml.PauliX(i) for i in range(len(wires))]
    mix_h = qml.Hamiltonian(coeffs_mix, obs_mix)

    return cost_h, mix_h


def qaoa_layer(gamma, alpha):
    qaoa.cost_layer(gamma, cost_h)
    qaoa.mixer_layer(alpha, mixer_h)


def create_problem_matrix(problem_tuple):
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


def create_problem_operator(linear_terms, quadratic_terms):
    # create a QUBO
    qubo = QuadraticProgram()
    for i in range(len(linear_terms)):
        qubo.binary_var('p' + str(i + 1))

    print(qubo)
    qubo.minimize(linear=linear_terms, quadratic=quadratic_terms)

    qubit_op, offset = qubo.to_ising()
    return qubit_op, qubo


def create_hamiltonian(problem, wires):
    linear, quadratic = create_problem_matrix(problem)
    qubit_op, qubo = create_problem_operator(linear, quadratic)
    cost_h, mix_h = create_cost_hamiltonian_2(qubit_op, wires)
    return cost_h, mix_h


def plot_costs(costs):
    costs[1] = [cost / -40 for cost in costs[1]]
    plt.plot(costs[0], costs[1])
    plt.xlabel('Optimizer step')
    plt.ylabel('costs')
    plt.show()


def create_binary_axis(wires):
    nr_of_binaries = 2 ** len(wires)
    binaries = []
    for i in range(nr_of_binaries):
        binaries.append(format(i, f'0{len(wires)}b'))

    return binaries


def plot_bar_probabilities(params):
    probs = probability_circuit(params[0], params[1])
    plt.style.use("seaborn")
    binary_axis = create_binary_axis(wires)
    plt.bar(binary_axis, probs)
    plt.title('Params')
    plt.show()


if __name__ == '__main__':
    # (nr_of_queries, cost_vector, savings_dict)
    problem_4qb = (2, [3, 13, 21, 1], {(1, 2): -14})

    solutions_df = pd.DataFrame(columns=['p', 'r', 'params'])

    wires = range(len(problem_4qb[1]))
    max_depth = 10

    #cost_h, mixer_h = create_hamiltonian(problem_4qb, wires)
    cost_h, mixer_h = create_cost_hamiltonian()
    print(cost_h)
    print(mixer_h)

    dev = qml.device('qulacs.simulator', wires=wires) #use_simulator(wires)

    for depth in range(1, max_depth):

        next_row = {'p':depth}

        def circuit(params, **kwargs):
            for w in wires:
                qml.Hadamard(wires=w)
            qml.layer(qaoa_layer, depth, params[0], params[1])

        cost_function = qml.ExpvalCost(circuit, cost_h, dev)
        optimizer = qml.GradientDescentOptimizer()
        steps = 30
        params = [[0.5] * depth] * 2

        costs = [[], []]
        exp_val = float("inf")
        opt_params = None
        opt_i = 0
        for i in range(steps):
            costs[0].append(i)
            costs[1].append(cost_function(params))
            params = optimizer.step(cost_function, params)
            new_costs = cost_function(params)
            if new_costs < exp_val:
                exp_val = new_costs
                opt_params = params
                opt_i = i

        #plot_costs(costs)
        print("Optimal Parameters")
        print(params)

        @qml.qnode(dev)
        def probability_circuit(gamma, alpha):
            circuit([gamma, alpha])
            return qml.probs(wires=wires)

        next_row['r'] = exp_val / (-40)
        next_row['params'] = params

        solutions_df = solutions_df.append(next_row, ignore_index=True)
        #plot_bar_probabilities(params)
        #plot_bar_probabilities(opt_params)
        print(f"Steps until best solution: {opt_i}")

    solutions_df.to_csv(r'penny_lane.csv', index=False, header=True)