#####
# the code is inspired by:
# https://qiskit.org/documentation/tutorials/optimization/4_grover_optimizer.html
# visit for more details
#####
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.optimization.algorithms import GroverOptimizer, MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram
from qiskit import BasicAer
from docplex.mp.model import Model
from qiskit.providers.aer import Aer


def use_simulator():
    backend = Aer.get_backend('qasm_simulator')
    return QuantumInstance(backend), backend


def create_problem_operator():
    model = Model()
    x0 = model.binary_var(name='x0')
    x1 = model.binary_var(name='x1')
    x2 = model.binary_var(name='x2')
    x3 = model.binary_var(name='x3')
    model.minimize(- 19 * x0 - 9 * x1 - x2 - 21 * x3 + 36 * x0 * x1 - 14 * x1 * x2 + 36 * x2 * x3)
    qp = QuadraticProgram()
    qp.from_docplex(model)
    return qp


if __name__ == '__main__':
    quantum_instance, backend = use_simulator()
    # -----------------------------------------------

    # Create Problems -------------------------------
    # (nr_of_queries, cost_vector, savings_dict)
    # (2, [3, 13, 21, 1], {(2, 3): -14})
    problem_4qb = (2, [3, 13, 21, 1], {(2, 3): -14})

    nr_of_qubits = len(problem_4qb[1])

    qubo = create_problem_operator()

    grover_optimizer = GroverOptimizer(nr_of_qubits, num_iterations=5, quantum_instance=backend)
    results = grover_optimizer.solve(qubo)
    print("x={}".format(results.x))
    print("fval={}".format(results.fval))
