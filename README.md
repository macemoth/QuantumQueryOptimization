# BA-Companion

Tobias Fankhauser, Marc E. Solèr

This repository contains additional material for the BA *"Multiple Query Optimization using a Hybrid Approach of Classical and Quantum Computing"*, supervised by Prof. Dr. Kurt Stockinger and Prof. Dr. Rudolf Füchslin.

## Prerequisites

- Python 3.8
- Anaconda
- Jupyter Notebook
- Qiskit

The Qiskit package can be installed with `pip install qiskit`.

To utilize a IBM Quantum Computer, a IBM Token is required. It can be acquired from [IBM Quantum Computing](https://quantum-computing.ibm.com). Then, the token must be saved once to the computer using

```python
from qiskit import IBMQ
IBMQ.save_account('MY_API_TOKEN')
```

## Programs

- `QAOA_Qiskit_main.py` is the plain-vanilla implementation of QAOA with Qiskit using default optimizers
- `QAOA_FOURIER.py` uses the improved *FOURIER* optimizer by [Zhou et al.](https://arxiv.org/abs/1812.01041) which is considerably better
- `QAOA_PL_main.py` uses [PennyLane](https://pennylane.ai) , a Quantum Machine Learning framework
- `GAS_main.py` is an implementation of [https://arxiv.org/abs/1912.04088](Grover Adaptive Search), an algorithm similar to QAOA

## Notes

- The `ibmq-melbourne` quantum computer is [retired](https://quantum-computing.ibm.com/notifications?type=Service%20Alert) by 7 July 2021. As there is currently no other system with that many qubits, experiments with ~15 qubits cannot be carried out.
