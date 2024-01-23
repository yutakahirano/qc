from qiskit import QuantumCircuit


def solve() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    # Write your code here:

    qc.h(0)
    qc.cx(0, 1)

    return qc