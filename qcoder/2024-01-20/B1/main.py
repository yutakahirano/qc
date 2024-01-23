from qiskit import QuantumCircuit, QuantumRegister


def solve() -> QuantumCircuit:
    x, y = QuantumRegister(1), QuantumRegister(1)
    qc = QuantumCircuit(x, y)
    # Write your code here:
    qc.cx(x, y)

    return qc


solve()