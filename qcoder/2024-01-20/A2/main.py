import qiskit
from qiskit import QuantumCircuit


def solve(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # Write your code here:

    for i in range(n):
        qc.h(i)
    return qc

if __name__ == "__main__":
    c = solve(2)
    print(c)