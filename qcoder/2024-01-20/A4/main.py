from qiskit import QuantumCircuit
 
 
def solve() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    # Write your code here:

    qc.h(0)
    qc.cx(0, 1)
    qc.ch(0, 1)
    qc.x(0)

    return qc


if __name__ == "__main__":
    c = solve()
    print(c)