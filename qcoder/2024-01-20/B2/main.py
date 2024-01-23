from qiskit import QuantumCircuit, QuantumRegister
 
 
def solve(n: int) -> QuantumCircuit:
    x, y = QuantumRegister(n), QuantumRegister(1)
    qc = QuantumCircuit(x, y)
    # Write your code here:

    for i in range(n):
        qc.cx(x[i], y)
 
    return qc

if __name__ == "__main__":
    c = solve(4)
    print(c)