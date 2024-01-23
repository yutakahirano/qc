from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
 
def solve(n: int, L: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # Write your code here:

    if n == 1:
        if L == 1:
            qc.z(0)
        else:
            assert L == 2
        return qc

    qr = qc.qregs[0]
    for i in range(L):
        gate = ZGate()
        for j in range(n):
            if i & (1 << j) == 0:
                qc.x(qr[j])
        qc.append(ZGate().control(n - 1), qr)
        for j in range(n):
            if i & (1 << j) == 0:
                qc.x(qr[j])

    return qc

if __name__ == "__main__":
    c = solve(1, 1)
    print(c)