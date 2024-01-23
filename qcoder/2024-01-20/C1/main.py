from qiskit import QuantumCircuit
import math 


def solve(n: int, L: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qr = qc.qregs[0]
    # Write your code here:

    if n == 1:
        if L == 1:
            pass
        else:
            assert L == 2
            qc.h(0)
        return qc.decompose()


    mid = 2 ** (n - 1)
    if L <= mid:
        c = solve(n - 1, L)
        qc.x(n - 1)
        qc.append(c.control(1), [qr[-1]] + qr[:-1])
        qc.x(n - 1)
        return qc.decompose()
    
    theta = 2 * math.acos(math.sqrt(mid / L))
    qc.ry(theta, n - 1)
    qc.x(n - 1)
    for i in range(n - 1):
        qc.ch(n - 1, i)
    qc.x(n - 1)

    c = solve(n - 1, L - mid)
    qc.append(c.control(1), [qr[-1]] + qr[:-1])
    return qc.decompose()


import qiskit_aer

def run(circuit):
    sim = qiskit_aer.AerSimulator()
    c = circuit.copy()
    c.save_statevector()
    return sim.run(c, shots=1).result()

if __name__ == "__main__":
    n = 7
    L = 5
    c = solve(n, L)
    print('depth = {}'.format(c.depth()))
#    print(c)
#    r = run(c)
#    print(r)
