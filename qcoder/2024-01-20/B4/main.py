from qiskit import QuantumCircuit, QuantumRegister
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

    if L * 2 <= 2 ** n:
        smaller_circuit = solve(n - 1, L).control(1).decompose()

        qc.x(n - 1)
        qc.append(smaller_circuit, [qc.qubits[n - 1]] + qc.qubits[0:n-1])
        qc.x(n - 1)
    else:
        qc.x(n - 1)
        qc.z(n - 1)
        qc.x(n - 1)
        smaller_circuit = solve(n - 1, L - 2 ** (n - 1)).control(1).decompose()
        qc.append(smaller_circuit, [qc.qubits[n - 1]] + qc.qubits[0:n-1])

    return qc.decompose()


import qiskit_aer

def run(circuit):
    sim = qiskit_aer.AerSimulator()
    c = circuit.copy()
    c.save_statevector()
    return sim.run(c, shots=1).result()

if __name__ == "__main__":
    for n in range(1, 6):
        for L in range(1 ** n, 2 ** n + 1):
            # c = solve(n, L)
            # assert c.depth() <= 50
            pass

    c = solve(2, 1)
    n = 2
    cc = QuantumCircuit(n)
    for i in range(n):
        cc.h(i)
    cc.append(c, cc.qubits)
    c = cc.decompose()
    print(c)
    print('depth = {}'.format(c.depth()))
    result = run(c)
    v = result.get_statevector()
    print('result:')
    for i in v:
        print('{:.2f}+{:.2f}i'.format(i.real+1e-10, i.imag+1e-10))
    print('=====')
