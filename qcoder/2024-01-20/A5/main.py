from qiskit import QuantumCircuit
import math 
 
def solve() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    # Write your code here:

    theta = 2 * math.acos(1 / math.sqrt(3))
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ch(0, 1)
    qc.z(1)
    qc.x(0)

    return qc

import qiskit_aer

def run(circuit):
    sim = qiskit_aer.AerSimulator()
    c = circuit.copy()
    c.save_statevector()
    return sim.run(c, shots=1).result()

if __name__ == "__main__":
    c = solve()
    print(c)
    r = run(c)
    print(r)
