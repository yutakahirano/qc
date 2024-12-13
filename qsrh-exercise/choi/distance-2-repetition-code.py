import qulacs

from math import asin, pi, exp, cos, sin, sqrt
from qulacs import gate
from qulacs.state import drop_qubit, inner_product, tensor_product


def fidelity(state1: qulacs.QuantumState, state2: qulacs.QuantumState) -> float:
    return abs(inner_product(state1, state2)) ** 2


def main() -> None:
    theta = pi / 8

    phi = 2 * asin((sin(theta / 2) ** 2) / sqrt(cos(theta / 2) ** 4 + sin(theta / 2) ** 4))
    print('theta = {}, phi = {}'.format(theta, phi))

    # L_X = X_0X_1, L_Z = Z_0 (= Z_1), L_Y = Y_0X_1.

    # Construct a logical |0> state.
    initial = qulacs.QuantumState(2)

    # Attach one qubit for measurement.
    actual = tensor_product(qulacs.QuantumState(1), initial)
    gate.RX(0, theta).update_quantum_state(actual)
    gate.RX(1, theta).update_quantum_state(actual)

    gate.CNOT(0, 2).update_quantum_state(actual)
    gate.CNOT(1, 2).update_quantum_state(actual)
    gate.P0(2).update_quantum_state(actual)

    norm = sqrt(actual.get_squared_norm())
    print('norm = {}'.format(norm))
    actual.multiply_coef(1 / norm)
    actual = drop_qubit(actual, [2], [0])

    expected = qulacs.QuantumState(2)
    # L_Y = Y_0Z_1.
    gate.PauliRotation([0, 1], [2, 1], phi).update_quantum_state(expected)

    assert expected.get_qubit_count() == actual.get_qubit_count()

    print('fidelity(expected, actual) = {}'.format(fidelity(expected, actual)))
    print('index: (expected[index]) vs (actual[index]))')
    for i in range(1 << expected.get_qubit_count()):
        ce = expected.get_vector()[i]
        ca = actual.get_vector()[i]
        print('{:02}: ({:.3f} + {:.3f}j) vs ({:.3f} + {:.3f}j)'.format(i, ce.real, ce.imag, ca.real, ca.imag))


if __name__ == '__main__':
    main()
