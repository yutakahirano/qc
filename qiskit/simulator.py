from typing import Iterable, List
import logging
import random

import qiskit


class ErrorSet:
    def __init__(self, set: Iterable[int] = set()):
        self._set = 0
        for e in set:
            self._set ^= (1 << e)

    def get(self, index: int):
        return (self._set & (1 << index)) != 0

    def add(self, index: int):
        self._set = self._set ^ (1 << index)

    def __add__(self, other):
        result = ErrorSet()
        result._set = self._set ^ other._set
        return result

    def __eq__(self, other):
        return self._set == other._set

    def __repr__(self):
        result = '{'
        for e in self:
            if result != '{':
                result += ', '
            result += '{}'.format(e)
        return result + '}'

    def __len__(self):
        length = 0
        for _ in self:
            length += 1
        return length

    def __iter__(self):
        current = 0
        set = self._set
        while set > 0:
            if set % 2 != 0:
                yield current
            current += 1
            set = set // 2


class ErrorDistribution:
    def __init__(self, p1, p2, p_measurement, p_preparation):
        self.p1 = p1
        self.p2 = p2
        self.p_measurement = p_measurement
        self.p_preparation = p_preparation

    def has_p1_error(self):
        return random.uniform(0, 1) < self.p1

    def has_p2_error(self):
        return random.uniform(0, 1) < self.p2

    def has_measurement_error(self):
        return random.uniform(0, 1) < self.p_measurement

    def has_preparation_error(self):
        return random.uniform(0, 1) < self.p_preparation


def guess_errors(errors: ErrorSet):
    # m1 is set to true when the corresponding generator detects an error.
    m1 = errors.get(3) ^ errors.get(4) ^ errors.get(5) ^ errors.get(6)
    # m2 is set to true when the corresponding generator detects an error.
    m2 = errors.get(1) ^ errors.get(2) ^ errors.get(5) ^ errors.get(6)
    # m3 is set to true when the corresponding generator detects an error.
    m3 = errors.get(0) ^ errors.get(2) ^ errors.get(4) ^ errors.get(6)

    logging.info('m1 = {}, m2 = {}, m3 = {}'.format(m1, m2, m3))

    if not m1 and not m2 and not m3:
        # No errors
        return ErrorSet()
    index = -1
    if m1:
        index += 4
    if m2:
        index += 2
    if m3:
        index += 1
    return ErrorSet({index})


def is_logically_trivial(errors: ErrorSet):
    if errors == ErrorSet():
        return True

    g1 = ErrorSet({3, 4, 5, 6})
    g2 = ErrorSet({1, 2, 5, 6})
    g3 = ErrorSet({0, 2, 4, 6})

    if errors == g1 or errors == g2 or errors == g3:
        return True

    g12 = g1 + g2
    g13 = g1 + g3
    g23 = g2 + g3

    if errors == g12 or errors == g13 or errors == g23:
        return True

    g123 = g1 + g2 + g3

    if errors == g123:
        return True

    return False


code_size = 7


def inject_p1_errors(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution):
    for j in range(code_size):
        if distribution.has_p1_error():
            # X errors
            x_errors.add(j)
        if distribution.has_p1_error():
            # Y errors
            x_errors.add(j)
            z_errors.add(j)
        if distribution.has_p1_error():
            # Z errors
            z_errors.add(j)


def inject_p2_errors(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        control: int,
        target: int,
        distribution: ErrorDistribution):
    for j in range(code_size):
        if distribution.has_p2_error():
            # IX errors
            x_errors[target].add(j)
        if distribution.has_p2_error():
            # IY errors
            x_errors[target].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # IZ errors
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # XI errors
            x_errors[control].add(j)
        if distribution.has_p2_error():
            # XX errors
            x_errors[control].add(j)
            x_errors[target].add(j)
        if distribution.has_p2_error():
            # XY errors
            x_errors[control].add(j)
            x_errors[target].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # XZ errors
            x_errors[control].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # YI errors
            x_errors[control].add(j)
            z_errors[control].add(j)
        if distribution.has_p2_error():
            # YX errors
            x_errors[control].add(j)
            z_errors[control].add(j)
            x_errors[target].add(j)
        if distribution.has_p2_error():
            # YY errors
            x_errors[control].add(j)
            z_errors[control].add(j)
            x_errors[target].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # YZ errors
            z_errors[control].add(j)
            z_errors[control].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # ZI errors
            z_errors[control].add(j)
        if distribution.has_p2_error():
            # ZX errors
            z_errors[control].add(j)
            x_errors[target].add(j)
        if distribution.has_p2_error():
            # ZY errors
            z_errors[control].add(j)
            x_errors[target].add(j)
            z_errors[target].add(j)
        if distribution.has_p2_error():
            # ZZ errors
            z_errors[control].add(j)
            z_errors[target].add(j)


def inject_syndrome_measurement_errors(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution):
    # g1 and g4
    for _ in range(2):
        for j in [3, 4, 5, 6]:
            if distribution.has_measurement_error():
                # X errors
                x_errors.add(j)
            if distribution.has_measurement_error():
                # Y errors
                x_errors.add(j)
                z_errors.add(j)
            if distribution.has_measurement_error():
                # Z errors
                z_errors.add(j)
    # g2 and g5
    for _ in range(2):
        for j in [1, 2, 5, 6]:
            if distribution.has_measurement_error():
                # X errors
                x_errors.add(j)
            if distribution.has_measurement_error():
                # Y errors
                x_errors.add(j)
                z_errors.add(j)
            if distribution.has_measurement_error():
                # Z errors
                z_errors.add(j)

    # g3 and g6
    for _ in range(2):
        for j in [0, 2, 4, 6]:
            if distribution.has_measurement_error():
                # X errors
                x_errors.add(j)
            if distribution.has_measurement_error():
                # Y errors
                x_errors.add(j)
                z_errors.add(j)
            if distribution.has_measurement_error():
                # Z errors
                z_errors.add(j)


def inject_errors_on_error_correction(
        correction: ErrorSet,
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution):
    for e in correction:
        if distribution.has_measurement_error():
            # X errors
            x_errors.add(e)
        if distribution.has_measurement_error():
            # Y errors
            x_errors.add(e)
            z_errors.add(e)
        if distribution.has_measurement_error():
            # Z errors
            z_errors.add(e)


def run_error_correction(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        circuit_with_error: qiskit.QuantumCircuit,
        index: int,
        distribution: ErrorDistribution):
    # Inject errors generated by error syndrome measurements.
    inject_syndrome_measurement_errors(x_errors, z_errors, distribution)

    # Try to correct errors.
    x_correction = guess_errors(x_errors)
    z_correction = guess_errors(z_errors)
    x_errors += x_correction
    z_errors += z_correction

    inject_errors_on_error_correction(
        x_correction, x_errors, z_errors, distribution)
    inject_errors_on_error_correction(
        z_correction, x_errors, z_errors, distribution)

    if not is_logically_trivial(x_errors):
        # There is no physical operation corresponding to this, so we
        # don't need to think about errors.
        x_errors += ErrorSet(range(code_size))
        circuit_with_error.x(index)

    if not is_logically_trivial(z_errors):
        # There is no physical operation corresponding to this, so we
        # don't need to think about errors.
        z_errors += ErrorSet(range(code_size))
        circuit_with_error.z(index)


def simulate(circuit: qiskit.QuantumCircuit, distribution: ErrorDistribution):
    circuit_with_error = qiskit.QuantumCircuit(circuit.num_qubits)
    x_errors = [ErrorSet() for _ in range(circuit.num_qubits)]
    z_errors = [ErrorSet() for _ in range(circuit.num_qubits)]

    # TODO: State preparation can inject errors.

    for gate in circuit.data:
        logging.info('operation name = {}'.format(gate.operation.name))
        if gate.operation.name in {'h', 's', 'sdg'}:
            # `gate` is a one-qubit clifford gate
            index = circuit.qubits.index(gate.qubits[0])

            if gate.operation.name == 'h':
                # Run the logical operation.
                circuit_with_error.h(index)

                # Update the error data.
                # We swap X errors and Z errors, given HX = ZH and HZ = XH.
                (x_errors[index], z_errors[index]) = \
                    (z_errors[index], x_errors[index])
            elif gate.operation.name == 's':
                # Run the logical operation.
                circuit_with_error.s(index)

                # Update the error data.
                # ZS acts on each qubit, and ZSX = -YZS and (ZS)Z = Z(ZS).
                z_errors[index] += x_errors[index]
            elif gate.operation.name == 'sdg':
                # Run the logical operation.
                circuit_with_error.sdg(index)

                # Update the error data.
                # This action is the inverse of the action for the S gate.
                z_errors[index] += x_errors[index]
            else:
                assert False, 'UNREACHABLE'

            # Inject errors generated by the logical operation.
            inject_p1_errors(x_errors[index], z_errors[index], distribution)

            run_error_correction(
                x_errors[index], z_errors[index],
                circuit_with_error, index, distribution)
        elif gate.operation.name == 'cx':
            # `gate` is a CNOT gate.
            control = circuit.qubits.index(gate.qubits[0])
            target = circuit.qubits.index(gate.qubits[1])

            # Run the logical operation.
            circuit_with_error.cnot(control, target)

            # Update the error data.
            # CX * X_CONTROL  = X_CONTROL * X_TARGET * CX
            # CX * X_TARGET   = X_TARGET * CX
            # CX * Z_CONTROL  = Z_CONTROL * CX
            # CX * Z_TARGET   = Z_CONTROL * Z_TARGET * CX
            x_errors[target] += x_errors[control]
            z_errors[control] += z_errors[target]

            # Inject errors generated by the logical operation.
            inject_p2_errors(x_errors, z_errors, control, target, distribution)

            run_error_correction(
                x_errors[control], z_errors[control],
                circuit_with_error, control, distribution)
            run_error_correction(
                x_errors[target], z_errors[target],
                circuit_with_error, target, distribution)
        else:
            raise RuntimeError(
                'Unsupported gate: {}'.format(gate.operation.name))
    return circuit_with_error
