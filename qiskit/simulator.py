from typing import Iterable, List, Tuple
import collections
import logging
import random

import qiskit
import qulacs
import qulacs.gate


class ErrorSet:
    def __init__(self, set: Iterable[int] = set()):
        self._set = 0
        for e in set:
            self._set ^= (1 << e)

    def get(self, index: int):
        return (self._set & (1 << index)) != 0

    def add(self, index: int):
        self._set = self._set ^ (1 << index)

    def clear(self, index: int):
        if self.get(index):
            self.add(index)

    def __add__(self, other):
        result = ErrorSet()
        result._set = self._set ^ other._set
        return result

    def __iadd__(self, other):
        self._set ^= other._set
        return self

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


STEANE_CODE_SIZE = 7
MEASUREMENT_REPETITION = 3


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


def place_physical_h(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        index: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # We swap the X error and Z error, given HX = ZH and HZ = XH.
    had_x_error = x_errors.get(index)
    had_z_error = z_errors.get(index)
    if had_x_error:
        x_errors.add(index)
        z_errors.add(index)
    if had_z_error:
        z_errors.add(index)
        x_errors.add(index)

    if distribution.has_p1_error():
        # X error
        x_errors.add(index)
    if distribution.has_p1_error():
        # Y error
        x_errors.add(index)
        z_errors.add(index)
    if distribution.has_p1_error():
        # Z error
        z_errors.add(index)


def place_physical_cnot(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        control: int,
        target: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # CX * X_CONTROL  = X_CONTROL * X_TARGET * CX
    # CX * X_TARGET   = X_TARGET * CX
    # CX * Z_CONTROL  = Z_CONTROL * CX
    # CX * Z_TARGET   = Z_CONTROL * Z_TARGET * CX
    if x_errors.get(control):
        x_errors.add(target)
    if z_errors.get(target):
        z_errors.add(control)

    inject_p2_errors_on_pysical_qubit(
        x_errors, z_errors, control, target, distribution)


# Returns (r1, r2) where
#  - `r1` is True when the measurement result is trivial, and
#  - `r2` is True when the flag qubit measurement result is trivial.
def run_x_stabilizer_measurement(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        pattern: List[int],
        distribution: ErrorDistribution,
        with_flag: bool) -> Tuple[bool, bool]:
    assert len(pattern) == 4
    # Create two ancilla qubits.
    a1 = STEANE_CODE_SIZE
    a2 = STEANE_CODE_SIZE + 1

    for j in [a1, a2]:
        if distribution.has_preparation_error():
            # X error
            x_errors.add(j)
        if distribution.has_preparation_error():
            # Y error
            x_errors.add(j)
            z_errors.add(j)
        if distribution.has_preparation_error():
            # Z error
            z_errors.add(j)

    place_physical_h(x_errors, z_errors, a1, distribution)
    place_physical_cnot(x_errors, z_errors, a1, pattern[0], distribution)
    if with_flag:
        place_physical_cnot(x_errors, z_errors, a1, a2, distribution)
    place_physical_cnot(x_errors, z_errors, a1, pattern[1], distribution)
    place_physical_cnot(x_errors, z_errors, a1, pattern[2], distribution)
    if with_flag:
        place_physical_cnot(x_errors, z_errors, a1, a2, distribution)
    place_physical_cnot(x_errors, z_errors, a1, pattern[3], distribution)
    place_physical_h(x_errors, z_errors, a1, distribution)

    r1 = not x_errors.get(a1)
    if distribution.has_measurement_error():
        r1 = not r1
    if with_flag:
        r2 = not x_errors.get(a2)
        if distribution.has_measurement_error():
            r2 = not r2
    else:
        r2 = True

    # Clear errors on the ancilla qubits.
    x_errors.clear(a1)
    x_errors.clear(a2)
    z_errors.clear(a1)
    z_errors.clear(a2)

    return (r1, r2)


# Returns (r1, r2) where
#  - `r1` is True when the measurement result is trivial, and
#  - `r2` is True when the flag qubit measurement result is trivial.
def run_z_stabilizer_measurement(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        pattern: List[int],
        distribution: ErrorDistribution,
        with_flag: bool) -> Tuple[bool, bool]:
    assert len(pattern) == 4
    # Create two ancilla qubits.
    a1 = STEANE_CODE_SIZE
    a2 = STEANE_CODE_SIZE + 1
    for j in [a1, a2]:
        if distribution.has_preparation_error():
            # X error
            x_errors.add(j)
        if distribution.has_preparation_error():
            # Y error
            x_errors.add(j)
            z_errors.add(j)
        if distribution.has_preparation_error():
            # Z error
            z_errors.add(j)

    place_physical_h(x_errors, z_errors, a2, distribution)
    place_physical_cnot(x_errors, z_errors, pattern[0], a1, distribution)
    if with_flag:
        place_physical_cnot(x_errors, z_errors, a2, a1, distribution)
    place_physical_cnot(x_errors, z_errors, pattern[1], a1, distribution)
    place_physical_cnot(x_errors, z_errors, pattern[2], a1, distribution)
    if with_flag:
        place_physical_cnot(x_errors, z_errors, a2, a1, distribution)
    place_physical_cnot(x_errors, z_errors, pattern[3], a1, distribution)
    place_physical_h(x_errors, z_errors, a2, distribution)

    r1 = not x_errors.get(a1)
    if distribution.has_measurement_error():
        r1 = not r1
    if with_flag:
        r2 = not x_errors.get(a2)
        if distribution.has_measurement_error():
            r2 = not r2
    else:
        r2 = True

    # Clear errors on the ancilla qubits.
    x_errors.clear(a1)
    x_errors.clear(a2)
    z_errors.clear(a1)
    z_errors.clear(a2)

    return (r1, r2)


# Guesses qubit errors and returns X and Z correction actions.
# See https://arxiv.org/abs/1705.02329 for the error syndrome measurement.
def guess_errors(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution) -> Tuple[ErrorSet, ErrorSet]:
    saw_non_trivial_syndrome = False
    flag_raised = None

    # g1 = X3X4X5X6
    # g2 = X1X2X5X6
    # g3 = X0X2X4X6
    # g4 = Z3Z4Z5Z6
    # g5 = Z1Z2Z5Z6
    # g6 = Z0Z2Z4Z6
    patterns = [[3, 4, 5, 6], [1, 2, 5, 6], [0, 2, 4, 6]]

    for i, pattern in enumerate(patterns):
        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, pattern, distribution, with_flag=True)
        if not r2:
            flag_raised = ('x', i)
            break
        if not r1:
            saw_non_trivial_syndrome = True
            break

        (r1, r2) = run_z_stabilizer_measurement(
            x_errors, z_errors, pattern, distribution, with_flag=True)

        if not r2:
            flag_raised = ('z', i)
            break
        if not r1:
            saw_non_trivial_syndrome = True
            break

    logging.info('saw_non_trivial_syndrome = {}, flag_raised = {}'.format(
        saw_non_trivial_syndrome, flag_raised))
    if not saw_non_trivial_syndrome and flag_raised is None:
        # No errors are found.
        return (ErrorSet({}), ErrorSet({}))

    # measurement results
    m = []
    for pattern in patterns:
        (r1, _) = run_x_stabilizer_measurement(
            x_errors, z_errors, pattern, distribution, with_flag=False)
        m.append(0 if r1 else 1)
    for pattern in patterns:
        (r1, _) = run_z_stabilizer_measurement(
            x_errors, z_errors, pattern, distribution, with_flag=False)
        m.append(0 if r1 else 1)

    x_index = m[3] * 4 + m[4] * 2 + m[5]
    z_index = m[0] * 4 + m[1] * 2 + m[2]

    guessed_x_errors = ErrorSet()
    guessed_z_errors = ErrorSet()

    if x_index > 0:
        guessed_x_errors.add(x_index - 1)
    if z_index > 0:
        guessed_z_errors.add(z_index - 1)

    if flag_raised == ('x', 0) and x_index == 1:
        guessed_x_errors = ErrorSet({5, 6})
    elif flag_raised == ('x', 1) and x_index == 1:
        guessed_x_errors = ErrorSet({5, 6})
    elif flag_raised == ('x', 2) and x_index == 2:
        guessed_x_errors = ErrorSet({4, 6})
    elif flag_raised == ('z', 0) and z_index == 1:
        guessed_z_errors = ErrorSet({5, 6})
    elif flag_raised == ('z', 1) and z_index == 1:
        guessed_z_errors = ErrorSet({5, 6})
    elif flag_raised == ('z', 2) and z_index == 2:
        guessed_z_errors = ErrorSet({4, 6})

    return (guessed_x_errors, guessed_z_errors)


def calculate_deviation(errors: ErrorSet):
    num_errors = len(errors)
    # Fast path
    if num_errors <= 1:
        return num_errors

    g1 = ErrorSet({3, 4, 5, 6})
    g2 = ErrorSet({1, 2, 5, 6})
    g3 = ErrorSet({0, 2, 4, 6})

    return min(
        num_errors,
        len(g1 + errors), len(g2 + errors), len(g3 + errors),
        len(g1 + g2 + errors), len(g1 + g3 + errors),
        len(g2 + g3 + errors), len(g1 + g2 + g3 + errors))


def inject_p1_errors(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution):
    for j in range(STEANE_CODE_SIZE):
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
    for j in range(STEANE_CODE_SIZE):
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


# Unlike `inject_p2_errors` which acts on logical qubits, this acts on
# physical qubits.
def inject_p2_errors_on_pysical_qubit(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        control: int,
        target: int,
        distribution: ErrorDistribution):
    if distribution.has_p2_error():
        # IX errors
        x_errors.add(target)
    if distribution.has_p2_error():
        # IY errors
        x_errors.add(target)
        z_errors.add(target)
    if distribution.has_p2_error():
        # IZ errors
        z_errors.add(target)
    if distribution.has_p2_error():
        # XI errors
        x_errors.add(control)
    if distribution.has_p2_error():
        # XX errors
        x_errors.add(control)
        x_errors.add(target)
    if distribution.has_p2_error():
        # XY errors
        x_errors.add(control)
        x_errors.add(target)
        z_errors.add(target)
    if distribution.has_p2_error():
        # XZ errors
        x_errors.add(control)
        z_errors.add(target)
    if distribution.has_p2_error():
        # YI errors
        x_errors.add(control)
        z_errors.add(control)
    if distribution.has_p2_error():
        # YX errors
        x_errors.add(control)
        z_errors.add(control)
        x_errors.add(target)
    if distribution.has_p2_error():
        # YY errors
        x_errors.add(control)
        z_errors.add(control)
        x_errors.add(target)
        z_errors.add(target)
    if distribution.has_p2_error():
        # YZ errors
        z_errors.add(control)
        z_errors.add(control)
        z_errors.add(target)
    if distribution.has_p2_error():
        # ZI errors
        z_errors.add(control)
    if distribution.has_p2_error():
        # ZX errors
        z_errors.add(control)
        x_errors.add(target)
    if distribution.has_p2_error():
        # ZY errors
        z_errors.add(control)
        x_errors.add(target)
        z_errors.add(target)
    if distribution.has_p2_error():
        # ZZ errors
        z_errors.add(control)
        z_errors.add(target)


# If logical X errors and Z errors are contained in `x_errors` and `z_errors`,
# then move them to `stae`.
def move_logical_errors_to_state(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        state: qulacs.QuantumState,
        index: int):
    if calculate_deviation(x_errors) >= 2:
        # There is no physical operation corresponding to this, so we
        # don't need to think about errors.
        x_errors += ErrorSet(range(STEANE_CODE_SIZE))
        qulacs.gate.X(index).update_quantum_state(state)
    if calculate_deviation(z_errors) >= 2:
        # There is no physical operation corresponding to this, so we
        # don't need to think about errors.
        z_errors += ErrorSet(range(STEANE_CODE_SIZE))
        qulacs.gate.Z(index).update_quantum_state(state)


def state_preparation_errors(
        distribution: ErrorDistribution) -> Tuple[ErrorSet, ErrorSet]:
    # See Figure 1.c in https://www.nature.com/articles/srep19578.
    while True:
        x_errors = ErrorSet()
        z_errors = ErrorSet()
        # Inject errors on 1-qubit state preparation.
        for j in range(STEANE_CODE_SIZE):
            if distribution.has_preparation_error():
                # X error
                x_errors.add(j)
            if distribution.has_preparation_error():
                # Y error
                x_errors.add(j)
                z_errors.add(j)
            if distribution.has_preparation_error():
                # Z error
                z_errors.add(j)

        # Run H gates on these three qubits:
        for j in [1, 2, 3]:
            place_physical_h(x_errors, z_errors, j, distribution)

        # A list consisting of control and target qubit indices.
        cnots = [
           (1, 0), (3, 5), (2, 6), (1, 4), (2, 0), (3, 6), (1, 5),
           (6, 4), (0, 7), (5, 7), (6, 7)
        ]
        for (control, target) in cnots:
            place_physical_cnot(
                x_errors, z_errors, control, target, distribution)
        verified = not x_errors.get(7)
        if distribution.has_measurement_error():
            verified = not verified
        # Verification fails! Re-run initialization.
        if not verified:
            logging.info(
                'verification fails on state preparation! restarting...')
            continue

        # Verification succeeds! Return the error sets.
        # x_errors[7], z_errors[7] are on ancilla qubits which we'll never use.
        # Clear them.
        x_errors.clear(7)
        z_errors.clear(7)
        return (x_errors, z_errors)


# Verifies if the cat state is correctly set up by checking `x_errors` and
# `z_errors`, and returns the result.
# Note that this verification step itself may inject errors.
def verify_cat_state(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution) -> bool:
    for i in range(0, STEANE_CODE_SIZE):
        for j in range(i + 1, STEANE_CODE_SIZE):
            # Define a 1-qubit ancilla and initialize it with |0>.
            target = STEANE_CODE_SIZE
            if distribution.has_preparation_error():
                # X error
                x_errors.add(target)
            if distribution.has_preparation_error():
                # Y error
                x_errors.add(target)
                z_errors.add(target)
            if distribution.has_preparation_error():
                # Z error
                z_errors.add(target)

            place_physical_cnot(x_errors, z_errors, i, target, distribution)
            place_physical_cnot(x_errors, z_errors, j, target, distribution)

            verification_result = not x_errors.get(target)
            if distribution.has_measurement_error():
                verification_result = not verification_result

            # We'll never use the ancilla, so let's clear the errors.
            x_errors.clear(target)
            z_errors.clear(target)

            if not verification_result:
                return False

    return True


# Prepares a cat state 1/sqrt(2) * (|0...0> + |1...1>), and returns attached
# X errors and Z errors.
def prepare_cat_state(
        distribution: ErrorDistribution) -> Tuple[ErrorSet, ErrorSet]:
    while True:
        x_errors = ErrorSet()
        z_errors = ErrorSet()

        # Inject errors on 1-qubit state preparation.
        for j in range(STEANE_CODE_SIZE):
            if distribution.has_preparation_error():
                # X error
                x_errors.add(j)
            if distribution.has_preparation_error():
                # Y error
                x_errors.add(j)
                z_errors.add(j)
            if distribution.has_preparation_error():
                # Z error
                z_errors.add(j)

        # Run an H gate on the first qubit.
        place_physical_h(x_errors, z_errors, 0, distribution)

        # Add CNOTs to create the cat state.
        for target in range(STEANE_CODE_SIZE):
            place_physical_cnot(x_errors, z_errors, 0, target, distribution)

        if verify_cat_state(x_errors, z_errors, distribution):
            return (x_errors, z_errors)

        # Verification failed, let's set up the cat state again...


def place_measurement(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        state: qulacs.QuantumState,
        q_index: int,
        c_index: int,
        distribution: ErrorDistribution):
    results = []
    for trial in range(MEASUREMENT_REPETITION):
        # Create a cat state.
        (ancilla_x_errors, ancilla_z_errors) = prepare_cat_state(distribution)

        # Copy the ancilla error information.
        for i in range(STEANE_CODE_SIZE):
            if ancilla_x_errors.get(i):
                x_errors.add(i + STEANE_CODE_SIZE)
            if ancilla_z_errors.get(i):
                z_errors.add(i + STEANE_CODE_SIZE)

        for i in range(STEANE_CODE_SIZE):
            control = i + STEANE_CODE_SIZE
            target = i
            place_physical_h(x_errors, z_errors, target, distribution)
            place_physical_cnot(
                x_errors, z_errors, control, target, distribution)
            place_physical_h(x_errors, z_errors, target, distribution)

        for i in range(1, STEANE_CODE_SIZE):
            place_physical_cnot(
                x_errors, z_errors,
                i + STEANE_CODE_SIZE, CODE_SIZE, distribution)

        # Clear the ancilla errors.
        for i in range(STEANE_CODE_SIZE):
            x_errors.clear(i + STEANE_CODE_SIZE)
            z_errors.clear(i + STEANE_CODE_SIZE)

        # If errors accumulate, treat them as logical errors.
        move_logical_errors_to_state(x_errors, z_errors, state, q_index)

        qulacs.gate.Measurement(q_index, c_index).update_quantum_state(state)
        result = state.get_classical_value(c_index)

        if distribution.has_measurement_error():
            result = 1 - result
        results.append(result)
    most_common = collections.Counter(results).most_common(1)[0][0]
    state.set_classical_value(c_index, most_common)


def run_error_correction(
        x_errors: ErrorSet,
        z_errors: ErrorSet,
        distribution: ErrorDistribution):
    # Try to correct errors.
    (x_correction, z_correction) = guess_errors(
        x_errors, z_errors, distribution)
    x_errors += x_correction
    z_errors += z_correction

    for correction in [x_correction, z_correction]:
        for e in correction:
            if distribution.has_p1_error():
                # X errors
                x_errors.add(e)
            if distribution.has_p1_error():
                # Y errors
                x_errors.add(e)
                z_errors.add(e)
            if distribution.has_p1_error():
                # Z errors
                z_errors.add(e)


def simulate(circuit: qiskit.QuantumCircuit, distribution: ErrorDistribution):
    num_qubits = circuit.num_qubits
    state = qulacs.QuantumState(num_qubits)
    x_errors: List[ErrorSet] = []
    z_errors: List[ErrorSet] = []
    for i in range(num_qubits):
        t = state_preparation_errors(distribution)
        x_errors.append(t[0])
        # We can ignore Z errors, given we're preparaing a logical |0>.
        z_errors.append(ErrorSet())
        run_error_correction(x_errors[i], z_errors[i], distribution)
        move_logical_errors_to_state(
            x_errors[i], z_errors[i], state, i)

    for gate in circuit.data:
        logging.info('operation name = {}'.format(gate.operation.name))
        if gate.operation.name in {'h', 's', 'sdg'}:
            # `gate` is a one-qubit clifford gate
            index = circuit.qubits.index(gate.qubits[0])

            if gate.operation.name == 'h':
                # Run the logical operation.
                qulacs.gate.H(index).update_quantum_state(state)

                # Update the error data.
                # We swap X errors and Z errors, given HX = ZH and HZ = XH.
                (x_errors[index], z_errors[index]) = \
                    (z_errors[index], x_errors[index])
            elif gate.operation.name == 's':
                # Run the logical operation.
                qulacs.gate.S(index).update_quantum_state(state)

                # Update the error data.
                # ZS acts on each qubit, and ZSX = -YZS and (ZS)Z = Z(ZS).
                z_errors[index] += x_errors[index]
            elif gate.operation.name == 'sdg':
                # Run the logical operation.
                qulacs.gate.Sdag(index).update_quantum_state(state)

                # Update the error data.
                # This action is the inverse of the action for the S gate.
                z_errors[index] += x_errors[index]
            else:
                assert False, 'UNREACHABLE'

            # Inject errors generated by the logical operation.
            inject_p1_errors(x_errors[index], z_errors[index], distribution)

            run_error_correction(
                x_errors[index], z_errors[index], distribution)
            move_logical_errors_to_state(
                x_errors[index], z_errors[index],
                state, index)
        elif gate.operation.name == 'cx':
            # `gate` is a CNOT gate.
            control = circuit.qubits.index(gate.qubits[0])
            target = circuit.qubits.index(gate.qubits[1])

            # Run the logical operation.
            qulacs.gate.CNOT(control, target).update_quantum_state(state)

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
                x_errors[control], z_errors[control], distribution)
            move_logical_errors_to_state(
                x_errors[control], z_errors[control], state, control)
            run_error_correction(
                x_errors[target], z_errors[target], distribution)
            move_logical_errors_to_state(
                x_errors[target], z_errors[target], state, target)
        elif gate.operation.name == 'measure':
            q_index = circuit.qubits.index(gate.qubits[0])
            c_index = circuit.clbits.index(gate.clbits[0])
            place_measurement(
                x_errors[index],
                z_errors[index],
                state,
                q_index,
                c_index,
                distribution)
        else:
            raise RuntimeError(
                'Unsupported gate: {}'.format(gate.operation.name))
    return state


# Runs state prepration `n` times, and returns the number of tries having
# logical errors.
def run_state_preparation_and_count_errors(n, distribution: ErrorDistribution):
    count = 0
    for _ in range(n):
        (x_errors, z_errors) = state_preparation_errors(distribution)
        run_error_correction(x_errors, z_errors, distribution)
        # We can ignore logical Z errors, given we're preparaing a logical |0>.
        if calculate_deviation(x_errors) >= 2:
            count += 1
    return count
