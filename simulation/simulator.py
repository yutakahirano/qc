from typing import Any, Iterable, List, Tuple
import cmath
import collections
import logging
import random
import time

import numpy as np
import qiskit
import qulacs
import qulacs.gate
import qulacs.state


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
NUM_MS_DISTILLATION_ANCILLA_QUBITS = 16
MEASUREMENT_REPETITION = 3


class ErrorDistribution:
    def __init__(self, p1, p2, p_measurement, p_preparation, p_t):
        self.p1 = p1
        self.p2 = p2
        self.p_measurement = p_measurement
        self.p_preparation = p_preparation
        self.p_t = p_t

    def has_p1_error(self):
        return random.uniform(0, 1) < self.p1

    def has_p2_error(self):
        return random.uniform(0, 1) < self.p2

    def has_measurement_error(self):
        return random.uniform(0, 1) < self.p_measurement

    def has_preparation_error(self):
        return random.uniform(0, 1) < self.p_preparation

    def has_unreliable_t_error(self):
        return random.uniform(0, 1) < self.p_t


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


# Places a logical measurement operation.
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
                i + STEANE_CODE_SIZE, STEANE_CODE_SIZE, distribution)

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


# Corrects errors.
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


# See 2.8.2 in https://arxiv.org/abs/1504.01444.
class MagicStateDistillator:
    # A cache to speed up magic state distillation. A Cache instance is
    # expected to be shared by multiple MagicStateDistillator instances.
    class Cache:
        def __init__(self):
            self.state = qulacs.QuantumState(
                NUM_MS_DISTILLATION_ANCILLA_QUBITS)
            self.has_value = False

    def __init__(self, cache: Cache, distribution: ErrorDistribution):
        self.x_errors: List[ErrorSet] = [
            ErrorSet() for _ in range(NUM_MS_DISTILLATION_ANCILLA_QUBITS)
        ]
        self.z_errors: List[ErrorSet] = [
            ErrorSet() for _ in range(NUM_MS_DISTILLATION_ANCILLA_QUBITS)
        ]
        self.logical_x_errors = ErrorSet()
        self.logical_z_errors = ErrorSet()
        self.logical_sx_errors = ErrorSet()
        self.cache = cache
        self.distribution = distribution

    # Runs the magic state distillation.
    # Generates a magic state (Tdag(H(|0>))) and resurns an gate that returns
    # the magic state when |0> is given (at the given qubit).
    #
    # This is the only public method (except for the constructor) in this
    # class.
    def run(self, index: int):
        count = 0
        distribution = self.distribution

        # Durations to measure perforamnce.
        d1 = 0.0
        d2 = 0.0
        d3 = 0.0
        d4 = 0.0
        d5 = 0.0
        last_time = time.time()
        while True:
            count += 1
            if count % 100 == 0:
                logging.info(
                    'magic state distribution: count = {}, duration = {}'
                    .format(count, time.time() - last_time))
                s = d1 + d2 + d3 + d4 + d5
                logging.info('relative cost: {}, {}, {}, {}, {}'.format(
                    d1 / s, d2 / s, d3 / s, d4 / s, d5 / s))
                d1 = 0.0
                d2 = 0.0
                d3 = 0.0
                d4 = 0.0
                d5 = 0.0

                last_time = time.time()
            t1 = time.time()
            self.reset_logical_qubits()

            self.place_h(0)
            self.place_h(1)
            self.place_h(3)
            self.place_h(7)
            self.place_h(14)

            self.place_cnot(14, 15)

            self.place_cnot(0, 5)
            self.place_cnot(0, 6)
            self.place_cnot(0, 9)
            self.place_cnot(0, 10)
            self.place_cnot(0, 11)
            self.place_cnot(0, 12)
            self.place_cnot(0, 14)

            self.place_cnot(1, 4)
            self.place_cnot(1, 6)
            self.place_cnot(1, 8)
            self.place_cnot(1, 10)
            self.place_cnot(1, 11)
            self.place_cnot(1, 13)
            self.place_cnot(1, 14)

            self.place_cnot(3, 2)
            self.place_cnot(3, 6)
            self.place_cnot(3, 8)
            self.place_cnot(3, 9)
            self.place_cnot(3, 12)
            self.place_cnot(3, 13)
            self.place_cnot(3, 14)

            self.place_cnot(7, 2)
            self.place_cnot(7, 4)
            self.place_cnot(7, 5)
            self.place_cnot(7, 10)
            self.place_cnot(7, 12)
            self.place_cnot(7, 13)
            self.place_cnot(7, 14)

            self.place_cnot(14, 2)
            self.place_cnot(14, 4)
            self.place_cnot(14, 5)
            self.place_cnot(14, 8)
            self.place_cnot(14, 9)
            self.place_cnot(14, 11)

            for i in range(15):
                self.place_t_but_error_correction(i)

            d1 += time.time() - t1
            t1 = time.time()

            if not self.cache.has_value:
                self.cache.has_value = True

            state = self.cache.state.copy()

            d2 += time.time() - t1
            t1 = time.time()

            # Let's add back logical errors if there are any...
            for i in range(NUM_MS_DISTILLATION_ANCILLA_QUBITS):
                assert not self.logical_x_errors.get(i)
                if self.logical_sx_errors.get(i):
                    qulacs.gate.X(i).update_quantum_state(state)
                    qulacs.gate.S(i).update_quantum_state(state)
                    self.logical_sx_errors.add(i)
                if self.logical_z_errors.get(i):
                    qulacs.gate.Z(i).update_quantum_state(state)
                    self.logical_z_errors.add(i)

            d3 += time.time() - t1
            t1 = time.time()
            for i in range(15):
                # Correct errors caused by the T operations above.
                run_error_correction(
                    self.x_errors[i], self.z_errors[i], distribution)
                move_logical_errors_to_state(
                    self.x_errors[i], self.z_errors[i], state, i)

                self.place_h_after_t(state, i)

            d4 += time.time() - t1
            t1 = time.time()
            measurement_durations = []
            has_measurement_error = False
            for i in range(15):
                cl_index = 0
                place_measurement(
                    self.x_errors[i], self.z_errors[i],
                    state, i, cl_index, self.distribution)
                if state.get_classical_value(cl_index) != 0:
                    has_measurement_error = True
                    break
            measurement_durations.append(time.time() - t1)
            if not has_measurement_error:
                # As we've measured all the other qubits, this density matrix
                # is pure.
                dm = qulacs.state.partial_trace(state, list(range(15)))
                assert abs(dm.get_matrix()[0][0].imag) < 1e-6
                assert abs(dm.get_matrix()[1][1].imag) < 1e-6
                a = cmath.sqrt(dm.get_matrix()[0][0])
                b = dm.get_matrix()[1][0] / a
                vector = state.get_vector()
                gate_data: np.ndarray[np.complex128, Any] = np.ndarray(
                    shape=(2, 2), dtype=np.complex128)
                gate_data[0][0] = a
                gate_data[0][1] = -b.conjugate()
                gate_data[1][0] = b
                gate_data[1][1] = a.conjugate()
                return qulacs.gate.DenseMatrix(index, gate_data)
            d5 += time.time() - t1
            t1 = time.time()

    def reset_logical_qubits(self):
        num_qubits = NUM_MS_DISTILLATION_ANCILLA_QUBITS
        self.x_errors = [ErrorSet() for _ in range(num_qubits)]
        self.z_errors = [ErrorSet() for _ in range(num_qubits)]
        self.logical_x_errors = ErrorSet()
        self.logical_z_errors = ErrorSet()
        self.logical_sx_errors = ErrorSet()
        for index in range(num_qubits):
            t = state_preparation_errors(self.distribution)
            self.x_errors[index] += t[0]
            # We can ignore Z errors, given we're preparaing a logical |0>.

            run_error_correction(
                self.x_errors[index], self.z_errors[index], self.distribution)
            self.move_logical_errors(index)

    def place_h(self, index: int):
        x_errors = self.x_errors
        z_errors = self.z_errors
        distribution = self.distribution

        (x_errors[index], z_errors[index]) = (z_errors[index], x_errors[index])
        logical_x_error = self.logical_x_errors.get(index)
        logical_z_error = self.logical_z_errors.get(index)
        if logical_x_error:
            self.logical_z_errors.add(index)
        if logical_z_error:
            self.logical_x_errors.add(index)

        if not self.cache.has_value:
            qulacs.gate.H(index).update_quantum_state(self.cache.state)

        inject_p1_errors(x_errors[index], z_errors[index], distribution)
        run_error_correction(x_errors[index], z_errors[index], distribution)
        self.move_logical_errors(index)

    def place_cnot(self, control: int, target: int):
        t1 = time.time()
        x_errors = self.x_errors
        z_errors = self.z_errors
        distribution = self.distribution

        x_errors[target] += x_errors[control]
        z_errors[control] += z_errors[target]
        if self.logical_x_errors.get(control):
            self.logical_x_errors.add(target)
        if self.logical_z_errors.get(target):
            self.logical_z_errors.add(control)

        if not self.cache.has_value:
            qulacs.gate.CNOT(control, target).update_quantum_state(
                self.cache.state)

        inject_p2_errors(x_errors, z_errors, control, target, distribution)

        run_error_correction(
            x_errors[control], z_errors[control], distribution)
        self.move_logical_errors(control)
        run_error_correction(x_errors[target], z_errors[target], distribution)
        self.move_logical_errors(target)

    def place_t_but_error_correction(self, index: int):
        x_errors = self.x_errors
        z_errors = self.z_errors
        distribution = self.distribution

        # We cannot simulate T actions on physical X errors and Z errors. Let's
        # assume leving it would bring us good approximation...

        # TX = SXT up to a global phase.
        if self.logical_x_errors.get(index):
            # Clear the X error.
            self.logical_x_errors.add(index)
            self.logical_sx_errors.add(index)
        # T and Z commute.

        if not self.cache.has_value:
            qulacs.gate.T(index).update_quantum_state(self.cache.state)

        for i in range(STEANE_CODE_SIZE):
            if distribution.has_unreliable_t_error():
                # X errors
                x_errors[index].add(i)
            if distribution.has_unreliable_t_error():
                # Y errors
                x_errors[index].add(i)
                z_errors[index].add(i)
            if distribution.has_unreliable_t_error():
                # Z errors
                z_errors[index].add(i)

    def place_h_after_t(self, state: qulacs.QuantumState, index: int):
        x_errors = self.x_errors
        z_errors = self.z_errors
        distribution = self.distribution

        (x_errors[index], z_errors[index]) = (z_errors[index], x_errors[index])

        # We stopped tracking logical errors and instead applying logical
        # errors immediately.
        assert self.logical_x_errors.get(index) == 0
        assert self.logical_z_errors.get(index) == 0

        qulacs.gate.H(index).update_quantum_state(state)

        for i in range(STEANE_CODE_SIZE):
            if distribution.has_unreliable_t_error():
                # X errors
                x_errors[index].add(i)
            if distribution.has_unreliable_t_error():
                # Y errors
                x_errors[index].add(i)
                z_errors[index].add(i)
            if distribution.has_unreliable_t_error():
                # Z errors
                z_errors[index].add(i)

        inject_p1_errors(x_errors[index], z_errors[index], distribution)
        run_error_correction(x_errors[index], z_errors[index], distribution)
        move_logical_errors_to_state(
            x_errors[index], z_errors[index], state, index)

    def move_logical_errors(self, index: int):
        x_errors = self.x_errors[index]
        z_errors = self.z_errors[index]

        if calculate_deviation(x_errors) >= 2:
            # There is no physical operation corresponding to this, so we
            # don't need to think about errors.
            x_errors += ErrorSet(range(STEANE_CODE_SIZE))
            self.logical_x_errors.add(index)
        if calculate_deviation(z_errors) >= 2:
            # There is no physical operation corresponding to this, so we
            # don't need to think about errors.
            z_errors += ErrorSet(range(STEANE_CODE_SIZE))
            self.logical_z_errors.add(index)


# Places a logical H operation.
def place_logical_h(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # We swap X errors and Z errors, given HX = ZH and HZ = XH.
    (x_errors[index], z_errors[index]) = (z_errors[index], x_errors[index])

    # Run the logical operation.
    qulacs.gate.H(index).update_quantum_state(state)
    inject_p1_errors(x_errors[index], z_errors[index], distribution)

    run_error_correction(x_errors[index], z_errors[index], distribution)
    move_logical_errors_to_state(
        x_errors[index], z_errors[index], state, index)


# Places a logical S operation.
def place_logical_s(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # ZS acts on each qubit, and ZSX = -YZS and (ZS)Z = Z(ZS).
    z_errors[index] += x_errors[index]

    # Run the logical operation.
    qulacs.gate.S(index).update_quantum_state(state)
    inject_p1_errors(x_errors[index], z_errors[index], distribution)

    run_error_correction(x_errors[index], z_errors[index], distribution)
    move_logical_errors_to_state(
        x_errors[index], z_errors[index], state, index)


# Places a logical Sdg operation.
def place_logical_sdg(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # This action is the inverse of the action for the S gate.
    z_errors[index] += x_errors[index]

    # Run the logical operation.
    qulacs.gate.Sdag(index).update_quantum_state(state)
    inject_p1_errors(x_errors[index], z_errors[index], distribution)

    run_error_correction(x_errors[index], z_errors[index], distribution)
    move_logical_errors_to_state(
        x_errors[index], z_errors[index], state, index)


# Places a logical CNOT operation.
def place_logical_cnot(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        control: int,
        target: int,
        distribution: ErrorDistribution):
    # Update the error data.
    # CX * X_CONTROL  = X_CONTROL * X_TARGET * CX
    # CX * X_TARGET   = X_TARGET * CX
    # CX * Z_CONTROL  = Z_CONTROL * CX
    # CX * Z_TARGET   = Z_CONTROL * Z_TARGET * CX
    x_errors[target] += x_errors[control]
    z_errors[control] += z_errors[target]

    # Run the logical operation.
    qulacs.gate.CNOT(control, target).update_quantum_state(state)
    inject_p2_errors(x_errors, z_errors, control, target, distribution)

    run_error_correction(x_errors[control], z_errors[control], distribution)
    move_logical_errors_to_state(
        x_errors[control], z_errors[control], state, control)
    run_error_correction(x_errors[target], z_errors[target], distribution)
    move_logical_errors_to_state(
        x_errors[target], z_errors[target], state, target)


# Resets given qubits in the given QuantumState.
def reset_logical_qubits(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        qubit_indices: Iterable[int],
        cl_index: int,
        distribution: ErrorDistribution):
    for index in qubit_indices:
        # These operations clears the qubit in `state`.
        qulacs.gate.Measurement(index, cl_index).update_quantum_state(state)
        if state.get_classical_value(cl_index) == 1:
            qulacs.gate.X(index).update_quantum_state(state)

        # These operations simulate the errors.
        t = state_preparation_errors(distribution)
        x_errors[index] += t[0]
        # We can ignore Z errors, given we're preparaing a logical |0>.

        run_error_correction(x_errors[index], z_errors[index], distribution)
        move_logical_errors_to_state(
            x_errors[index], z_errors[index], state, index)


# Prepares a magic state distillation, without simulating errors.
def fast_magic_state_distillation(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        cl_index: int,
        distribution: ErrorDistribution):
    num_qubits = NUM_MS_DISTILLATION_ANCILLA_QUBITS
    reset_logical_qubits(
        x_errors, z_errors,
        state, range(index, index + num_qubits), cl_index, distribution)
    qulacs.gate.H(index + num_qubits - 1).update_quantum_state(state)
    qulacs.gate.Tdag(index + num_qubits - 1).update_quantum_state(state)


# Places a logical T operation.
def place_logical_t(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        magic_state_index: int,
        cl_index: int,
        magic_state_distillation_cache: MagicStateDistillator.Cache,
        distribution: ErrorDistribution,
        simulate_magic_state_distillation=True):
    if simulate_magic_state_distillation:
        reset_logical_qubits(
            x_errors, z_errors,
            state, [magic_state_index], cl_index, distribution)
        gate = MagicStateDistillator(
            magic_state_distillation_cache,
            distribution).run(magic_state_index)
        gate.update_quantum_state(state)
    else:
        fast_magic_state_distillation(
            x_errors, z_errors,
            state, magic_state_index, cl_index, distribution)
    # We have a magic state at `magic_state_index`!
    place_logical_s(x_errors, z_errors, state, magic_state_index, distribution)
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)

    place_measurement(
        x_errors[index], z_errors[index], state, index, cl_index, distribution)
    if state.get_classical_value(cl_index) == 1:
        place_logical_h(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_s(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_s(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_h(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_s(
            x_errors, z_errors, state, magic_state_index, distribution)

    # Swap the qubits.
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)
    place_logical_cnot(
        x_errors, z_errors, state, index, magic_state_index, distribution)
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)

    reset_logical_qubits(
        x_errors, z_errors, state, [magic_state_index], cl_index, distribution)


# Places a logical Tdg operation.
def place_logical_tdg(
        x_errors: List[ErrorSet],
        z_errors: List[ErrorSet],
        state: qulacs.QuantumState,
        index: int,
        magic_state_index: int,
        cl_index: int,
        magic_state_distillation_cache: MagicStateDistillator.Cache,
        distribution: ErrorDistribution,
        simulate_magic_state_distillation=True):
    if simulate_magic_state_distillation:
        reset_logical_qubits(
            x_errors, z_errors,
            state, [magic_state_index], cl_index, distribution)
        gate = MagicStateDistillator(
            magic_state_distillation_cache,
            distribution).run(magic_state_index)
        gate.update_quantum_state(state)
    else:
        fast_magic_state_distillation(
            x_errors, z_errors,
            state, magic_state_index, cl_index, distribution)
    # We have a magic state at `magic_state_index`!
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)

    place_measurement(
        x_errors[index], z_errors[index], state, index, cl_index, distribution)
    if state.get_classical_value(cl_index) == 1:
        place_logical_h(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_s(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_s(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_h(
            x_errors, z_errors, state, magic_state_index, distribution)
        place_logical_sdg(
            x_errors, z_errors, state, magic_state_index, distribution)

    # Swap the qubits.
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)
    place_logical_cnot(
        x_errors, z_errors, state, index, magic_state_index, distribution)
    place_logical_cnot(
        x_errors, z_errors, state, magic_state_index, index, distribution)

    reset_logical_qubits(
        x_errors, z_errors, state, [magic_state_index], cl_index, distribution)


def simulate(
        circuit: qiskit.QuantumCircuit, distribution: ErrorDistribution,
        simulate_magic_state_distillation: bool = True):
    num_qubits = circuit.num_qubits + 1
    magic_state_ancilla_index = circuit.num_qubits
    utility_cl_index = circuit.num_clbits
    state = qulacs.QuantumState(num_qubits)
    x_errors: List[ErrorSet] = [ErrorSet() for _ in range(num_qubits)]
    z_errors: List[ErrorSet] = [ErrorSet() for _ in range(num_qubits)]

    magic_state_distillation_cache = MagicStateDistillator.Cache()

    reset_logical_qubits(
        x_errors, z_errors,
        state, range(num_qubits), utility_cl_index, distribution)

    for gate in circuit.data:
        logging.info('operation name = {}'.format(gate.operation.name))
        if gate.operation.name == 'h':
            index = circuit.qubits.index(gate.qubits[0])
            place_logical_h(x_errors, z_errors, state, index, distribution)
        elif gate.operation.name == 's':
            index = circuit.qubits.index(gate.qubits[0])
            place_logical_s(x_errors, z_errors, state, index, distribution)
        elif gate.operation.name == 'sdg':
            index = circuit.qubits.index(gate.qubits[0])
            place_logical_sdg(x_errors, z_errors, state, index, distribution)
        elif gate.operation.name == 't':
            index = circuit.qubits.index(gate.qubits[0])
            place_logical_t(
                x_errors, z_errors,
                state, index, magic_state_ancilla_index, utility_cl_index,
                magic_state_distillation_cache,
                distribution, simulate_magic_state_distillation)
        elif gate.operation.name == 'tdg':
            index = circuit.qubits.index(gate.qubits[0])
            place_logical_tdg(
                x_errors, z_errors,
                state, index, magic_state_ancilla_index, utility_cl_index,
                magic_state_distillation_cache,
                distribution, simulate_magic_state_distillation)
        elif gate.operation.name == 'cx':
            control = circuit.qubits.index(gate.qubits[0])
            target = circuit.qubits.index(gate.qubits[1])
            place_logical_cnot(
                x_errors, z_errors, state, control, target, distribution)
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
