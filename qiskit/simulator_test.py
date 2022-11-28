import unittest
import qiskit
import qulacs

from simulator import *


class CountErrorDistribution(ErrorDistribution):
    def __init__(self, count):
        self.current = 0
        self.count = count

    def has_p1_error(self):
        self.current += 1
        return self.current - 1 == self.count

    def has_p2_error(self):
        self.current += 1
        return self.current - 1 == self.count

    def has_measurement_error(self):
        self.current += 1
        return self.current - 1 == self.count

    def has_preparation_error(self):
        self.current += 1
        return self.current - 1 == self.count

    def has_caused_error(self):
        return self.current >= self.count


class TestErrorSet(unittest.TestCase):
    def test_iteration(self):
        self.assertEqual(list(ErrorSet()), [])
        self.assertEqual(list(ErrorSet({1, 7, 2})), [1, 2, 7])

    def test_equality(self):
        self.assertEqual(ErrorSet(), ErrorSet())
        self.assertEqual(ErrorSet({1, 3}), ErrorSet({3, 1}))
        self.assertNotEqual(ErrorSet(), ErrorSet({1}))
        self.assertNotEqual(ErrorSet({1, 3}), ErrorSet({1}))

    def test_length(self):
        self.assertEqual(len(ErrorSet()), 0)
        self.assertEqual(len(ErrorSet({1, 2, 3, 4})), 4)

    def test_add(self):
        errors = ErrorSet({2})
        errors.add(6)
        self.assertEqual(errors, ErrorSet({2, 6}))
        errors.add(2)
        self.assertEqual(errors, ErrorSet({6}))

    def test_get(self):
        errors = ErrorSet({1, 3, 7})
        self.assertFalse(errors.get(0))
        self.assertTrue(errors.get(1))
        self.assertFalse(errors.get(2))
        self.assertTrue(errors.get(3))
        self.assertFalse(errors.get(4))
        self.assertFalse(errors.get(5))
        self.assertFalse(errors.get(6))
        self.assertTrue(errors.get(7))
        self.assertFalse(errors.get(8))

    def test_clear(self):
        errors = ErrorSet({1, 3, 8})
        errors.clear(0)
        self.assertEqual(errors, ErrorSet({1, 3, 8}))
        errors.clear(1)
        self.assertEqual(errors, ErrorSet({3, 8}))
        errors.clear(2)
        self.assertEqual(errors, ErrorSet({3, 8}))
        errors.clear(3)
        self.assertEqual(errors, ErrorSet({8}))

    def test_addition(self):
        self.assertEqual(ErrorSet() + ErrorSet(), ErrorSet())
        self.assertEqual(ErrorSet({1, 2}) + ErrorSet({2, 5}), ErrorSet({1, 5}))

    def test_inplace_addition(self):
        errors = ErrorSet({1, 2, 3})
        original = errors
        errors += ErrorSet({7, 2})

        self.assertEqual(errors, ErrorSet({1, 3, 7}))
        self.assertEqual(original, ErrorSet({1, 3, 7}))


class TestErrorGuessing(unittest.TestCase):
    def test_deviation(self):
        self.assertEqual(0, calculate_deviation(ErrorSet({})))
        self.assertEqual(0, calculate_deviation(ErrorSet({3, 4, 5, 6})))
        self.assertEqual(0, calculate_deviation(ErrorSet({1, 2, 5, 6})))
        self.assertEqual(0, calculate_deviation(ErrorSet({0, 2, 4, 6})))
        self.assertEqual(0, calculate_deviation(ErrorSet({1, 2, 3, 4})))
        self.assertEqual(0, calculate_deviation(ErrorSet({0, 1, 3, 6})))
        self.assertEqual(1, calculate_deviation(ErrorSet({2})))
        self.assertEqual(2, calculate_deviation(ErrorSet({2, 3})))
        self.assertEqual(1, calculate_deviation(ErrorSet({3, 4, 5})))
        self.assertEqual(3, calculate_deviation(ErrorSet({1, 3, 5})))

    def test_x_stabilizer_measurement_with_no_error(self):
        distribution = CountErrorDistribution(-1)
        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [3, 4, 5, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [1, 2, 5, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [0, 2, 4, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [3, 4, 5, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [1, 2, 5, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_x_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [0, 2, 4, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

    def test_x_stabilizer_measurement_with_input_error(self):
        distribution = CountErrorDistribution(-1)
        x_errors = ErrorSet({1, 2, 4, 5})
        z_errors = ErrorSet({2, 4})
        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=True)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({1, 2, 4, 5}))
        self.assertEqual(z_errors, ErrorSet({2, 4}))

        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=False)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({1, 2, 4, 5}))
        self.assertEqual(z_errors, ErrorSet({2, 4}))

        x_errors = ErrorSet({1})
        z_errors = ErrorSet({4})
        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, [0, 2, 4, 6], distribution, with_flag=True)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({1}))
        self.assertEqual(z_errors, ErrorSet({4}))

        x_errors = ErrorSet({4})
        z_errors = ErrorSet({1})
        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, [0, 2, 4, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({4}))
        self.assertEqual(z_errors, ErrorSet({1}))

        x_errors = ErrorSet({})
        z_errors = ErrorSet({5, 6})
        (r1, r2) = run_x_stabilizer_measurement(
            x_errors, z_errors, [0, 2, 4, 6], distribution, with_flag=False)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({}))
        self.assertEqual(z_errors, ErrorSet({5, 6}))

    def test_x_stabilizer_measurement_fault_tolerance(self):
        saw_error = True
        i = -1
        while saw_error:
            i += 1
            distribution = CountErrorDistribution(i)
            x_errors = ErrorSet({})
            z_errors = ErrorSet({})
            for j in range(CODE_SIZE):
                if distribution.has_preparation_error():
                    # X error
                    x_errors.add(j)
                if distribution.has_preparation_error():
                    # Y error
                    x_errors.add(j)
                    z_errors.add(j)
                if distribution.has_preparation_error():
                    # X error
                    z_errors.add(j)
            (r1, r2) = run_x_stabilizer_measurement(
                x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=True)
            saw_error = distribution.has_caused_error()

            # It is possible that multiple qubits are affected by errors, but
            # in that case `r2` must be false.
            self.assertLess(calculate_deviation(x_errors), 3)
            if calculate_deviation(x_errors) == 2:
                self.assertFalse(r2)
            self.assertLess(calculate_deviation(z_errors), 2)

    def test_z_stabilizer_measurement_with_no_error(self):
        distribution = CountErrorDistribution(-1)
        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [3, 4, 5, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [1, 2, 5, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [0, 2, 4, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [3, 4, 5, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [1, 2, 5, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

        (r1, r2) = run_z_stabilizer_measurement(
            ErrorSet(), ErrorSet(), [0, 2, 4, 6], distribution,
            with_flag=False)
        self.assertTrue(r1)
        self.assertTrue(r2)

    def test_z_stabilizer_measurement_with_input_error(self):
        distribution = CountErrorDistribution(-1)
        x_errors = ErrorSet({2, 4})
        z_errors = ErrorSet({1, 2, 4, 5})
        (r1, r2) = run_z_stabilizer_measurement(
            x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=True)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({2, 4}))
        self.assertEqual(z_errors, ErrorSet({1, 2, 4, 5}))

        (r1, r2) = run_z_stabilizer_measurement(
            x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=False)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({2, 4}))
        self.assertEqual(z_errors, ErrorSet({1, 2, 4, 5}))

        x_errors = ErrorSet({4})
        z_errors = ErrorSet({1})
        (r1, r2) = run_z_stabilizer_measurement(
            x_errors, z_errors, [0, 2, 4, 6], distribution, with_flag=True)
        self.assertFalse(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({4}))
        self.assertEqual(z_errors, ErrorSet({1}))

        x_errors = ErrorSet({1})
        z_errors = ErrorSet({4})
        (r1, r2) = run_z_stabilizer_measurement(
            x_errors, z_errors, [0, 2, 4, 6], distribution, with_flag=True)
        self.assertTrue(r1)
        self.assertTrue(r2)
        # `x_errors` and `z_errors` shouldn't mutate.
        self.assertEqual(x_errors, ErrorSet({1}))
        self.assertEqual(z_errors, ErrorSet({4}))

    def test_z_stabilizer_measurement_fault_tolerance(self):
        saw_error = True
        i = -1
        while saw_error:
            i += 1
            distribution = CountErrorDistribution(i)
            x_errors = ErrorSet({})
            z_errors = ErrorSet({})
            for j in range(CODE_SIZE):
                if distribution.has_preparation_error():
                    # X error
                    x_errors.add(j)
                if distribution.has_preparation_error():
                    # Y error
                    x_errors.add(j)
                    z_errors.add(j)
                if distribution.has_preparation_error():
                    # X error
                    z_errors.add(j)
            (r1, r2) = run_z_stabilizer_measurement(
                x_errors, z_errors, [3, 4, 5, 6], distribution, with_flag=True)
            saw_error = distribution.has_caused_error()

            # It is possible that multiple qubits are affected by errors, but
            # in that case `r2` must be false.
            self.assertLess(calculate_deviation(z_errors), 3)
            if calculate_deviation(z_errors) == 2:
                self.assertFalse(r2)
            self.assertLess(calculate_deviation(x_errors), 2)

    def test_guessing_with_input_error(self):
        distribution = CountErrorDistribution(-1)
        for i in range(7):
            x_errors = ErrorSet({i})
            z_errors = ErrorSet()
            (guessed_x_errors, guessed_z_errors) = guess_errors(
                x_errors, z_errors, distribution)
            self.assertEqual(guessed_x_errors, ErrorSet({i}))
            self.assertEqual(guessed_z_errors, ErrorSet())
            self.assertEqual(x_errors, ErrorSet({i}))
            self.assertEqual(z_errors, ErrorSet())

            x_errors = ErrorSet()
            z_errors = ErrorSet({i})
            (guessed_x_errors, guessed_z_errors) = guess_errors(
                x_errors, z_errors, distribution)
            self.assertEqual(guessed_x_errors, ErrorSet())
            self.assertEqual(guessed_z_errors, ErrorSet({i}))
            self.assertEqual(x_errors, ErrorSet())
            self.assertEqual(z_errors, ErrorSet({i}))

    def test_guessing_fault_tolerance(self):
        saw_error = True
        i = -1
        while saw_error:
            i += 1
            distribution = CountErrorDistribution(i)
            x_errors = ErrorSet({})
            z_errors = ErrorSet({})
            for j in range(CODE_SIZE):
                if distribution.has_preparation_error():
                    # X error
                    x_errors.add(j)
                if distribution.has_preparation_error():
                    # Y error
                    x_errors.add(j)
                    z_errors.add(j)
                if distribution.has_preparation_error():
                    # X error
                    z_errors.add(j)

            (guessed_x_errors, guessed_z_errors) = guess_errors(
                x_errors, z_errors, distribution)
            saw_error = distribution.has_caused_error()

            self.assertLess(
                calculate_deviation(x_errors + guessed_x_errors), 2)
            self.assertLess(
                calculate_deviation(z_errors + guessed_z_errors), 2)


class TestStatePreparation(unittest.TestCase):
    def test_fault_tolerance(self):
        saw_error = True
        i = -1
        while saw_error:
            i += 1
            distribution = CountErrorDistribution(i)
            (x_errors, z_errors) = state_preparation_errors(distribution)
            saw_error = distribution.has_caused_error()

            # We ignore Z errors because we're preparing a logical |0>.
            self.assertLess(calculate_deviation(x_errors), 2)


class TestMeasurement(unittest.TestCase):
    def test_cat_state(self):
        saw_error = True
        i = -1
        while saw_error:
            i += 1
            distribution = CountErrorDistribution(i)
            (x_errors, z_errors) = prepare_cat_state(distribution)

            saw_error = distribution.has_caused_error()

            # We cannot use `calculate_deviation`, given the cat state is not
            # a valid encoded state.

            # We don't take care of Z errors because such errors should be
            # taken care by the majority vote of multiple measurement results.

            if len(set(x_errors)) < 2:
                # OK: Errors are found at most one qubit.
                continue

            if len(set(x_errors)) > CODE_SIZE - 2:
                # Swapping 0 and 1 for all qubits doesn't change the cat state,
                # so this is OK too.
                continue

            self.assertTrue(false)


class TestSimulation(unittest.TestCase):
    def test_simulation(self):
        # A smoke test checking that the simulation works when there are no
        # errors.

        circuit = qiskit.QuantumCircuit(4, 0)
        circuit.sdg(0)
        circuit.h(0)
        circuit.h(1)
        circuit.cnot(2, 1)
        circuit.cnot(0, 2)
        circuit.s(2)
        circuit.s(2)
        circuit.s(0)
        circuit.sdg(1)
        circuit.h(2)

        distribution = ErrorDistribution(0, 0, 0, 0)
        state = simulate(circuit, distribution)

        state2 = qulacs.QuantumState(4)
        qulacs.gate.Sdag(0).update_quantum_state(state2)
        qulacs.gate.H(0).update_quantum_state(state2)
        qulacs.gate.H(1).update_quantum_state(state2)
        qulacs.gate.CNOT(2, 1).update_quantum_state(state2)
        qulacs.gate.CNOT(0, 2).update_quantum_state(state2)
        qulacs.gate.S(2).update_quantum_state(state2)
        qulacs.gate.S(2).update_quantum_state(state2)
        qulacs.gate.S(0).update_quantum_state(state2)
        qulacs.gate.Sdag(1).update_quantum_state(state2)
        qulacs.gate.H(2).update_quantum_state(state2)

        difference = abs(state.get_vector() - state2.get_vector())
        for e in difference:
            self.assertLess(e, 1e-6)
