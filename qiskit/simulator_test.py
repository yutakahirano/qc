import unittest


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

    def test_guessing(self):
        self.assertEqual(guess_errors(ErrorSet()), ErrorSet())

        # Every one error should be guessable.
        for i in range(7):
            errors = ErrorSet({i})
            self.assertEqual(guess_errors(errors), errors)

        self.assertEqual(guess_errors(ErrorSet({1, 3})), ErrorSet({5}))


class TestStatePreparation(unittest.TestCase):
    def test_fault_tolerance(self):
        for i in range(1000):
            distribution = CountErrorDistribution(i)
            (x_errors, z_errors) = state_preparation_errors(distribution)
            # We ignore Z errors because we're preparing a logical |0>.
            self.assertLess(calculate_deviation(x_errors), 2)
