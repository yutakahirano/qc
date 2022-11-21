import unittest


from simulator import *


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

    def test_addition(self):
        self.assertEqual(ErrorSet() + ErrorSet(), ErrorSet())
        self.assertEqual(ErrorSet({1, 2}) + ErrorSet({2, 5}), ErrorSet({1, 5}))


class TestErrorGuessing(unittest.TestCase):
    def test_guessing(self):
        self.assertEqual(guess_errors(ErrorSet()), ErrorSet())

        # Every one error should be guessable.
        for i in range(7):
            errors = ErrorSet({i})
            self.assertEqual(guess_errors(errors), errors)

        self.assertEqual(guess_errors(ErrorSet({1, 3})), ErrorSet({5}))
