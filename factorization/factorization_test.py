import cmath
import factorization
import math
import qiskit_aer
import unittest


from functools import reduce
from math import pi
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


def run(circuit):
    sim = qiskit_aer.AerSimulator()
    c = circuit.copy()
    c.save_statevector()
    return sim.run(c, shots=1).result()


def tensor(a: Statevector, b: Statevector):
    return a.tensor(b)


def qft_expectation(a, n):
    result = None
    for i in range(n):
        arg = 0
        for j in range(i + 1):
            if a & (1 << (i - j)) != 0:
                arg += 2 * pi * 1j / (2 << j)
            r = Statevector([1, cmath.exp(arg)]) / math.sqrt(2)
        if result is None:
            result = r
        else:
            result = result.tensor(r)
    return result


class TestFactorization(unittest.TestCase):
    def test_gcd(self):
        def slow_gcd(a, b):
            if a == 0: return b
            if b == 0: return a

            for i in range(min(a, b), 0, -1):
                if a % i == 0 and b % i == 0:
                    return i

        for a in range(20):
            for b in range(20):
                if a == 0 and b == 0: continue
                with self.subTest(a=a, b=b):
                    (c, (d, e)) = factorization.gcd(a, b)
                    self.assertEqual(c, slow_gcd(a, b))
                    self.assertEqual(d * a + b * e, c)

        
    def test_qft_zero(self):
        c = QuantumCircuit(4)
        factorization.qft(c, 4, 0)
        result = run(c)
        expected = qft_expectation(0, 4)
        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_qft_one(self):
        n = 4
        offset = 1
        c = QuantumCircuit(n + offset)
        c.x(offset)
        factorization.qft(c, n, offset)
        result = run(c)
        expected = qft_expectation(1, n)
        expected = reduce(tensor, [
            qft_expectation(1, n),
            Statevector.from_int(0, 2 ** offset),
        ])
        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_qft(self):
        offset = 2
        n = 4
        for i in range(2 ** n):
            c = QuantumCircuit(n + offset)
            for j in range(n):
                if (i & (1 << j)) != 0:
                    c.x(j + offset)
            factorization.qft(c, n, offset)
            result = run(c)
            expected = qft_expectation(i, n)
            for j in range(offset):
                expected = expected.tensor(Statevector([1, 0]))
            self.assertAlmostEqual(
                1.0, state_fidelity(result.get_statevector(), expected))

    def test_qft_dagger(self):
        offset = 2
        n = 4

        for i in range(2 ** n):
            c = QuantumCircuit(n + offset)
            for j in range(n):
                if (i & (1 << j)) != 0:
                    c.x(n - 1 - j + offset)
            expected = run(c).get_statevector()
            factorization.qft(c, n, offset)
            factorization.qft_dagger(c, n, offset)
            result = run(c.decompose())
            self.assertAlmostEqual(
                1.0, state_fidelity(result.get_statevector(), expected))

    def test_add_ten(self):
        n = 4
        c = QuantumCircuit(n)
        factorization.qft(c, n, 0)
        factorization.add(c, 3, n, 0)
        factorization.add(c, 7, n, 0)
        factorization.qft_dagger(c, n, 0)
        result = run(c.decompose())
        expected = Statevector.from_int(10, 2 ** n)
        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_add(self):
        n = 4
        offset = 2
        for i in range(2 ** n):
            for j in range(2 ** n):
                c = QuantumCircuit(n + offset)
                factorization.qft(c, n, offset)
                factorization.add(c, i, n, offset)
                factorization.add(c, j, n, offset)
                factorization.qft_dagger(c, n, offset)
                result = run(c.decompose())
                expected = (
                    Statevector.from_int((i + j) % 2 ** n, 2 ** n).tensor(
                        Statevector.from_int(0, 2 ** offset)))
                self.assertAlmostEqual(
                    1.0, state_fidelity(result.get_statevector(), expected))

    def test_c_add_mod_nop(self):
        n = 4
        N = 7
        c = QuantumCircuit(n + 3)
        c.x(0)

        factorization.qft(c, n, 2)
        factorization.add(c, 1, n, 2)
        factorization.c_add_mod(c, 2, n, N, 0)
        factorization.qft_dagger(c, n, 2)
        result = run(c.decompose())
        expected = (
            Statevector.from_int(0, 2).tensor(
                Statevector.from_int(1, 2 ** n).tensor(
                    Statevector.from_int(1, 4))))

        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_c_add_mod_three(self):
        n = 4
        N = 7
        c = QuantumCircuit(n + 3)
        c.x(0)
        c.x(1)

        factorization.qft(c, n, 2)
        factorization.add(c, 1, n, 2)
        factorization.c_add_mod(c, 2, n, N, 0)
        factorization.qft_dagger(c, n, 2)
        result = run(c.decompose())
        expected = (
            Statevector.from_int(0, 2).tensor(
                Statevector.from_int(3, 2 ** n).tensor(
                    Statevector.from_int(3, 4))))

        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_c_add_mod_three_overflow(self):
        n = 4
        N = 7
        c = QuantumCircuit(n + 3)
        c.x(0)
        c.x(1)

        factorization.qft(c, n, 2)
        factorization.add(c, 5, n, 2)
        factorization.c_add_mod(c, 5, n, N, 0)
        factorization.qft_dagger(c, n, 2)
        result = run(c.decompose())
        expected = (
            Statevector.from_int(0, 2).tensor(
                Statevector.from_int(3, 2 ** n).tensor(
                    Statevector.from_int(3, 4))))

        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_c_add_mod(self):
        n = 4
        offset = 1
        for c1 in [False, True]:
            for c2 in [False, True]:
                for N in range(2, 2 ** (n - 1) + 1):
                    for i in range(N):
                        for j in range(N):
                            with self.subTest(c1=c1, c2=c2, N=N, i=i, j=j):
                                c = QuantumCircuit(n + 3 + offset)
                                conditional = 0
                                if c1:
                                    c.x(offset)
                                    conditional += 1
                                if c2:
                                    c.x(offset + 1)
                                    conditional += 2
                                factorization.qft(c, n, 2 + offset)
                                factorization.add(c, i, n, 2 + offset)
                                factorization.c_add_mod(c, j, n, N, offset)
                                factorization.qft_dagger(c, n, 2 + offset)
                                result = run(c.decompose())
                                int_result = (i + j) % N if c1 and c2 else i
                                expected = reduce(tensor, [
                                    Statevector.from_int(0, 2),
                                    Statevector.from_int(int_result, 2 ** n),
                                    Statevector.from_int(conditional, 4),
                                    Statevector.from_int(0, 2 ** offset)
                                ])
                                self.assertAlmostEqual(
                                    1.0, state_fidelity(
                                        result.get_statevector(), expected))

    def test_c_mult_add_mod_5_by_4(self):
        n = 3
        N = 7
        a = 4

        c = QuantumCircuit(2 * n + 3)
        # Turning the control qubit on.
        c.x(0)

        # Setting `x` as 5.
        c.x(1)
        c.x(3)

        # Setting `b` as 6.
        c.x(5)
        c.x(6)

        factorization.c_mult_add_mod(c, a, n, N)
        result = run(c.decompose())
        expected = reduce(tensor, [
            Statevector.from_int(0, 2 ** 2),
            Statevector.from_int(5, 2 ** n),
            Statevector.from_int(5, 2 ** n),
            Statevector.from_int(1, 2 ** 1)
        ])

        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))

    def test_c_mult_add_mod(self):
        n = 3
        for c_bit in [0, 1]:
            for N in range(2, 2 ** n):
                for a in range(N):
                    for x in range(N):
                        for b in range(N):
                            with self.subTest(c=c_bit, N=N, a=a, x=x, b=b):
                                c = QuantumCircuit(2 * n + 3)
                                if c_bit != 0:
                                    c.x(0)
                                for i in range(n):
                                    if (x & (1 << i)) != 0:
                                        c.x(i + 1)
                                    if (b & (1 << i)) != 0:
                                        c.x(n + i + 1)
                                factorization.c_mult_add_mod(c, a, n, N)
                                result = run(c.decompose())
                                expected = reduce(tensor, [
                                    Statevector.from_int(0, 2 ** 2),
                                    Statevector.from_int(
                                        (a * x + b) % N if c_bit != 0 else b,
                                        2 ** n),
                                    Statevector.from_int(x, 2 ** n),
                                    Statevector.from_int(c_bit, 2 ** 1)
                                ])
                                self.assertAlmostEqual(
                                    1.0, state_fidelity(
                                        result.get_statevector(), expected))

    def test_phase_estimation(self):
        n = 4
        a = 7
        N = 15
        plus = Statevector([1, 1]) / math.sqrt(2)

        c = factorization.phase_estimation_circuit_except_for_measurements(
            a, n, N)
        # We don't need the QFT dagger at the last of the circuit, so let's
        # cancel it.
        factorization.qft(c, n, 0)
        result = run(c.decompose())

        v = Statevector([0 for _ in range(2 ** (2 * n))])
        for i in range(2 ** n):
            v += reduce(tensor, [
                Statevector.from_int((a ** i) % N, 2 ** n),
                Statevector.from_int(i, 2 ** n)
            ]) / (2 ** (n / 2))
        plus = Statevector([1, 1]) / math.sqrt(2)

        expected = reduce(tensor, [
            Statevector.from_int(0, 2 ** (n + 2)),
            v
        ])
        self.assertAlmostEqual(
            1.0, state_fidelity(result.get_statevector(), expected))
