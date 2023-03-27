import qiskit
import pandas as pd

from math import pi
from qiskit import QuantumCircuit
from fractions import Fraction
from typing import Tuple


# Given `a` and `b`, returns (`c`, (`d`, `e`)) where
#   - `c` is the greatest common divisor of `a` and `b`, and
#   - `d` * `a` + `e` * `b` == `c`.
def gcd(a: int, b: int) -> Tuple[int, Tuple[int, int]]:
    assert a >= 0
    assert b >= 0
    assert a > 0 or b > 0

    if a > b:
        (c, (d, e)) = gcd(b, a)
        return (c, (e, d))

    if a == 0:
        return (b, (0, 1))
    (c, (d, e)) = gcd(a, b % a)
    return (c, (d - e * (b // a), e))
        

# Adds gates which adds `a` to registers [offset:offset + n] in the
# Fourier space, to `circuit`.
# See 2.1 in https://arxiv.org/pdf/quant-ph/0205095.pdf for details.
# `a` must be non-negative and less than `2 ** n`.
def add(circuit: QuantumCircuit, a: int, n: int, offset: int):
    assert a >= 0
    assert a < 2 ** n
    assert circuit.num_qubits >= n + offset

    for i in range(n):
        for j in range(n - i):
            if (a & (1 << (n - 1 - (i + j)))) != 0:
                circuit.p(2 * pi / 2 ** (j + 1), i + offset)


# Adds gates which adds `a` mod `N` to registers
# `qreg[offset + 2, offset + n + 2]` in the Fourier space, with two controll
# qubits `qreg[offset]` and `qreg[offset + 1]`. We use the last register
# `qreg[offset + n + 2]` during the computation, but at the beginning and
# the end of the computation its value should be |0>.
# See 2.2 in https://arxiv.org/pdf/quant-ph/0205095.pdf for details.
# `N` must be positive and less than or equal to `2 ** (n - 1)`.
# `a` must be non-negative and less than `N` (and hence `2 ** (n - 1)`).
def c_add_mod(circuit: QuantumCircuit, a: int, n: int, N: int, offset: int):
    assert N > 0
    assert 2 * N <= 2 ** n
    assert a >= 0
    assert a < N
    assert circuit.num_qubits >= n + offset + 3

    if a == 0:
        return

    add_a = QuantumCircuit(n)
    add(add_a, a, n, 0)

    add_n = QuantumCircuit(n)
    add(add_n, N, n, 0)

    sub_a = QuantumCircuit(n)
    add(sub_a, 2 ** n - a, n, 0)

    qreg = circuit.qregs[0]

    circuit.append(
        add_a.control(2).decompose(), qreg[offset:offset + n + 2])
    add(circuit, 2 ** n - N, n, offset + 2)
    qft_dagger(circuit, n, offset + 2)
    circuit.cnot(offset + n + 1, offset + n + 2)
    qft(circuit, n, offset + 2)

    circuit.append(
        add_n.control(1).decompose(),
        [qreg[offset + n + 2]] + qreg[offset + 2:offset + n + 2])

    circuit.append(
        sub_a.control(2).decompose(), qreg[offset:offset + n + 2])
    qft_dagger(circuit, n, offset + 2)
    circuit.x(offset + n + 1)
    circuit.cnot(offset + n + 1, offset + n + 2)
    circuit.x(offset + n + 1)
    qft(circuit, n, offset + 2)
    circuit.append(
        add_a.control(2).decompose(), qreg[offset:offset + n + 2])


# Adds the following gates to `circuit`. The gates:
#    - are given 2`n` + 1 qubits as inputs, namely,
#      - `c`: 1 qubit, interpreted as a boolean,
#      - `x`: `n` qubits, a little-endian integer, and
#      - `b`: `n` qubits, a little-endian integer.
#    - use extra two qubits as ancillas. They'll be cleaned up during
#      the computation.
#    - replace `b` with `ax + b mod N`.
# See 2.3 in https://arxiv.org/pdf/quant-ph/0205095.pdf for details.
# `N` must be positive, and less than 2 ** `n`.
# `a` must be non-negative and less than `N`.
def c_mult_add_mod(circuit: QuantumCircuit, a: int, n: int, N: int):
    if a == 0:
        return
    qr = circuit.qregs[0]
    qft(circuit, n + 1, n + 1)
    for i in range(n):
        current_a = (2 ** i) * a % N
        name = 'add{}-mod{}'.format(current_a, N)
        c = QuantumCircuit(2 + n + 2, name=name)
        # We allocate `n + 1` qubits, not `n`, to meet `c_add_mod`'s
        # precondition: "2 * N <= 2 ** n"
        c_add_mod(c, current_a, n + 1, N, 0)
        circuit.append(
            c.decompose(),
            [qr[0], qr[i + 1]] + qr[n + 1:])

    qft_dagger(circuit, n + 1, n + 1)


# Adds the following gates to `circuit`. The gates:
#    - are given 2`n` + 1 qubits as inputs, namely,
#      - `c`: 1 qubit, interpreted as a boolean,
#      - `x`: `n` qubits, a little-endian integer, and
#    - use extra `n` + 2 qubits as ancillas. They'll be cleaned up during
#      the computation.
#    - replace `x` with `ax mod N`.
# See 2.3 in https://arxiv.org/pdf/quant-ph/0205095.pdf for details.
# `N` must be positive, and less than 2 ** `n`.
# `a` must be non-negative and less than `N`.
def c_mult_mod(circuit: QuantumCircuit, a: int, n: int, N: int):
    c_mult_add_mod(circuit, a, n, N)
    for i in range(n):
        circuit.cswap(0, i + 1, n + i + 1)

    (g, (d, e)) = gcd(a, N)
    assert g == 1
    a_inverse = d % N
    c = QuantumCircuit(2 * n + 3)
    c_mult_add_mod(c, a_inverse, n, N)
    circuit.append(c.inverse().decompose(), circuit.qregs[0][:c.num_qubits])


# Creates and returns a QuantumCircuit for phase estimation.
# The last measurements are not included in the circuit.
def phase_estimation_circuit_except_for_measurements(
        a: int, n: int, N: int) -> QuantumCircuit:
    assert a >= 0
    assert n > 0
    assert N > 0
    assert N < 2 ** n
    assert a < N

    circuit = QuantumCircuit(3 * n + 2, n)
    current_a = a
    qr = circuit.qregs[0]
    circuit.x(n)

    for i in range(n):
        circuit.h(i)
    for i in range(n):
        name = 'mult{}-mod{}'.format(current_a, N)
        c = QuantumCircuit(2 * n + 3, name=name)
        c_mult_mod(c, current_a, n, N)
        circuit.append(c.decompose(), [qr[i]] + qr[n:])
        current_a = current_a * current_a % N

    qft_dagger(circuit, n, 0)
    return circuit


# Creates and returns a QuantumCircuit for phase estimation.
def phase_estimation_circuit(a: int, n: int, N: int) -> QuantumCircuit:
    assert a >= 0
    assert n > 0
    assert N > 0
    assert N < 2 ** n
    assert a < N
    circuit = phase_estimation_circuit_except_for_measurements(a, n, N)
    circuit.measure(range(n), range(n))
    return circuit


# Analizes the result of the phase estimation measurements and outputs the
# analysis on stdout.
def report(r: qiskit.result.result.Result, n: int):
    rows = []
    measured_phases = []
    num_total_shots = sum(r.get_counts().values())

    for output, count in r.get_counts().items():
        if count / num_total_shots < 0.1:
            continue
        decimal = int(output, 2)
        phase = decimal / (2 ** n)
        measured_phases.append(phase)
        rows.append([f'{output}(bin) = {decimal:>3}(dec)',
                     f'{decimal}/{2**n} = {phase:.2f}'])
    headers = ['Register Output', 'Phase']
    df = pd.DataFrame(rows, columns=headers)
    print(df)

    rows = []
    for phase in measured_phases:
        frac = Fraction(phase).limit_denominator(n)
        rows.append([phase, f'{frac.numerator}/{frac.denominator}',
                     f'{frac.denominator}'])
    headers = ['Phase', 'Fraction', 'Guess for r']
    df = pd.DataFrame(rows, columns=headers)
    print(df)


def qft(circuit: QuantumCircuit, n: int, offset: int):
    def qft_step(n: int):
        circuit.h(n - 1 + offset)
        for i in range(n - 1):
            circuit.cp(pi / (2 ** (n - 1 - i)), i + offset, n - 1 + offset)

    for i in range(n):
        qft_step(n - i)
    for i in range(n // 2):
        circuit.swap(i + offset, n - 1 - i + offset)


def qft_dagger(circuit: QuantumCircuit, n: int, offset: int):
    c = QuantumCircuit(n, name='qft')
    qft(c, n, 0)

    circuit.append(c.inverse(), circuit.qubits[offset:offset + n])
