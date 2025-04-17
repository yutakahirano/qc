from __future__ import annotations

import argparse
import enum
import numpy as np
import sinter
import stim

from enum import auto
from collections.abc import Callable
from util import QubitMapping, Circuit, MeasurementIdentifier, ObservableIdentifier
from surface_code import SurfaceStabilizerPattern
from surface_code import SurfaceSyndromeMeasurement, SurfaceXSyndromeMeasurement, SurfaceZSyndromeMeasurement


class InitialValue(enum.Enum):
    Plus = auto(),
    Zero = auto(),


class SurfaceCodeInitialization:
    def __init__(self, mapping: QubitMapping, distance: int, initial_value: InitialValue,
                 error_probability: float, full_post_selection: bool) -> None:
        self.mapping = mapping
        self.distance = distance
        self.circuit = Circuit(mapping, error_probability)
        self.initial_value = initial_value
        self.full_post_selection = full_post_selection
        self.surface_syndrome_measurements: dict[tuple[int, int], SurfaceSyndromeMeasurement] = {}
        self.surface_x_measurements: dict[tuple[int, int], MeasurementIdentifier] = {}
        self.surface_z_measurements: dict[tuple[int, int], MeasurementIdentifier] = {}

        self.offset_x = 1
        self.offset_y = 1

        self._setup_syndrome_measurements()

    def _setup_syndrome_measurements(self) -> None:
        distance = self.distance
        offset_x = self.offset_x
        offset_y = self.offset_y

        FOUR_WEIGHT = SurfaceStabilizerPattern.FOUR_WEIGHT
        TWO_WEIGHT_UP = SurfaceStabilizerPattern.TWO_WEIGHT_UP
        TWO_WEIGHT_DOWN = SurfaceStabilizerPattern.TWO_WEIGHT_DOWN
        TWO_WEIGHT_LEFT = SurfaceStabilizerPattern.TWO_WEIGHT_LEFT
        TWO_WEIGHT_RIGHT = SurfaceStabilizerPattern.TWO_WEIGHT_RIGHT

        match self.initial_value:
            case InitialValue.Plus:
                satisfied_x = True
                satisfied_z = False
            case InitialValue.Zero:
                satisfied_x = False
                satisfied_z = True

        m: SurfaceSyndromeMeasurement
        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2

                # Weight-two syndrome measurements:
                if i == 0 and j % 2 == 0 and j < distance - 1:
                    m = SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y - 1), TWO_WEIGHT_DOWN, satisfied_x)
                    self.surface_syndrome_measurements[(x + 1, y - 1)] = m
                if i == distance - 1 and j % 2 == 1:
                    m = SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y + 1), TWO_WEIGHT_UP, satisfied_x)
                    self.surface_syndrome_measurements[(x + 1, y + 1)] = m
                if j == 0 and i % 2 == 1:
                    m = SurfaceZSyndromeMeasurement(self.circuit, (x - 1, y + 1), TWO_WEIGHT_RIGHT, satisfied_z)
                    self.surface_syndrome_measurements[(x - 1, y + 1)] = m
                if j == distance - 1 and i % 2 == 0 and i < distance - 1:
                    m = SurfaceZSyndromeMeasurement(self.circuit, (x + 1, y + 1), TWO_WEIGHT_LEFT, satisfied_z)
                    self.surface_syndrome_measurements[(x + 1, y + 1)] = m

                # Weight-four syndrome measurements:
                if i < distance - 1 and j < distance - 1:
                    if (i + j) % 2 == 0:
                        m = SurfaceZSyndromeMeasurement(self.circuit, (x + 1, y + 1), FOUR_WEIGHT, satisfied_z)
                    else:
                        m = SurfaceXSyndromeMeasurement(self.circuit, (x + 1, y + 1), FOUR_WEIGHT, satisfied_x)
                    self.surface_syndrome_measurements[(x + 1, y + 1)] = m

        if self.full_post_selection:
            for m in self.surface_syndrome_measurements.values():
                m.set_post_selection(True)
    
    def build_circuit(self):
        depth_for_surface_code_syndrome_measurement = 6
        distance = self.distance
        circuit = self.circuit

        offset_x = self.offset_x
        offset_y = self.offset_y

        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2
                match self.initial_value:
                    case InitialValue.Plus:
                        circuit.place_reset_x((x, y))
                    case InitialValue.Zero:
                        circuit.place_reset_z((x, y))

        for i in range(depth_for_surface_code_syndrome_measurement * distance):
            for m in self.surface_syndrome_measurements.values():
                m.run()
            circuit.place_tick()

        match self.initial_value:
            case InitialValue.Plus:
                self._perform_destructive_x_measurement()
                measurements = self.surface_x_measurements
                xs = [measurements[(offset_x, offset_y + i * 2)] for i in range(distance)]
                circuit.place_observable_include(xs, ObservableIdentifier(0))
            case InitialValue.Zero:
                self._perform_destructive_z_measurement()
                measurements = self.surface_z_measurements
                zs = [measurements[(offset_x + j * 2, offset_y + 2)] for j in range(distance)]
                circuit.place_observable_include(zs, ObservableIdentifier(0))

    def _perform_destructive_x_measurement(self) -> None:
        distance = self.distance
        circuit = self.circuit
        offset_x = self.offset_x
        offset_y = self.offset_y
        post_selection = self.full_post_selection

        last_measurements: dict[tuple[int, int], MeasurementIdentifier | None] = {
            pos: m.last_measurement for (pos, m) in self.surface_syndrome_measurements.items()
        }
        measurements: dict[tuple[int, int], MeasurementIdentifier] = self.surface_x_measurements
        assert len(measurements) == 0

        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2
                id = circuit.place_measurement_x((x, y))
                measurements[(x, y)] = id

        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2

                # Weight-two syndrome measurements:
                if i == 0 and j % 2 == 0 and j < distance - 1:
                    last = last_measurements[(x + 1, y - 1)]
                    assert last is not None
                    circuit.place_detector([measurements[(x, y)], measurements[(x + 2, y)], last], post_selection)
                if i == distance - 1 and j % 2 == 1:
                    last = last_measurements[(x + 1, y + 1)]
                    assert last is not None
                    circuit.place_detector([measurements[(x, y)], measurements[(x + 2, y)], last], post_selection)

                # Weight-four syndrome measurements:
                if i < distance - 1 and j < distance - 1 and (i + j) % 2 == 1:
                    last = last_measurements[(x + 1, y + 1)]
                    assert last is not None
                    circuit.place_detector([
                        measurements[(x, y)],
                        measurements[(x + 2, y)],
                        measurements[(x, y + 2)],
                        measurements[(x + 2, y + 2)],
                        last
                    ], post_selection=post_selection)

    def _perform_destructive_z_measurement(self) -> None:
        distance = self.distance
        circuit = self.circuit
        offset_x = self.offset_x
        offset_y = self.offset_y
        post_selection = self.full_post_selection

        last_measurements: dict[tuple[int, int], MeasurementIdentifier | None] = {
            pos: m.last_measurement for (pos, m) in self.surface_syndrome_measurements.items()
        }
        measurements: dict[tuple[int, int], MeasurementIdentifier] = self.surface_z_measurements
        assert len(measurements) == 0

        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2
                id = circuit.place_measurement_z((x, y))
                measurements[(x, y)] = id

        for i in range(distance):
            for j in range(distance):
                x = offset_x + j * 2
                y = offset_y + i * 2

                # Weight-two syndrome measurements:
                if j == 0 and i % 2 == 1:
                    last = last_measurements[(x - 1, y + 1)]
                    assert last is not None
                    circuit.place_detector([measurements[(x, y)], measurements[(x, y + 2)], last], post_selection)
                if j == distance - 1 and i % 2 == 0 and i < distance - 1:
                    last = last_measurements[(x + 1, y + 1)]
                    assert last is not None
                    circuit.place_detector([measurements[(x, y)], measurements[(x, y + 2)], last], post_selection)

                # Weight-four syndrome measurements:
                if i < distance - 1 and j < distance - 1 and (i + j) % 2 == 0:
                    last = last_measurements[(x + 1, y + 1)]
                    assert last is not None
                    circuit.place_detector([
                        measurements[(x, y)],
                        measurements[(x + 2, y)],
                        measurements[(x, y + 2)],
                        measurements[(x + 2, y + 2)],
                        last
                    ], post_selection)


def main() -> None:
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--max-shots', type=int, default=1000)
    parser.add_argument('--max-errors', type=int, default=100)
    parser.add_argument('--error-probability', type=float, default=0)
    parser.add_argument('--parallelism', type=int, default=1)
    parser.add_argument('--distance', type=int, default=3)
    parser.add_argument('--initial-value', choices=['+', '0'], default='+')
    parser.add_argument('--full-post-selection', action='store_true')

    args = parser.parse_args()

    print('  max-shots = {}'.format(args.max_shots))
    print('  max-errors = {}'.format(args.max_errors))
    print('  error-probability = {}'.format(args.error_probability))
    print('  parallelism = {}'.format(args.parallelism))
    print('  distance = {}'.format(args.distance))
    print('  initial-value = {}'.format(args.initial_value))
    print('  full-post-selection = {}'.format(args.full_post_selection))

    max_shots: int = args.max_shots
    max_errors: int = args.max_errors
    error_probability: float = args.error_probability
    parallelism: int = args.parallelism
    distance: int = args.distance
    match args.initial_value:
        case '+':
            initial_value = InitialValue.Plus
        case '0':
            initial_value = InitialValue.Zero
        case _:
            assert False
    full_post_selection: bool = args.full_post_selection

    mapping = QubitMapping(30, 30)
    r = SurfaceCodeInitialization(mapping, distance, initial_value, error_probability, full_post_selection)
    circuit = r.circuit
    stim_circuit = circuit.circuit
    r.build_circuit()

    # Note that Sinter has a bug regarding `postselection_mask`.
    # See https://github.com/quantumlib/Stim/issues/887. We use Sinter 1.13 to avoid the issue.
    postselection_mask = np.zeros(stim_circuit.num_detectors, dtype='uint8')
    for id in circuit.detectors_for_post_selection:
        postselection_mask[id.id] = 1
    postselection_mask = np.packbits(postselection_mask, bitorder='little')

    task = sinter.Task(circuit=stim_circuit, postselection_mask=postselection_mask)
    collected_stats: list[sinter.TaskStats] = sinter.collect(
        num_workers=parallelism,
        tasks=[task],
        decoders=['pymatching'],
        max_shots=max_shots,
        max_errors=max_errors,
        max_batch_size=1_000_000,
    )
    num_wrong = collected_stats[0].errors
    num_discarded = collected_stats[0].discards
    num_valid = collected_stats[0].shots - num_wrong - num_discarded

    print('VALID = {}, WRONG = {}, DISCARDED = {}'.format(num_valid, num_wrong, num_discarded))
    print('WRONG / VALID = {:.3e}'.format(num_wrong / num_valid))
    print('(VALID + WRONG) / SHOTS = {:.3f}'.format(
        (num_valid + num_wrong) / (num_valid + num_wrong + num_discarded)))


if __name__ == '__main__':
    main()
