import numpy as np
import pymatching
import stim


def calculate_weight(syndrome: np.ndarray, matcher: pymatching.Matching) -> float:
    all_edges = matcher.edges()
    weight = 0
    for [x, y] in matcher.decode_to_edges_array(syndrome):
        y = None if y == -1 else y
        [w] = [attr['weight'] for (u, v, attr) in all_edges if (u, v) == (x, y) or (u, v) == (y, x)]
        weight += w
    return weight


def main() -> None:
    num_shots = 1000
    circuit: stim.Circuit
    with open('surface-code.stim') as f:
        circuit = stim.Circuit(f.read())
    dem = circuit.detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler()

    detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)

    for shot in range(num_shots):
        syndrome = detection_events[shot]
        weight = calculate_weight(syndrome, matcher)

        syndrome[-1] = 1 - syndrome[-1]
        complementary_weight = calculate_weight(syndrome, matcher)
        syndrome[-1] = 1 - syndrome[-1]

        gap = complementary_weight - weight
        print('weight = {:8.3f}, complementary_weight = {:8.3f}, gap = {:8.3f}'.format(
            weight, complementary_weight, gap))


if __name__ == '__main__':
    main()
