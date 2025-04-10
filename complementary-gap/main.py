import pymatching
import stim


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
        _, weight = matcher.decode(syndrome, return_weight=True)

        syndrome[-1] = not syndrome[-1]
        _, complementary_weight = matcher.decode(syndrome, return_weight=True)
        syndrome[-1] = not syndrome[-1]

        gap = complementary_weight - weight
        print('weight = {:8.3f}, complementary_weight = {:8.3f}, gap = {:8.3f}'.format(
            weight, complementary_weight, gap))


if __name__ == '__main__':
    main()
