import numpy as np
import pytest

from masserstein import Spectrum, estimate_proportions
from masserstein.deconv_simplex import (
    dualdeconv2,
    dualdeconv2_alternative,
    dualdeconv4,
    dualdeconv4_with_costs,
    intensity_generator,
)


def mixture(components, proportions):
    """Sum normalized component spectra with the given proportions, then normalize."""
    mix = components[0] * proportions[0]
    for comp, p in zip(components[1:], proportions[1:]):
        mix = mix + comp * p
    mix.normalize()
    return mix


class TestIntensityGenerator:
    def test_intensities_are_placed_on_the_axis(self):
        confs = [(1.0, 0.5), (3.0, 0.5)]
        axis = [1.0, 2.0, 3.0]
        assert list(intensity_generator(confs, axis)) == [0.5, 0.0, 0.5]

    def test_axis_positions_before_the_first_peak_are_zero(self):
        assert list(intensity_generator([(3.0, 1.0)], [1.0, 2.0, 3.0])) == [0.0, 0.0, 1.0]

    def test_axis_positions_after_the_last_peak_are_zero(self):
        assert list(intensity_generator([(1.0, 1.0)], [1.0, 2.0, 3.0])) == [1.0, 0.0, 0.0]

    def test_output_length_always_matches_the_axis(self):
        confs = [(1.0, 0.3), (2.0, 0.7)]
        axis = list(np.arange(0.0, 5.0, 0.5))
        assert len(list(intensity_generator(confs, axis))) == len(axis)

    def test_peaks_outside_the_axis_are_dropped(self):
        assert list(intensity_generator([(0.0, 1.0), (2.0, 1.0)], [2.0])) == [1.0]

    def test_empty_spectrum_yields_zeros(self):
        assert list(intensity_generator([], [1.0, 2.0])) == [0.0, 0.0]


class TestDualdeconv2:
    def test_recovers_a_pure_component(self, ethanol, glucose):
        res = dualdeconv2(ethanol, [ethanol, glucose], penalty=0.1)
        assert res["probs"] == pytest.approx([1.0, 0.0], abs=1e-6)
        assert sum(res["trash"]) == pytest.approx(0.0, abs=1e-6)

    def test_recovers_a_known_mixture(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        res = dualdeconv2(mix, [ethanol, glucose], penalty=0.1)
        assert res["probs"] == pytest.approx([0.7, 0.3], abs=1e-6)

    def test_solver_reaches_optimality(self, ethanol, glucose):
        res = dualdeconv2(mixture([ethanol, glucose], [0.5, 0.5]), [ethanol, glucose], penalty=0.1)
        assert res["status"] == 1  # pulp.LpStatusOptimal

    def test_signal_and_noise_sum_to_one(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.6, 0.4])
        res = dualdeconv2(mix, [ethanol, glucose], penalty=0.1)
        assert sum(res["probs"]) + sum(res["trash"]) == pytest.approx(1.0, abs=1e-6)

    def test_global_mass_axis_is_the_sorted_union_of_all_masses(self, ethanol, glucose):
        res = dualdeconv2(ethanol, [ethanol, glucose], penalty=0.1)
        axis = res["global_mass_axis"]
        expected = sorted({mz for s in (ethanol, ethanol, glucose) for mz, _ in s.confs})
        assert axis == pytest.approx(expected)
        assert len(res["trash"]) == len(axis)

    def test_unexplained_signal_becomes_noise(self, ethanol, glucose):
        # Glucose is in the spectrum but not in the query, so its signal is noise.
        mix = mixture([ethanol, glucose], [0.8, 0.2])
        res = dualdeconv2(mix, [ethanol], penalty=0.1)
        assert res["probs"][0] == pytest.approx(0.8, abs=1e-3)
        assert sum(res["trash"]) == pytest.approx(0.2, abs=1e-3)

    def test_a_high_penalty_discourages_denoising(self, ethanol, glucose):
        # A peak far from any reference: with a small penalty it is cheaper to
        # discard it, with a large one the solver prefers to transport it.
        contaminated = mixture([ethanol, Spectrum(confs=[(60.0, 1.0)], label="junk")], [0.9, 0.1])
        cheap = dualdeconv2(contaminated, [ethanol], penalty=0.1)
        expensive = dualdeconv2(contaminated, [ethanol], penalty=100.0)
        assert sum(cheap["trash"]) > sum(expensive["trash"])

    def test_unnormalized_experimental_spectrum_is_rejected(self, ethanol):
        bad = ethanol * 2
        with pytest.raises(AssertionError, match="Experimental spectrum not normalized"):
            dualdeconv2(bad, [ethanol], penalty=0.1)

    def test_unnormalized_theoretical_spectrum_is_rejected(self, ethanol):
        with pytest.raises(AssertionError, match="Theoretical spectrum 0 not normalized"):
            dualdeconv2(ethanol, [ethanol * 2], penalty=0.1)

    def test_alternative_formulation_agrees(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        primal = dualdeconv2(mix, [ethanol, glucose], penalty=0.1)
        alternative = dualdeconv2_alternative(mix, [ethanol, glucose], penalty=0.1)
        assert alternative["probs"] == pytest.approx(primal["probs"], abs=1e-6)


class TestDualdeconv4:
    def test_recovers_a_known_mixture(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        res = dualdeconv4(mix, [ethanol, glucose], penalty=0.1, penalty_th=0.1)
        assert res["probs"] == pytest.approx([0.7, 0.3], abs=1e-6)

    def test_reports_noise_in_the_theoretical_spectra(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.5, 0.5])
        res = dualdeconv4(mix, [ethanol, glucose], penalty=0.1, penalty_th=0.1)
        assert "noise_in_theoretical" in res
        assert "theoretical_trash" in res
        assert res["noise_in_theoretical"] == pytest.approx(0.0, abs=1e-6)

    def test_zero_costs_reproduce_the_cost_free_solution(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        plain = dualdeconv4(mix, [ethanol, glucose], penalty=0.1, penalty_th=0.1)
        with_costs = dualdeconv4_with_costs(
            mix, [ethanol, glucose], costs=[0.0, 0.0], penalty=0.1, penalty_th=0.1
        )
        assert with_costs["probs"] == pytest.approx(plain["probs"], abs=1e-6)

    def test_a_prohibitive_cost_suppresses_a_component(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        res = dualdeconv4_with_costs(
            mix, [ethanol, glucose], costs=[0.0, 1e3], penalty=0.1, penalty_th=0.1
        )
        assert res["probs"][1] == pytest.approx(0.0, abs=1e-6)


class TestEstimateProportions:
    def test_recovers_a_pure_component(self, ethanol, glucose):
        res = estimate_proportions(ethanol, [ethanol, glucose], MTD=0.1, progress=False)
        assert res["proportions"] == pytest.approx([1.0, 0.0], abs=1e-6)

    def test_recovers_a_two_component_mixture(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        res = estimate_proportions(mix, [ethanol, glucose], MTD=0.1, progress=False)
        assert res["proportions"] == pytest.approx([0.7, 0.3], abs=1e-6)

    @pytest.mark.parametrize("true", [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.99, 0.01]])
    def test_recovers_a_range_of_proportions(self, ethanol, glucose, true):
        mix = mixture([ethanol, glucose], true)
        res = estimate_proportions(mix, [ethanol, glucose], MTD=0.1, progress=False)
        assert res["proportions"] == pytest.approx(true, abs=1e-4)

    def test_recovers_a_three_component_mixture(self, ethanol, glucose):
        caffeine = Spectrum("C8H10N4O2", threshold=0.001, label="caffeine")
        caffeine.normalize()
        true = [0.2, 0.5, 0.3]
        mix = mixture([ethanol, glucose, caffeine], true)
        res = estimate_proportions(mix, [ethanol, glucose, caffeine], MTD=0.1, progress=False)
        assert res["proportions"] == pytest.approx(true, abs=1e-4)

    def test_returned_keys_without_theoretical_noise(self, ethanol, glucose):
        res = estimate_proportions(ethanol, [ethanol, glucose], MTD=0.1, progress=False)
        assert set(res) == {"proportions", "noise", "global_mass_axis"}
        assert len(res["noise"]) == len(res["global_mass_axis"])

    def test_returned_keys_with_theoretical_noise(self, ethanol, glucose):
        res = estimate_proportions(
            ethanol, [ethanol, glucose], MTD=0.1, MTD_th=0.1, progress=False
        )
        assert set(res) == {
            "proportions",
            "noise",
            "noise_in_theoretical",
            "proportion_of_noise_in_theoretical",
            "global_mass_axis",
        }
        assert len(res["noise_in_theoretical"]) == len(res["global_mass_axis"])

    def test_signal_and_noise_sum_to_one(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.6, 0.4])
        res = estimate_proportions(mix, [ethanol, glucose], MTD=0.1, progress=False)
        assert sum(res["proportions"]) + sum(res["noise"]) == pytest.approx(1.0, abs=1e-4)

    def test_chemical_noise_is_pushed_into_the_noise_component(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        mix.add_chemical_noise(nb_of_noise_peaks=30, noise_fraction=0.2)
        mix.normalize()
        res = estimate_proportions(mix, [ethanol, glucose], MTD=0.05, progress=False)
        # Roughly 20% of the signal is noise; the rest keeps the 7:3 ratio.
        assert sum(res["noise"]) == pytest.approx(0.2, abs=0.05)
        p = res["proportions"]
        assert p[0] / (p[0] + p[1]) == pytest.approx(0.7, abs=0.05)

    def test_absent_component_gets_zero(self, ethanol, glucose):
        caffeine = Spectrum("C8H10N4O2", threshold=0.001, label="caffeine")
        caffeine.normalize()
        mix = mixture([ethanol, glucose], [0.5, 0.5])
        res = estimate_proportions(mix, [ethanol, glucose, caffeine], MTD=0.1, progress=False)
        assert res["proportions"][2] == pytest.approx(0.0, abs=1e-6)

    def test_zero_costs_match_the_cost_free_run(self, ethanol, glucose):
        mix = mixture([ethanol, glucose], [0.7, 0.3])
        plain = estimate_proportions(mix, [ethanol, glucose], MTD=0.1, MTD_th=0.1, progress=False)
        costed = estimate_proportions(
            mix, [ethanol, glucose], costs=[0.0, 0.0], MTD=0.1, MTD_th=0.1, progress=False
        )
        assert costed["proportions"] == pytest.approx(plain["proportions"], abs=1e-6)

    def test_mismatched_costs_length_is_rejected(self, ethanol, glucose):
        with pytest.raises(AssertionError, match="len\\(costs\\)"):
            estimate_proportions(
                ethanol, [ethanol, glucose], costs=[1.0], MTD=0.1, MTD_th=0.1, progress=False
            )

    def test_unnormalized_experimental_spectrum_is_rejected(self, ethanol):
        with pytest.raises(AssertionError, match="not normalized"):
            estimate_proportions(ethanol * 2, [ethanol], MTD=0.1, progress=False)

    def test_unnormalized_query_spectrum_is_rejected(self, ethanol):
        with pytest.raises(AssertionError, match="Theoretical spectrum 0 is not normalized"):
            estimate_proportions(ethanol, [ethanol * 2], MTD=0.1, progress=False)

    def test_negative_intensities_are_rejected(self, ethanol):
        broken = ethanol.copy()
        broken.confs = [(broken.confs[0][0], -0.5)] + broken.confs[1:]
        broken.normalize()
        with pytest.raises(ValueError, match="negative intensities"):
            estimate_proportions(broken, [ethanol], MTD=0.1, progress=False)

    def test_non_spectrum_input_is_rejected(self, ethanol):
        with pytest.raises(TypeError, match="confs"):
            estimate_proportions(object(), [ethanol], MTD=0.1, progress=False)

    def test_everything_filtered_out_is_an_error(self, ethanol):
        # MMD forces the query mode to sit near an experimental peak; a query
        # 1000 Da away from the spectrum cannot match anything.
        far_away = Spectrum("C60", threshold=0.01, label="far")
        far_away.normalize()
        with pytest.raises(AssertionError, match="No valid"):
            estimate_proportions(ethanol, [far_away], MTD=0.1, MMD=0.1, progress=False)

    def test_mdc_filters_out_components_without_matching_current(self, ethanol, glucose):
        # Glucose has no signal in a pure ethanol spectrum, so it is filtered
        # out early rather than being assigned a proportion.
        res = estimate_proportions(
            ethanol, [ethanol, glucose], MTD=0.1, MDC=1e-3, progress=False
        )
        assert res["proportions"] == pytest.approx([1.0, 0.0], abs=1e-6)

    def test_far_apart_components_are_deconvolved_in_separate_chunks(self, ethanol, glucose):
        # Ethanol (m/z ~46) and glucose (m/z ~180) are far apart, which forces
        # the chunking code path; results must still be correct.
        mix = mixture([ethanol, glucose], [0.4, 0.6])
        res = estimate_proportions(mix, [ethanol, glucose], MTD=0.01, progress=False)
        assert res["proportions"] == pytest.approx([0.4, 0.6], abs=1e-4)

    def test_shifted_spectrum_is_still_explained_within_mtd(self, ethanol):
        # A small calibration error is absorbed as long as it stays below MTD.
        shifted = Spectrum(confs=[(mz + 0.01, i) for mz, i in ethanol.confs], label="shifted")
        res = estimate_proportions(shifted, [ethanol], MTD=0.1, progress=False)
        assert res["proportions"][0] == pytest.approx(1.0, abs=1e-3)

    def test_shift_beyond_mtd_is_treated_as_noise(self, ethanol):
        shifted = Spectrum(confs=[(mz + 0.5, i) for mz, i in ethanol.confs], label="shifted")
        res = estimate_proportions(shifted, [ethanol], MTD=0.05, MDC=0.0, progress=False)
        assert res["proportions"][0] == pytest.approx(0.0, abs=1e-3)
        assert sum(res["noise"]) == pytest.approx(1.0, abs=1e-3)

    def test_verbose_mode_runs(self, ethanol, glucose, capsys):
        mix = mixture([ethanol, glucose], [0.5, 0.5])
        estimate_proportions(mix, [ethanol, glucose], MTD=0.1, verbose=True, progress=False)
        assert "Envelope bounds" in capsys.readouterr().out
