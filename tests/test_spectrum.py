import math

import numpy as np
import pytest

from masserstein import Spectrum
from masserstein.base_spectrum import BaseSpectrum

from helpers import gaussian_profile

PROTON_MASS = 1.00782503207


class TestFormulaSimulation:
    def test_simulating_from_a_formula_produces_peaks(self):
        s = Spectrum("C2H5OH", threshold=0.001)
        assert len(s) > 1
        assert s.formula == "C2H5OH"
        assert s.label == "C2H5OH"

    def test_probabilities_of_a_simulated_envelope_sum_to_at_most_one(self):
        s = Spectrum("C6H12O6", threshold=0.001)
        total = sum(i for _, i in s.confs)
        assert 0.9 < total <= 1.0 + 1e-9

    def test_monoisotopic_peak_is_the_lightest_and_most_abundant(self):
        s = Spectrum("C2H5OH", threshold=0.001)
        assert s.confs[0][0] == pytest.approx(46.0418648, abs=1e-4)
        assert s.get_modal_peak() == s.confs[0]

    def test_lower_threshold_yields_more_peaks(self):
        coarse = Spectrum("C6H12O6", threshold=0.01)
        fine = Spectrum("C6H12O6", threshold=1e-6)
        assert len(fine) > len(coarse)

    def test_total_prob_covers_the_requested_probability(self):
        s = Spectrum("C6H12O6", total_prob=0.999)
        assert sum(i for _, i in s.confs) >= 0.999

    def test_total_prob_overrides_threshold(self):
        # A threshold of 0.5 alone would keep a single peak; total_prob wins.
        s = Spectrum("C6H12O6", threshold=0.5, total_prob=0.99)
        assert len(s) > 1

    def test_repeated_elements_in_a_formula_are_summed(self):
        # C2H5OH and C2H6O describe the same molecule.
        a = Spectrum("C2H5OH", threshold=1e-6)
        b = Spectrum("C2H6O", threshold=1e-6)
        assert [round(mz, 6) for mz, _ in a.confs] == [round(mz, 6) for mz, _ in b.confs]

    def test_multi_letter_elements_are_parsed(self):
        s = Spectrum("Cl2", threshold=0.01)
        # Chlorine has two abundant isotopes, so Cl2 has at least three peaks.
        assert len(s) >= 3
        assert s.confs[0][0] == pytest.approx(69.9377, abs=1e-3)

    def test_confs_from_formula_is_usable_standalone(self):
        confs = Spectrum.confs_from_formula("H2O", threshold=0.01)
        assert confs == Spectrum("H2O", threshold=0.01).confs


class TestChargeAndAdduct:
    def test_charge_divides_the_mz_axis(self):
        neutral = Spectrum("C6H12O6", threshold=0.01, charge=1)
        doubly = Spectrum("C6H12O6", threshold=0.01, charge=2)
        assert doubly.confs[0][0] == pytest.approx(neutral.confs[0][0] / 2)
        assert doubly.charge == 2

    def test_negative_charge_uses_absolute_value_for_mz(self):
        pos = Spectrum("C6H12O6", threshold=0.01, charge=2)
        neg = Spectrum("C6H12O6", threshold=0.01, charge=-2)
        assert neg.confs[0][0] == pytest.approx(pos.confs[0][0])

    def test_adduct_adds_charge_many_atoms(self):
        plain = Spectrum("H2O", threshold=0.01)
        protonated = Spectrum("H2O", threshold=0.01, adduct="H", charge=1)
        assert protonated.confs[0][0] == pytest.approx(plain.confs[0][0] + PROTON_MASS, abs=1e-4)

    def test_adduct_scales_with_charge(self):
        plain = Spectrum("H2O", threshold=0.01)
        doubly = Spectrum("H2O", threshold=0.01, adduct="H", charge=2)
        expected = (plain.confs[0][0] + 2 * PROTON_MASS) / 2
        assert doubly.confs[0][0] == pytest.approx(expected, abs=1e-4)

    def test_sodium_adduct(self):
        sodiated = Spectrum("H2O", threshold=0.01, adduct="Na")
        plain = Spectrum("H2O", threshold=0.01)
        assert sodiated.confs[0][0] - plain.confs[0][0] == pytest.approx(22.9898, abs=1e-3)


class TestConfsConstruction:
    def test_confs_can_be_supplied_directly(self):
        s = Spectrum(confs=[(2.0, 1.0), (1.0, 1.0)], label="manual")
        assert s.confs == [(1.0, 1.0), (2.0, 1.0)]
        assert s.label == "manual"

    def test_formula_and_confs_together_are_rejected(self):
        with pytest.raises(ValueError, match="Formula and confs cannot be set at the same time"):
            Spectrum("H2O", confs=[(1.0, 1.0)])

    def test_empty_spectrum_can_be_created(self):
        s = Spectrum("", empty=True)
        assert s.confs == []
        assert s.empty

    def test_label_defaults_to_formula(self):
        assert Spectrum("H2O").label == "H2O"

    def test_explicit_label_wins_over_formula(self):
        assert Spectrum("H2O", label="water").label == "water"

    def test_spectrum_is_a_base_spectrum(self):
        assert isinstance(Spectrum("H2O"), BaseSpectrum)

    def test_arithmetic_returns_a_spectrum(self):
        a = Spectrum(confs=[(1.0, 1.0)])
        b = Spectrum(confs=[(2.0, 1.0)])
        assert isinstance(a + b, Spectrum)
        assert isinstance(2 * a, Spectrum)


class TestFasta:
    def test_new_from_fasta_matches_the_peptide_formula(self):
        from masserstein.peptides import get_protein_formula

        peptide = "GAG"
        s = Spectrum.new_from_fasta(peptide, threshold=0.01)
        expected = Spectrum(get_protein_formula(peptide), threshold=0.01)
        assert s.confs == expected.confs


class TestMassStatistics:
    def test_average_mass_of_a_single_peak(self):
        s = Spectrum(confs=[(10.0, 0.4)])
        assert s.average_mass() == pytest.approx(10.0)

    def test_average_mass_is_intensity_weighted(self):
        s = Spectrum(confs=[(10.0, 0.25), (20.0, 0.75)])
        assert s.average_mass() == pytest.approx(17.5)

    def test_average_mass_is_scale_invariant(self):
        s = Spectrum(confs=[(10.0, 1.0), (20.0, 3.0)])
        scaled = s * 7
        assert scaled.average_mass() == pytest.approx(s.average_mass())

    def test_average_mass_lies_between_the_extreme_peaks(self):
        s = Spectrum("C6H12O6", threshold=1e-6)
        s.normalize()
        assert s.confs[0][0] < s.average_mass() < s.confs[-1][0]


class TestBinning:
    def test_bin_to_nominal_rounds_to_integers(self):
        s = Spectrum("C2H5OH", threshold=1e-6)
        first = s.confs[0][0]
        s.bin_to_nominal()
        # Masses become the monoisotopic mass plus integer offsets.
        offsets = [mz - first for mz, _ in s.confs]
        assert all(off == pytest.approx(round(off)) for off in offsets)

    def test_bin_to_nominal_preserves_total_intensity(self):
        s = Spectrum("C6H12O6", threshold=1e-6)
        s.normalize()
        before = sum(i for _, i in s.confs)
        s.bin_to_nominal()
        assert sum(i for _, i in s.confs) == pytest.approx(before)

    def test_bin_to_nominal_merges_peaks(self):
        s = Spectrum("C6H12O6", threshold=1e-6)
        before = len(s)
        s.bin_to_nominal()
        assert len(s) < before

    def test_bin_to_nominal_accounts_for_charge(self):
        s = Spectrum("C6H12O6", threshold=1e-6, charge=2)
        first = s.confs[0][0]
        s.bin_to_nominal()
        # On the m/z axis, a charge of 2 puts isotopic peaks half a unit apart.
        offsets = [(mz - first) * 2 for mz, _ in s.confs]
        assert all(off == pytest.approx(round(off)) for off in offsets)

    def test_coarse_bin_rounds_mz(self):
        s = Spectrum(confs=[(1.234, 0.5), (1.239, 0.5)])
        s.coarse_bin(nb_of_digits=2)
        assert s.confs == [(1.23, 0.5), (1.24, 0.5)]

    def test_coarse_bin_merges_collapsed_peaks(self):
        s = Spectrum(confs=[(1.234, 0.5), (1.236, 0.25)])
        s.coarse_bin(nb_of_digits=1)
        assert s.confs == [(1.2, 0.75)]


class TestDistortion:
    def test_distort_mz_shifts_by_the_requested_mean(self):
        s = Spectrum(confs=[(float(i), 1.0) for i in range(500)])
        original = [mz for mz, _ in s.confs]
        shift = s.distort_mz(mean=0.5, sd=0.01)
        assert len(shift) == 500
        moved = [mz for mz, _ in s.confs]
        assert np.mean(np.array(moved) - np.array(original)) == pytest.approx(0.5, abs=0.01)

    def test_distort_mz_keeps_intensities(self):
        s = Spectrum(confs=[(float(i), 0.25) for i in range(4)])
        s.distort_mz(0.0, 0.01)
        assert all(i == 0.25 for _, i in s.confs)

    def test_distort_mz_leaves_confs_sorted(self):
        s = Spectrum(confs=[(float(i), 1.0) for i in range(100)])
        s.distort_mz(0.0, 5.0)
        assert [mz for mz, _ in s.confs] == sorted(mz for mz, _ in s.confs)

    def test_distort_intensity_requires_normalization(self):
        s = Spectrum(confs=[(1.0, 5.0)])
        with pytest.raises(AssertionError, match="normalized"):
            s.distort_intensity(N=100, gain=1.0, sd=0.1)

    def test_distort_intensity_scales_signal_by_n_times_gain(self):
        s = Spectrum("C6H12O6", threshold=1e-4)
        s.normalize()
        s.distort_intensity(N=10**6, gain=2.0, sd=0.1)
        assert sum(i for _, i in s.confs) == pytest.approx(2 * 10**6, rel=0.01)

    def test_distort_intensity_never_returns_negative_intensities(self):
        s = Spectrum("C6H12O6", threshold=1e-4)
        s.normalize()
        s.distort_intensity(N=10, gain=1.0, sd=10.0)
        assert all(i >= 0 for _, i in s.confs)

    def test_distort_intensity_returns_the_deviations(self):
        s = Spectrum("C2H5OH", threshold=1e-4)
        s.normalize()
        deviations = s.distort_intensity(N=1000, gain=1.0, sd=0.1)
        assert len(deviations) == len(s)

    def test_sample_multinomial_requires_normalization(self):
        ref = Spectrum(confs=[(1.0, 3.0)], label="ref")
        with pytest.raises(AssertionError, match="normalized"):
            Spectrum.sample_multinomial(ref, N=10, gain=1.0, sd=0.1)

    def test_sample_multinomial_preserves_the_mass_axis(self):
        ref = Spectrum("C2H5OH", threshold=1e-4)
        ref.normalize()
        sampled = Spectrum.sample_multinomial(ref, N=10000, gain=1.0, sd=0.01)
        assert [mz for mz, _ in sampled.confs] == [mz for mz, _ in ref.confs]
        assert sampled.label == "Sampled " + ref.label

    def test_sample_multinomial_approximates_the_reference(self):
        ref = Spectrum("C2H5OH", threshold=1e-4)
        ref.normalize()
        sampled = Spectrum.sample_multinomial(ref, N=10**6, gain=1.0, sd=0.01)
        sampled.normalize()
        assert sampled.WSDistance(ref) == pytest.approx(0.0, abs=1e-3)


class TestSmoothing:
    def test_gaussian_smoothing_conserves_area(self):
        s = Spectrum(confs=[(100.0, 1.0)])
        s.gaussian_smoothing(sd=0.05, new_mz=0.001)
        mz = np.array([m for m, _ in s.confs])
        intensity = np.array([i for _, i in s.confs])
        assert np.trapezoid(intensity, mz) == pytest.approx(1.0, rel=1e-3)

    def test_gaussian_smoothing_centers_the_peak_on_the_original_mass(self):
        s = Spectrum(confs=[(100.0, 1.0)])
        s.gaussian_smoothing(sd=0.05, new_mz=0.001)
        assert s.get_modal_peak()[0] == pytest.approx(100.0, abs=1e-2)

    def test_gaussian_smoothing_accepts_an_explicit_mass_axis(self):
        axis = np.linspace(99.0, 101.0, 201)
        s = Spectrum(confs=[(100.0, 1.0)])
        s.gaussian_smoothing(sd=0.05, new_mz=axis)
        assert [mz for mz, _ in s.confs] == pytest.approx(list(axis))

    def test_gaussian_smoothing_rejects_an_unsorted_axis(self):
        s = Spectrum(confs=[(100.0, 1.0)])
        with pytest.raises(AssertionError, match="sorted"):
            s.gaussian_smoothing(sd=0.05, new_mz=np.array([101.0, 99.0]))

    def test_wider_filter_produces_a_lower_apex(self):
        narrow = Spectrum(confs=[(100.0, 1.0)])
        wide = Spectrum(confs=[(100.0, 1.0)])
        narrow.gaussian_smoothing(sd=0.02, new_mz=0.001)
        wide.gaussian_smoothing(sd=0.10, new_mz=0.001)
        assert wide.get_modal_peak()[1] < narrow.get_modal_peak()[1]

    def test_fuzzify_peaks_conserves_area(self):
        s = Spectrum("C2H5OH", threshold=1e-4)
        s.normalize()
        s.fuzzify_peaks(sd=0.05, step=0.001)
        mz = np.array([m for m, _ in s.confs])
        intensity = np.array([i for _, i in s.confs])
        assert np.trapezoid(intensity, mz) == pytest.approx(1.0, rel=1e-2)

    def test_fuzzify_and_gaussian_smoothing_agree(self):
        # Both truncate the Gaussian at 4*sd but disagree on whether the boundary
        # sample itself is included, so only the interior is compared.
        fuzzy = Spectrum(confs=[(100.0, 1.0)])
        smooth = Spectrum(confs=[(100.0, 1.0)])
        fuzzy.fuzzify_peaks(sd=0.05, step=0.001)
        smooth.gaussian_smoothing(sd=0.05, new_mz=0.001)
        assert len(fuzzy) == len(smooth)
        assert [i for _, i in fuzzy.confs[1:-1]] == pytest.approx(
            [i for _, i in smooth.confs[1:-1]], abs=1e-9
        )

    def test_smoothed_centroid_recovers_the_original_peaks(self):
        # Simulate a profile spectrum, then centroid it back into a peak list.
        s = Spectrum("C2H5OH", threshold=0.01)
        s.normalize()
        original = list(s.confs)
        s.fuzzify_peaks(sd=0.01, step=0.0005)
        centroids, _ = s.centroid(max_width=0.3)
        assert len(centroids) == len(original)
        for (found_mz, _), (true_mz, _) in zip(centroids, original):
            assert found_mz == pytest.approx(true_mz, abs=1e-3)


class TestResample:
    def test_resample_interpolates_within_a_peak(self):
        s = Spectrum(confs=[(1.0, 0.0), (1.01, 1.0), (1.02, 0.0)])
        target = [1.0, 1.005, 1.01, 1.015, 1.02]
        r = s.resample(target, mz_distance_threshold=0.05)
        assert [mz for mz, _ in r.confs] == pytest.approx(target)
        assert r.confs[1][1] == pytest.approx(0.5)
        assert r.confs[2][1] == pytest.approx(1.0)
        assert r.confs[3][1] == pytest.approx(0.5)

    def test_resample_leaves_background_at_zero(self):
        # The two measurements are further apart than the threshold, so the
        # region between them counts as background rather than a peak.
        s = Spectrum(confs=[(1.0, 1.0), (5.0, 1.0)])
        r = s.resample([1.0, 3.0, 5.0], mz_distance_threshold=0.05)
        assert r.confs[1][1] == 0.0

    def test_resample_rejects_an_unsorted_target(self):
        s = Spectrum(confs=[(1.0, 1.0), (2.0, 1.0)])
        with pytest.raises(AssertionError, match="not sorted"):
            s.resample([2.0, 1.0])

    def test_resample_onto_a_profile_spectrum_preserves_shape(self):
        s = Spectrum(confs=gaussian_profile([100.0], [1.0], sd=0.05, step=0.001))
        target = list(np.arange(99.5, 100.5, 0.005))
        r = s.resample(target, mz_distance_threshold=0.01)
        assert r.get_modal_peak()[0] == pytest.approx(100.0, abs=0.01)
        assert r.get_modal_peak()[1] == pytest.approx(1.0, rel=0.01)
