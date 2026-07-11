from collections import Counter

import numpy as np
import pytest

from masserstein import Spectrum

from helpers import gaussian_profile

# polymers pulls in pandas/pyteomics/matplotlib, which are optional extras.
pd = pytest.importorskip("pandas")
polymers = pytest.importorskip("masserstein.polymers")


def label_frame(labels_to_proportions):
    """Build the frame the homocoupling measures expect: one row, polymers as columns."""
    return pd.DataFrame(
        {label: [value] for label, value in labels_to_proportions.items()},
        index=["proportion"],
    )


class TestMCounter:
    def test_addition_sums_counts(self):
        a = polymers.MCounter({"C": 2, "H": 4})
        b = polymers.MCounter({"C": 1, "O": 1})
        assert dict(a + b) == {"C": 3, "H": 4, "O": 1}

    def test_addition_returns_an_mcounter(self):
        result = polymers.MCounter({"C": 1}) + polymers.MCounter({"C": 1})
        assert isinstance(result, polymers.MCounter)

    def test_addition_with_a_plain_counter_is_not_supported(self):
        with pytest.raises(TypeError):
            polymers.MCounter({"C": 1}) + Counter({"C": 1})

    def test_multiplication_scales_counts(self):
        assert dict(polymers.MCounter({"C": 2, "H": 4}) * 3) == {"C": 6, "H": 12}

    def test_multiplication_by_zero_gives_zero_counts(self):
        assert dict(polymers.MCounter({"C": 2}) * 0) == {"C": 0}

    def test_multiplication_by_a_non_int_is_rejected(self):
        with pytest.raises(TypeError, match="Non-int factor"):
            polymers.MCounter({"C": 2}) * 1.5

    def test_formula_from_counter(self):
        assert polymers.MCounter({"C": 2, "H": 6}).formula_from_counter() == "C2H6"

    def test_formula_from_counter_is_parseable_by_spectrum(self):
        counter = polymers.MCounter({"C": 2, "H": 6, "O": 1})
        s = Spectrum(formula=counter.formula_from_counter(), threshold=0.01)
        assert s.confs[0][0] == pytest.approx(46.0418648, abs=1e-4)


class TestSetSimilarity:
    def test_jaccard_of_identical_sets(self):
        assert polymers.jaccard(["a", "b"], ["a", "b"]) == 1.0

    def test_jaccard_of_disjoint_sets(self):
        assert polymers.jaccard(["a"], ["b"]) == 0.0

    def test_jaccard_of_partial_overlap(self):
        # intersection 1, union 2 + 2 - 1 = 3
        assert polymers.jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)

    def test_sensitivity_is_recall_over_the_expert_set(self):
        assert polymers.sensitivity(["a", "b", "c"], ["a", "b"]) == 1.0
        assert polymers.sensitivity(["a"], ["a", "b"]) == 0.5

    def test_sensitivity_ignores_extra_predictions(self):
        assert polymers.sensitivity(["a", "x", "y", "z"], ["a"]) == 1.0


class TestSpectrumPreprocessing:
    def test_restrict_keeps_the_requested_window(self):
        s = Spectrum(confs=[(float(i), 1.0) for i in range(10)])
        r = polymers.restrict(s, 3.0, 5.0)
        assert [mz for mz, _ in r.confs] == [3.0, 4.0, 5.0]

    def test_restrict_does_not_modify_the_input(self):
        s = Spectrum(confs=[(float(i), 1.0) for i in range(10)])
        polymers.restrict(s, 3.0, 5.0)
        assert len(s) == 10

    def test_restrict_to_an_empty_window(self):
        s = Spectrum(confs=[(1.0, 1.0), (2.0, 1.0)])
        assert polymers.restrict(s, 10.0, 20.0).confs == []

    def test_correct_baseline_lowers_the_intensities(self):
        s = Spectrum(confs=[(1.0, 1000.0), (2.0, 5000.0)])
        corrected = polymers.correct_baseline(s, base=100.0)
        assert all(i < orig for (_, i), (_, orig) in zip(corrected.confs, s.confs))

    def test_correct_baseline_keeps_intensities_positive(self):
        s = Spectrum(confs=[(float(i), float(i) * 100) for i in range(1, 20)])
        corrected = polymers.correct_baseline(s, base=1000.0)
        assert all(i > 0 for _, i in corrected.confs)

    def test_correct_baseline_preserves_the_mass_axis(self):
        s = Spectrum(confs=[(1.0, 2000.0), (2.0, 3000.0)])
        corrected = polymers.correct_baseline(s)
        assert [mz for mz, _ in corrected.confs] == [1.0, 2.0]

    def test_remove_low_signal_drops_peaks_below_the_relative_threshold(self):
        s = Spectrum(confs=[(1.0, 100.0), (2.0, 0.05), (3.0, 50.0)])
        filtered = polymers.remove_low_signal(s, signal_proportion=0.01)
        # The threshold is 1% of the modal peak (100.0), i.e. 1.0.
        assert [mz for mz, _ in filtered.confs] == [1.0, 3.0]

    def test_remove_low_signal_does_not_modify_the_input(self):
        s = Spectrum(confs=[(1.0, 100.0), (2.0, 0.05)])
        polymers.remove_low_signal(s)
        assert len(s) == 2

    def test_remove_low_signal_keeps_everything_at_a_zero_threshold(self):
        s = Spectrum(confs=[(1.0, 100.0), (2.0, 1.0)])
        assert len(polymers.remove_low_signal(s, signal_proportion=0.0)) == 2

    def test_centroided_returns_a_peak_list(self):
        s = Spectrum(confs=gaussian_profile([100.0, 101.0], [1.0, 2.0], sd=0.02, step=0.001))
        s.label = "profile"
        centroided = polymers.centroided(s, max_width=0.5)
        assert isinstance(centroided, Spectrum)
        assert [round(mz, 1) for mz, _ in centroided.confs] == [100.0, 101.0]
        assert centroided.label == "profile"

    def test_reduce_groups_peaks_into_unit_bins(self):
        s = Spectrum(confs=[(100.0, 1.0), (100.1, 2.0), (101.0, 4.0), (101.2, 1.0)])
        reduced = polymers.reduce(s)
        # Each nominal mass collapses into one peak carrying the summed intensity.
        assert len(reduced) == 2
        assert sum(i for _, i in reduced.confs) == pytest.approx(8.0)

    def test_normalize_helper_returns_a_new_spectrum(self):
        s = Spectrum(confs=[(1.0, 3.0), (2.0, 1.0)])
        normalized = polymers._normalize(s)
        assert sum(i for _, i in normalized.confs) == pytest.approx(1.0)
        assert sum(i for _, i in s.confs) == 4.0  # original untouched


class TestGetPossibleCompounds:
    @pytest.fixture
    def monomers(self):
        heavier = ("BT", polymers.MCounter({"C": 10, "H": 6, "S": 1}))
        lighter = ("TT", polymers.MCounter({"C": 6, "H": 2, "S": 2}))
        end_groups = {
            "H": polymers.MCounter({"H": 1}),
            "Methyl": polymers.MCounter({"C": 1, "H": 3}),
        }
        return heavier, lighter, end_groups

    def test_generates_normalized_labelled_spectra(self, monomers):
        heavier, lighter, end_groups = monomers
        compounds = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=700.0, max_count_diff=1
        )
        assert compounds
        for s in compounds:
            assert isinstance(s, Spectrum)
            assert sum(i for _, i in s.confs) == pytest.approx(1.0)
            assert s.label

    def test_all_compounds_fall_inside_the_requested_window(self, monomers):
        heavier, lighter, end_groups = monomers
        compounds = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=300.0, max_mz=600.0, max_count_diff=1
        )
        assert all(300.0 <= s.confs[0][0] <= 600.0 for s in compounds)

    def test_results_are_sorted_by_monoisotopic_mass(self, monomers):
        heavier, lighter, end_groups = monomers
        compounds = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=700.0, max_count_diff=1
        )
        masses = [s.confs[0][0] for s in compounds]
        assert masses == sorted(masses)

    def test_labels_follow_the_documented_convention(self, monomers):
        heavier, lighter, end_groups = monomers
        compounds = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=700.0, max_count_diff=1
        )
        for s in compounds:
            parts = s.label.split("+")
            assert parts[0].endswith("BT")
            assert parts[1].endswith("TT")
            assert len(parts) in (3, 4)  # "2End" or "End1+End2"

    def test_plain_counters_are_accepted(self, monomers):
        _, lighter, end_groups = monomers
        heavier = ("BT", Counter({"C": 10, "H": 6, "S": 1}))
        compounds = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=700.0, max_count_diff=1
        )
        assert compounds

    def test_a_wider_window_yields_more_compounds(self, monomers):
        heavier, lighter, end_groups = monomers
        narrow = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=400.0, max_count_diff=1
        )
        wide = polymers.get_possible_compounds(
            heavier, lighter, end_groups, min_mz=200.0, max_mz=900.0, max_count_diff=1
        )
        assert len(wide) > len(narrow)

    def test_adducts_are_appended_to_the_label(self, monomers):
        heavier, lighter, end_groups = monomers
        compounds = polymers.get_possible_compounds(
            heavier,
            lighter,
            end_groups,
            min_mz=200.0,
            max_mz=700.0,
            max_count_diff=1,
            adducts={"Na": polymers.MCounter({"Na": 1})},
        )
        assert compounds
        assert all(s.label.endswith("+Na") for s in compounds)


class TestGenerateCosts:
    def test_costs_follow_the_end_group_labels(self):
        spectra = [
            Spectrum(confs=[(1.0, 1.0)], label="1BT+1TT+2H"),
            Spectrum(confs=[(1.0, 1.0)], label="1BT+1TT+H+Methyl"),
        ]
        costs = polymers.generate_costs_by_end_group(spectra, {"2H": 0.5, "H+Methyl": 1.25})
        assert costs == [0.5, 1.25]

    def test_unlabelled_spectra_are_rejected(self):
        spectra = [Spectrum(confs=[(1.0, 1.0)])]  # label defaults to ""
        with pytest.raises(Exception, match="without label"):
            polymers.generate_costs_by_end_group(spectra, {})

    def test_labels_off_convention_get_zero_cost_and_a_warning(self):
        spectra = [Spectrum(confs=[(1.0, 1.0)], label="nonsense")]
        with pytest.warns(UserWarning, match="naming convention"):
            costs = polymers.generate_costs_by_end_group(spectra, {})
        assert costs == [0]


class TestHomocouplingMeasures:
    @pytest.fixture
    def frame(self):
        # 2:2 is perfectly alternating; 3:1 carries two extra heavier monomers.
        return label_frame({"2BT+2TT+2H": 0.75, "3BT+1TT+2H": 0.25})

    def test_homocoupling_frequency_reports_count_differences(self, frame):
        assert polymers.homocoupling_frequency(frame) == [0, -2]

    def test_monomer_difference_frequency_sums_probabilities_per_difference(self, frame):
        diffs = polymers.monomer_difference_frequency(frame)
        assert diffs == {0: pytest.approx(0.75), -2: pytest.approx(0.25)}

    def test_normalization_makes_the_probabilities_sum_to_one(self):
        frame = label_frame({"2BT+2TT+2H": 3.0, "3BT+1TT+2H": 1.0})
        diffs = polymers.monomer_difference_frequency(frame, normalize=True)
        assert sum(diffs.values()) == pytest.approx(1.0)
        assert diffs[0] == pytest.approx(0.75)

    def test_unnormalized_measures_keep_the_raw_values(self):
        frame = label_frame({"2BT+2TT+2H": 3.0, "3BT+1TT+2H": 1.0})
        diffs = polymers.monomer_difference_frequency(frame, normalize=False)
        assert sum(diffs.values()) == pytest.approx(4.0)

    def test_threshold_drops_low_abundance_polymers(self, frame):
        diffs = polymers.monomer_difference_frequency(frame, thr=0.5)
        assert diffs == {0: pytest.approx(1.0)}  # only the 0.75 entry survives

    def test_a_negative_threshold_is_rejected(self, frame):
        with pytest.raises(AssertionError):
            polymers.monomer_difference_frequency(frame, thr=-1.0)

    def test_monomer_frequency_splits_the_two_monomers(self, frame):
        a_freq, b_freq = polymers.monomer_frequency(frame)
        assert a_freq == {2: pytest.approx(0.75), 3: pytest.approx(0.25)}
        assert b_freq == {2: pytest.approx(0.75), 1: pytest.approx(0.25)}

    def test_estimate_homocoupling_is_the_expected_absolute_difference(self, frame):
        # 0.75 * |2-2| + 0.25 * |1-3| = 0.5
        assert polymers.estimate_homocoupling(frame) == pytest.approx(0.5)

    def test_homocoupling_proportion_counts_only_differences_above_one(self, frame):
        assert polymers.homocoupling_proportion(frame) == pytest.approx(0.25)

    def test_alternating_polymers_have_no_homocoupling(self):
        frame = label_frame({"2BT+2TT+2H": 1.0})
        assert polymers.estimate_homocoupling(frame) == pytest.approx(0.0)
        assert polymers.homocoupling_proportion(frame) == pytest.approx(0.0)

    def test_difference_frequency_by_end_group_splits_by_end_groups(self):
        frame = label_frame({"2BT+2TT+2H": 0.5, "3BT+1TT+H+Methyl": 0.5})
        by_end, diff_min, diff_max = polymers.monomer_difference_frequency_by_end_group(frame)
        assert set(by_end) == {"2H", "H+Methyl"}
        assert by_end["2H"] == {0: pytest.approx(0.5)}
        assert by_end["H+Methyl"] == {-2: pytest.approx(0.5)}
        assert (diff_min, diff_max) == (-2, 0)

    def test_constrained_homocoupling_on_alternating_chain_with_neutral_ends(self):
        frame = label_frame({"2BT+2TT+2H": 1.0})
        counts, weighted = polymers.estimate_constrained_homocoupling(frame)
        assert counts == [0]
        assert weighted == pytest.approx(0.0)

    def test_constrained_homocoupling_with_determining_end_groups(self):
        # Both ends are Br (an "a" end group), so two of the 3 BT units sit at the
        # chain ends: min homocoupling = (3-2) - 1 + 1 = 1.
        frame = label_frame({"3BT+1TT+Br+Br": 1.0})
        counts, weighted = polymers.estimate_constrained_homocoupling(frame)
        assert counts == [1]
        assert weighted == pytest.approx(1.0)

    def test_constrained_homocoupling_with_mixed_end_groups(self):
        # One Br end (BT) and one Stannyl end (TT): min homocoupling = |a - b|.
        frame = label_frame({"3BT+1TT+Br+Stannyl": 1.0})
        counts, _ = polymers.estimate_constrained_homocoupling(frame)
        assert counts == [2]

    def test_constrained_homocoupling_weights_by_probability(self):
        frame = label_frame({"3BT+1TT+Br+Stannyl": 0.5, "2BT+2TT+2H": 0.5})
        _, weighted = polymers.estimate_constrained_homocoupling(frame)
        assert weighted == pytest.approx(1.0)  # 0.5 * 2 + 0.5 * 0


class TestLoadCentroidedSpectrum:
    def test_reads_a_two_line_csv(self, tmp_path):
        path = tmp_path / "centroided.csv"
        path.write_text("100.0,101.0,102.0\n0.5,0.25,0.25\n")
        s = polymers.load_centroided_spectrum(str(path), spectrum_label="sample")
        assert s.confs == [(100.0, 0.5), (101.0, 0.25), (102.0, 0.25)]
        assert s.label == "sample"
