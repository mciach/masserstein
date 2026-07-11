import math

import numpy as np
import pytest

from masserstein.base_spectrum import BaseSpectrum

from helpers import gaussian_profile


class TestConstruction:
    def test_empty_spectrum_has_no_confs(self):
        s = BaseSpectrum()
        assert s.confs == []
        assert s.empty
        assert len(s) == 0

    def test_confs_are_stored(self):
        s = BaseSpectrum(confs=[(1.0, 2.0)], label="x")
        assert s.confs == [(1.0, 2.0)]
        assert s.label == "x"
        assert not s.empty
        assert len(s) == 1

    def test_label_defaults_to_the_empty_string(self):
        assert BaseSpectrum().label == ""
        assert BaseSpectrum(confs=[(1.0, 1.0)]).label == ""

    def test_confs_are_sorted_on_construction(self):
        s = BaseSpectrum(confs=[(3.0, 1.0), (1.0, 1.0), (2.0, 1.0)])
        assert [mz for mz, _ in s.confs] == [1.0, 2.0, 3.0]

    def test_duplicate_masses_are_merged_on_construction(self):
        s = BaseSpectrum(confs=[(1.0, 0.25), (2.0, 0.5), (1.0, 0.75)])
        assert s.confs == [(1.0, 1.0), (2.0, 0.5)]

    def test_set_confs_replaces_previous_content(self, two_peaks):
        two_peaks.set_confs([(5.0, 1.0)])
        assert two_peaks.confs == [(5.0, 1.0)]

    def test_empty_reflects_confs_after_mutation(self, two_peaks):
        assert not two_peaks.empty
        two_peaks.confs = []
        assert two_peaks.empty

    def test_copy_is_deep(self, two_peaks):
        clone = two_peaks.copy()
        clone.confs[0] = (99.0, 99.0)
        assert two_peaks.confs[0] == (1.0, 0.5)

    def test_new_random_stays_in_domain(self):
        s = BaseSpectrum.new_random(domain=(10.0, 20.0), peaks=25)
        assert len(s) == 25
        assert all(10.0 <= mz <= 20.0 for mz, _ in s.confs)
        assert all(0.0 <= i <= 1.0 for _, i in s.confs)
        assert [mz for mz, _ in s.confs] == sorted(mz for mz, _ in s.confs)


class TestCsvRoundTrip:
    def test_new_from_csv_skips_header_and_comments(self, tmp_path):
        path = tmp_path / "spectrum.csv"
        path.write_text("mz,intensity\n2.0,0.5\n# a comment\n1.0,0.25\n")
        s = BaseSpectrum.new_from_csv(str(path))
        assert s.confs == [(1.0, 0.25), (2.0, 0.5)]
        assert s.label == str(path)

    def test_new_from_csv_honours_delimiter_and_merges(self, tmp_path):
        path = tmp_path / "spectrum.tsv"
        path.write_text("mz\tintensity\n1.0\t0.25\n1.0\t0.75\n")
        s = BaseSpectrum.new_from_csv(str(path), delimiter="\t")
        assert s.confs == [(1.0, 1.0)]


class TestArithmetic:
    def test_add_sums_intensities_at_shared_masses(self):
        a = BaseSpectrum(confs=[(1.0, 1.0), (2.0, 1.0)], label="a")
        b = BaseSpectrum(confs=[(2.0, 3.0), (3.0, 1.0)], label="b")
        c = a + b
        assert c.confs == [(1.0, 1.0), (2.0, 4.0), (3.0, 1.0)]
        assert c.label == "a + b"

    def test_add_does_not_modify_operands(self):
        a = BaseSpectrum(confs=[(1.0, 1.0)])
        b = BaseSpectrum(confs=[(1.0, 1.0)])
        a + b
        assert a.confs == [(1.0, 1.0)]
        assert b.confs == [(1.0, 1.0)]

    def test_unlabelled_spectra_can_be_added(self):
        a = BaseSpectrum(confs=[(1.0, 1.0)])
        b = BaseSpectrum(confs=[(2.0, 1.0)])
        c = a + b
        assert c.confs == [(1.0, 1.0), (2.0, 1.0)]
        assert c.label == " + "

    def test_mul_scales_intensities_only(self, two_peaks):
        scaled = two_peaks * 4
        assert scaled.confs == [(1.0, 2.0), (2.0, 2.0)]
        assert two_peaks.confs == [(1.0, 0.5), (2.0, 0.5)]

    def test_mul_is_commutative(self, two_peaks):
        assert (3 * two_peaks).confs == (two_peaks * 3).confs

    def test_scalar_product_merges_weighted_spectra(self):
        a = BaseSpectrum(confs=[(1.0, 1.0), (2.0, 1.0)])
        b = BaseSpectrum(confs=[(2.0, 1.0), (3.0, 1.0)])
        res = BaseSpectrum.ScalarProduct([a, b], [2.0, 10.0])
        assert res.confs == [(1.0, 2.0), (2.0, 12.0), (3.0, 10.0)]


class TestNormalization:
    def test_normalize_sums_to_one(self):
        s = BaseSpectrum(confs=[(1.0, 3.0), (2.0, 1.0)])
        s.normalize()
        assert math.isclose(sum(i for _, i in s.confs), 1.0)
        assert math.isclose(s.confs[0][1], 0.75)

    def test_normalize_to_arbitrary_target(self, two_peaks):
        two_peaks.normalize(target_value=10.0)
        assert math.isclose(sum(i for _, i in two_peaks.confs), 10.0)

    def test_normalize_preserves_masses(self, two_peaks):
        two_peaks.normalize(3.0)
        assert [mz for mz, _ in two_peaks.confs] == [1.0, 2.0]


class TestWassersteinDistance:
    def test_distance_between_identical_spectra_is_zero(self, two_peaks):
        assert two_peaks.WSDistance(two_peaks.copy()) == pytest.approx(0.0)

    def test_shifted_dirac_distance_equals_shift(self):
        a = BaseSpectrum(confs=[(0.0, 1.0)])
        b = BaseSpectrum(confs=[(2.5, 1.0)])
        assert a.WSDistance(b) == pytest.approx(2.5)

    def test_split_mass_is_averaged(self):
        # Half the mass travels 2 units, half travels 1 unit -> 1.5 on average.
        a = BaseSpectrum(confs=[(0.0, 0.5), (1.0, 0.5)])
        b = BaseSpectrum(confs=[(2.0, 1.0)])
        assert a.WSDistance(b) == pytest.approx(1.5)

    def test_distance_is_symmetric(self, two_peaks):
        other = BaseSpectrum(confs=[(1.5, 0.3), (4.0, 0.7)])
        assert two_peaks.WSDistance(other) == pytest.approx(other.WSDistance(two_peaks))

    def test_transport_plan_conserves_mass(self):
        a = BaseSpectrum(confs=[(0.0, 0.4), (1.0, 0.6)])
        b = BaseSpectrum(confs=[(0.5, 0.5), (2.0, 0.5)])
        plan = list(a.WSDistanceMoves(b))
        assert math.fsum(m for _, _, m in plan) == pytest.approx(1.0)
        # Every unit of transported mass is accounted for on both sides.
        assert math.fsum(m for src, _, m in plan if src == 0.5) == pytest.approx(0.5)
        assert math.fsum(m for _, dst, m in plan if dst == 1.0) == pytest.approx(0.6)

    def test_plan_cost_equals_distance(self):
        a = BaseSpectrum(confs=[(0.0, 0.4), (1.0, 0.6)])
        b = BaseSpectrum(confs=[(0.5, 0.5), (2.0, 0.5)])
        cost = math.fsum(abs(src - dst) * m for src, dst, m in a.WSDistanceMoves(b))
        assert cost == pytest.approx(a.WSDistance(b))

    def test_unnormalized_self_is_rejected(self, two_peaks):
        unnormalized = BaseSpectrum(confs=[(1.0, 2.0)])
        with pytest.raises(ValueError, match="Self is not normalized"):
            unnormalized.WSDistance(two_peaks)

    def test_unnormalized_other_is_rejected(self, two_peaks):
        unnormalized = BaseSpectrum(confs=[(1.0, 2.0)])
        with pytest.raises(ValueError, match="Other is not normalized"):
            two_peaks.WSDistance(unnormalized)

    def test_triangle_inequality(self):
        a = BaseSpectrum(confs=[(0.0, 0.5), (1.0, 0.5)])
        b = BaseSpectrum(confs=[(1.0, 0.25), (3.0, 0.75)])
        c = BaseSpectrum(confs=[(2.0, 1.0)])
        assert a.WSDistance(c) <= a.WSDistance(b) + b.WSDistance(c) + 1e-12


class TestPeakStatistics:
    def test_get_modal_peak(self):
        s = BaseSpectrum(confs=[(1.0, 0.2), (2.0, 0.9), (3.0, 0.4)])
        assert s.get_modal_peak() == (2.0, 0.9)

    def test_explained_intensity_is_pointwise_minimum(self):
        a = BaseSpectrum(confs=[(1.0, 0.3), (2.0, 0.7)])
        b = BaseSpectrum(confs=[(1.0, 0.5), (2.0, 0.5)])
        assert a.explained_intensity(b) == pytest.approx(0.8)

    def test_explained_intensity_of_self_is_total_intensity(self, two_peaks):
        assert two_peaks.explained_intensity(two_peaks) == pytest.approx(1.0)

    def test_find_peaks_reports_interior_local_maxima(self):
        s = BaseSpectrum(confs=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0), (3.0, 2.0), (4.0, 0.0)])
        assert s.find_peaks() == [(1.0, 1.0), (3.0, 2.0)]

    def test_find_peaks_ignores_boundary_maxima(self):
        s = BaseSpectrum(confs=[(0.0, 5.0), (1.0, 1.0), (2.0, 5.0)])
        assert s.find_peaks() == []

    def test_find_peaks_on_monotonic_signal(self):
        s = BaseSpectrum(confs=[(float(i), float(i)) for i in range(5)])
        assert s.find_peaks() == []


class TestCleaning:
    def test_trim_negative_intensities(self):
        s = BaseSpectrum(confs=[(1.0, -0.5), (2.0, 0.5)])
        s.trim_negative_intensities()
        assert s.confs == [(1.0, 0.0), (2.0, 0.5)]

    def test_trim_keeps_masses_and_positive_values(self):
        s = BaseSpectrum(confs=[(1.0, -1e-9), (2.0, 3.0)])
        s.trim_negative_intensities()
        assert [mz for mz, _ in s.confs] == [1.0, 2.0]
        assert s.confs[1][1] == 3.0

    def test_cut_smallest_peaks_removes_up_to_the_budget(self):
        s = BaseSpectrum(confs=[(1.0, 0.001), (2.0, 0.004), (3.0, 0.995)])
        s.cut_smallest_peaks(removed_proportion=0.005)
        # 0.001 + 0.004 = 0.005 of a total of 1.0 may be dropped.
        assert s.confs == [(3.0, 0.995)]

    def test_cut_smallest_peaks_respects_the_budget(self):
        s = BaseSpectrum(confs=[(1.0, 0.1), (2.0, 0.9)])
        s.cut_smallest_peaks(removed_proportion=0.05)
        assert len(s) == 2  # dropping the 0.1 peak would exceed a 0.05 budget

    def test_cut_smallest_peaks_leaves_confs_sorted(self):
        s = BaseSpectrum(confs=[(float(i), float(i)) for i in range(10)])
        s.cut_smallest_peaks(removed_proportion=0.1)
        assert [mz for mz, _ in s.confs] == sorted(mz for mz, _ in s.confs)

    def test_zero_proportion_removes_only_zero_peaks(self):
        s = BaseSpectrum(confs=[(1.0, 0.0), (2.0, 1.0)])
        s.cut_smallest_peaks(removed_proportion=0.0)
        assert s.confs == [(2.0, 1.0)]


class TestFilterAgainstOther:
    def test_keeps_peaks_within_margin_of_a_single_spectrum(self):
        subject = BaseSpectrum(confs=[(1.0, 1.0), (5.0, 1.0)])
        reference = BaseSpectrum(confs=[(1.05, 1.0)])
        filtered = subject.filter_against_other(reference, margin=0.15)
        assert filtered.confs == [(1.0, 1.0)]

    def test_accepts_an_iterable_of_spectra(self):
        subject = BaseSpectrum(confs=[(1.0, 1.0), (5.0, 1.0), (9.0, 1.0)])
        refs = [BaseSpectrum(confs=[(1.0, 1.0)]), BaseSpectrum(confs=[(9.0, 1.0)])]
        filtered = subject.filter_against_other(refs, margin=0.15)
        assert [mz for mz, _ in filtered.confs] == [1.0, 9.0]

    def test_does_not_modify_the_subject(self):
        subject = BaseSpectrum(confs=[(1.0, 1.0), (5.0, 1.0)])
        subject.filter_against_other(BaseSpectrum(confs=[(1.0, 1.0)]))
        assert len(subject) == 2

    def test_wide_margin_keeps_everything(self):
        subject = BaseSpectrum(confs=[(1.0, 1.0), (5.0, 1.0)])
        filtered = subject.filter_against_other(BaseSpectrum(confs=[(3.0, 1.0)]), margin=10.0)
        assert len(filtered) == 2

    def test_label_is_preserved(self):
        subject = BaseSpectrum(confs=[(1.0, 1.0)], label="subject")
        filtered = subject.filter_against_other(BaseSpectrum(confs=[(1.0, 1.0)]))
        assert filtered.label == "subject"


class TestNoise:
    def test_chemical_noise_adds_the_requested_number_of_peaks(self, two_peaks):
        two_peaks.add_chemical_noise(nb_of_noise_peaks=8, noise_fraction=0.2)
        assert len(two_peaks) == 10

    def test_chemical_noise_has_the_requested_fraction(self):
        s = BaseSpectrum(confs=[(1.0, 0.5), (2.0, 0.5)])
        signal = 1.0
        s.add_chemical_noise(nb_of_noise_peaks=50, noise_fraction=0.25)
        total = sum(i for _, i in s.confs)
        # noise / total == 0.25 for a signal of 1.0
        assert (total - signal) / total == pytest.approx(0.25)

    def test_chemical_noise_respects_an_explicit_span(self):
        s = BaseSpectrum(confs=[(10.0, 0.5), (11.0, 0.5)])
        s.add_chemical_noise(20, 0.3, span=(100.0, 200.0))
        noise_mz = [mz for mz, _ in s.confs if mz > 50]
        assert len(noise_mz) == 20
        assert all(100.0 <= mz <= 200.0 for mz in noise_mz)

    def test_chemical_noise_widens_the_range_by_a_float_span(self):
        s = BaseSpectrum(confs=[(10.0, 0.5), (20.0, 0.5)])
        s.add_chemical_noise(30, 0.3, span=1.5)
        # span=1.5 grows the (10, 20) range by 25% on each side.
        assert all(7.5 <= mz <= 22.5 for mz, _ in s.confs)

    def test_gaussian_noise_perturbs_intensities_but_not_masses(self):
        s = BaseSpectrum(confs=[(float(i), 10.0) for i in range(50)])
        s.add_gaussian_noise(sd=0.5)
        assert all(mz == float(i) for i, (mz, _) in enumerate(s.confs))
        assert any(i != 10.0 for _, i in s.confs)
        assert np.mean([i for _, i in s.confs]) == pytest.approx(10.0, abs=0.5)

    def test_gaussian_noise_drops_non_positive_peaks(self):
        s = BaseSpectrum(confs=[(float(i), 0.0) for i in range(200)])
        s.add_gaussian_noise(sd=1.0)
        assert len(s) < 200
        assert all(i > 0 for _, i in s.confs)


class TestCentroiding:
    def test_recovers_position_and_area_of_a_single_gaussian(self):
        sd, height = 0.02, 3.0
        s = BaseSpectrum(confs=gaussian_profile([100.0], [height], sd=sd, step=0.001))
        centroids, apices = s.centroid(max_width=0.5)
        assert len(centroids) == 1
        mz, area = centroids[0]
        assert mz == pytest.approx(100.0, abs=1e-3)
        # The FWHM region of a Gaussian holds ~76% of its total area.
        full_area = height * sd * math.sqrt(2 * math.pi)
        assert area == pytest.approx(0.7610 * full_area, rel=0.02)
        assert apices[0][0] == pytest.approx(100.0, abs=1e-2)
        assert apices[0][1] == pytest.approx(height, rel=0.01)

    def test_resolves_two_separated_peaks(self):
        s = BaseSpectrum(confs=gaussian_profile([100.0, 101.0], [1.0, 2.0], sd=0.02, step=0.001))
        centroids, apices = s.centroid(max_width=0.5)
        assert [round(mz, 1) for mz, _ in centroids] == [100.0, 101.0]
        # Areas keep the 1:2 height ratio of the underlying peaks.
        assert centroids[1][1] / centroids[0][1] == pytest.approx(2.0, rel=0.02)

    def test_max_width_discards_broad_peaks(self):
        s = BaseSpectrum(confs=gaussian_profile([100.0], [1.0], sd=0.05, step=0.002))
        wide_enough, _ = s.centroid(max_width=1.0)
        too_narrow, _ = s.centroid(max_width=0.01)
        assert len(wide_enough) == 1
        assert len(too_narrow) == 0

    def test_peak_height_fraction_changes_the_integrated_area(self):
        s = BaseSpectrum(confs=gaussian_profile([100.0], [1.0], sd=0.02, step=0.001))
        half, _ = s.centroid(max_width=0.5, peak_height_fraction=0.5)
        low, _ = s.centroid(max_width=0.5, peak_height_fraction=0.1)
        # Integrating down to 10% of the apex captures more of the peak than FWHM does.
        assert low[0][1] > half[0][1]
        assert low[0][0] == pytest.approx(half[0][0], abs=1e-3)

    def test_warns_on_negative_intensities(self):
        confs = gaussian_profile([100.0], [1.0], sd=0.02, step=0.002)
        confs[0] = (confs[0][0], -1.0)
        s = BaseSpectrum(confs=confs)
        with pytest.warns(UserWarning, match="negative intensities"):
            s.centroid(max_width=0.5)


class TestPlotting:
    """Plotting needs the optional 'graphics' extra."""

    @pytest.fixture(autouse=True)
    def plt(self):
        yield pytest.importorskip("matplotlib.pyplot")

    def test_plot_does_not_raise(self, two_peaks, plt):
        two_peaks.plot(show=False)
        two_peaks.plot(show=False, profile=True)
        plt.close("all")

    def test_plot_all_does_not_raise(self, two_peaks, plt):
        BaseSpectrum.plot_all([two_peaks, two_peaks.copy()], show=False)
        plt.close("all")
