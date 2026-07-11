import pytest

from masserstein.misc import closest, extract_range


@pytest.fixture
def confs():
    return [(1.0, 0.1), (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)]


class TestExtractRange:
    def test_extracts_an_inclusive_range(self, confs):
        assert list(extract_range(confs, 2.0, 3.0)) == [(2.0, 0.2), (3.0, 0.3)]

    def test_lower_bound_is_inclusive(self, confs):
        assert list(extract_range(confs, 2.0, 3.5)) == [(2.0, 0.2), (3.0, 0.3)]

    def test_bounds_are_inclusive_whatever_the_intensity(self, confs):
        # The bounds must be decided by the mass alone. Intensities above, below
        # and equal to zero at the boundary all have to be kept.
        L = [(1.0, -0.5), (2.0, 0.0), (3.0, 0.3)]
        assert list(extract_range(L, 1.0, 3.0)) == L

    def test_bounds_need_not_match_a_peak(self, confs):
        assert list(extract_range(confs, 1.5, 3.5)) == [(2.0, 0.2), (3.0, 0.3)]

    def test_range_covering_everything(self, confs):
        assert list(extract_range(confs, 0.0, 100.0)) == confs

    def test_range_entirely_above_the_data(self, confs):
        assert list(extract_range(confs, 10.0, 20.0)) == []

    def test_range_entirely_below_the_data(self, confs):
        assert list(extract_range(confs, -5.0, 0.5)) == []

    def test_empty_gap_inside_the_data(self, confs):
        assert list(extract_range(confs, 2.2, 2.8)) == []

    def test_single_point_range(self, confs):
        assert list(extract_range(confs, 3.0, 3.0)) == [(3.0, 0.3)]

    def test_empty_list(self):
        assert list(extract_range([], 0.0, 1.0)) == []


class TestClosest:
    def test_exact_match(self, confs):
        assert closest(confs, 3.0) == (3.0, 0.3)

    def test_rounds_to_the_nearer_neighbour(self, confs):
        assert closest(confs, 2.4) == (2.0, 0.2)
        assert closest(confs, 2.6) == (3.0, 0.3)

    def test_below_the_range_returns_the_first_element(self, confs):
        assert closest(confs, -100.0) == (1.0, 0.1)

    def test_above_the_range_returns_the_last_element(self, confs):
        assert closest(confs, 100.0) == (4.0, 0.4)

    def test_single_element_list(self):
        assert closest([(7.0, 1.0)], 0.0) == (7.0, 1.0)

    def test_ties_are_resolved_consistently(self, confs):
        # Exactly halfway between 2.0 and 3.0; min() keeps the first candidate.
        assert closest(confs, 2.5) in {(2.0, 0.2), (3.0, 0.3)}
