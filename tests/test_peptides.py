from collections import Counter

import pytest

from masserstein.peptides import (
    aacnt_to_elecnt,
    aminoacids,
    daminoacids,
    get_protein_counter,
    get_protein_formula,
)


def formula_to_counter(formula):
    import re

    counts = Counter()
    for element, number in re.findall("([A-Z][a-z]*)([0-9]*)", formula):
        if element:
            counts[element] += int(number) if number else 1
    return counts


class TestAminoacidTable:
    def test_all_twenty_standard_aminoacids_are_present(self):
        standard = set("ACDEFGHIKLMNPQRSTVWY")
        assert standard <= set(aminoacids)

    def test_leucine_and_isoleucine_are_isomers(self):
        assert aminoacids["L"] == aminoacids["I"]

    def test_whitespace_maps_to_an_empty_formula(self):
        assert aminoacids[" "] == ""
        assert aminoacids["\n"] == ""

    def test_parsed_counters_match_the_formula_strings(self):
        assert daminoacids["G"] == Counter({"C": 2, "H": 3, "N": 1, "O": 1})
        assert daminoacids["C"]["S"] == 1
        assert daminoacids["U"]["Se"] == 1


class TestElementCounts:
    def test_water_is_added_by_default(self):
        assert aacnt_to_elecnt(Counter()) == Counter({"H": 2, "O": 1})

    def test_water_can_be_suppressed(self):
        assert aacnt_to_elecnt(Counter(), add_water=False) == Counter()

    def test_single_residue_plus_water(self):
        # Glycine residue is C2H3NO; the free aminoacid adds H2O.
        assert aacnt_to_elecnt(Counter({"G": 1})) == Counter({"C": 2, "H": 5, "N": 1, "O": 2})

    def test_residue_counts_are_multiplied(self):
        assert aacnt_to_elecnt(Counter({"G": 3}), add_water=False) == Counter(
            {"C": 6, "H": 9, "N": 3, "O": 3}
        )


class TestProteinFormula:
    def test_glycine(self):
        assert formula_to_counter(get_protein_formula("G")) == Counter(
            {"C": 2, "H": 5, "N": 1, "O": 2}
        )

    def test_a_peptide_is_the_sum_of_its_residues_plus_water(self):
        counts = formula_to_counter(get_protein_formula("GGG"))
        assert counts == Counter({"C": 6, "H": 11, "N": 3, "O": 4})

    def test_formula_is_order_independent(self):
        assert get_protein_formula("GAV") == get_protein_formula("VAG")

    def test_water_can_be_suppressed(self):
        with_water = formula_to_counter(get_protein_formula("GA"))
        without = formula_to_counter(get_protein_formula("GA", add_water=False))
        assert with_water - without == Counter({"H": 2, "O": 1})

    def test_whitespace_and_newlines_are_ignored(self):
        assert get_protein_formula("GA GA\n") == get_protein_formula("GAGA")

    def test_unknown_symbols_are_ignored(self):
        assert get_protein_formula("GXZ") == get_protein_formula("G")

    def test_formula_is_sorted_by_element(self):
        formula = get_protein_formula("CGM")  # contains sulfur
        elements = [e for e, _ in __import__("re").findall("([A-Z][a-z]*)([0-9]*)", formula) if e]
        assert elements == sorted(elements)

    def test_formula_is_parseable_by_spectrum(self):
        from masserstein import Spectrum

        s = Spectrum(get_protein_formula("PEPTIDE"), threshold=0.01)
        assert len(s) > 1
        # PEPTIDE has a monoisotopic mass of ~799.36 Da.
        assert s.confs[0][0] == pytest.approx(799.36, abs=0.01)

    def test_protein_counter_matches_the_formula(self):
        counter = get_protein_counter("PEPTIDE")
        assert counter == formula_to_counter(get_protein_formula("PEPTIDE"))


class TestModifications:
    def test_oxidation_adds_one_oxygen(self):
        plain = formula_to_counter(get_protein_formula("MGG"))
        oxidized = formula_to_counter(get_protein_formula("oxMGG"))
        assert oxidized - plain == Counter({"O": 1})

    def test_deamidation_of_asparagine(self):
        # deaN: -H, +N, +O relative to the unmodified sequence.
        plain = formula_to_counter(get_protein_formula("NGG"))
        modified = formula_to_counter(get_protein_formula("deaNGG"))
        assert modified["O"] - plain["O"] == 1
        assert modified["N"] - plain["N"] == 1
        assert modified["H"] - plain["H"] == -1

    def test_carbamidomethylation_of_cysteine(self):
        plain = formula_to_counter(get_protein_formula("CGG"))
        modified = formula_to_counter(get_protein_formula("carCGG"))
        assert modified - plain == Counter({"C": 2, "H": 4, "N": 2, "O": 1})

    def test_modifications_are_counted_once_per_occurrence(self):
        plain = formula_to_counter(get_protein_formula("MM"))
        once = formula_to_counter(get_protein_formula("oxMM"))
        twice = formula_to_counter(get_protein_formula("oxMoxM"))
        assert once["O"] - plain["O"] == 1
        assert twice["O"] - plain["O"] == 2
