import pytest

import masserstein


class TestPublicApi:
    def test_version_is_exposed(self):
        assert isinstance(masserstein.__version__, str)
        assert masserstein.__version__

    @pytest.mark.parametrize(
        "name",
        [
            "Spectrum",
            "estimate_proportions",
            "dualdeconv2",
            "intensity_generator",
        ],
    )
    def test_top_level_names_are_importable(self, name):
        assert hasattr(masserstein, name)

    def test_submodules_import_without_side_effects(self):
        # masserstein.model_selection is deliberately excluded: importing it runs a
        # 10k-iteration simulation and writes fig.png (see the notes in the PR).
        import masserstein.base_spectrum  # noqa: F401
        import masserstein.deconv_simplex  # noqa: F401
        import masserstein.misc  # noqa: F401
        import masserstein.peptides  # noqa: F401
        import masserstein.spectrum  # noqa: F401
