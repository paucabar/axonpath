"""
Tests for the BioImage.IO export metadata (README, citation, config defaults).

These cover what ends up permanently in published model packages, without
running a model export.

Run from the axonpath conda environment:
    pytest tests/test_export_metadata.py -v
"""

from pathlib import Path

import pytest

from axonpath.export_utils.config import BioimageioExportConfig
from axonpath.export_utils.create_bioimageio_model import build_cite_entry, write_readme

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_config(tmp_path, **overrides):
    values = dict(
        model_path="unused.pth",
        model_name="em_cns",
        model_version="0.1.0",
        test_img_path="unused.tif",
        output_dir=str(tmp_path),
        model_pixel_size=0.008,
        min_diameter=20,
        predict_inner_cylinder=True,
        description="AxonPath model for EM images of the central nervous system (0.008 µm/pixel).",
        citation_text="Carrillo-Barberà P, et al. (2026). AxonPath segmentation models (v0.1.0). Zenodo.",
        citation_doi="10.5281/zenodo.22982902",
        author_names=["Pau Carrillo-Barberà", "Alan O'Callaghan"],
    )
    values.update(overrides)
    return BioimageioExportConfig(**values)


def read_readme(tmp_path, config):
    return (tmp_path / write_readme(config)).read_text(encoding="utf-8")


def test_readme_em_model(tmp_path):
    """EM README describes the three-level hierarchy, pixel size, usage and citation."""
    text = read_readme(tmp_path, make_config(tmp_path))
    assert text.startswith("# em_cns-0.1.0")
    assert "central nervous system" in text
    assert "Fibre > InnerCylinder > Axon" in text
    assert "0.008 µm/pixel" in text
    assert "https://github.com/paucabar/qupath-extension-axonpath" in text
    assert "https://github.com/paucabar/axonpath" in text
    assert "https://doi.org/10.5281/zenodo.22982902" in text
    assert "Alan O'Callaghan" in text


def test_readme_brightfield_model(tmp_path):
    """Without inner cylinders, the README describes the two-level hierarchy only."""
    config = make_config(tmp_path, model_name="brightfield", predict_inner_cylinder=False,
                         model_pixel_size=0.071, min_diameter=15)
    text = read_readme(tmp_path, config)
    assert "Fibre > Axon" in text
    assert "InnerCylinder" not in text
    assert "0.071 µm/pixel" in text


def test_readme_without_doi(tmp_path):
    """No DOI line is written when no DOI is given."""
    text = read_readme(tmp_path, make_config(tmp_path, citation_doi=None))
    assert "doi.org" not in text


def test_cite_entry_uses_doi(tmp_path):
    cite = build_cite_entry(make_config(tmp_path))
    assert str(cite.doi) == "10.5281/zenodo.22982902"


def test_cite_entry_falls_back_to_repo_url(tmp_path):
    """BioImage.IO needs a DOI or URL; without a DOI the training repo is cited."""
    cite = build_cite_entry(make_config(tmp_path, citation_doi=None))
    assert cite.doi is None
    assert str(cite.url).rstrip("/") == "https://github.com/paucabar/axonpath"


def test_defaults_point_to_renamed_repo(tmp_path):
    config = make_config(tmp_path)
    assert config.git_repo == "https://github.com/paucabar/axonpath"


@pytest.mark.parametrize("path", ["scripts/export_bioimageio.py", "axonpath/export_utils/config.py",
                                  "axonpath/export_utils/create_bioimageio_model.py"])
def test_no_placeholder_doi_or_old_repo(path):
    """A placeholder DOI or the old repo name must never reach a published package."""
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert "fake-doi" not in source
    assert "paucabar/aimseg-dl" not in source
