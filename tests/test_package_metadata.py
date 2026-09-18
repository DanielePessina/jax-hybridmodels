"""Release metadata stays aligned with the installed distribution."""

from __future__ import annotations

import importlib.metadata as metadata

import jaxhybridmodels


def test_distribution_metadata_matches_public_version() -> None:
    package = metadata.metadata("jaxhybridmodels")

    assert package["Name"] == "jaxhybridmodels"
    assert package["Version"] == "0.2.0b1"
    assert package["License-Expression"] == "BSD-3-Clause"
    assert "LICENSE" in package.get_all("License-File")
    assert "Development Status :: 4 - Beta" in package.get_all("Classifier")
    assert jaxhybridmodels.__version__ == package["Version"]
