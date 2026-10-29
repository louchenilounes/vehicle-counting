"""Basic checks: the package and core libraries can be imported."""

import importlib

import pytest

import vehicle_counter


def test_package_version():
    assert vehicle_counter.__version__ == "0.1.0"


@pytest.mark.parametrize("module", ["cv2", "numpy", "torch", "ultralytics"])
def test_core_dependency_imports(module):
    importlib.import_module(module)
