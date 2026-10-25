import shutil

import pytest

from vectorprep import native


@pytest.fixture(scope="session")
def native_lib(tmp_path_factory):
    """Build the C++ backend into a temp dir (skips if no compiler)."""
    if not (shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("no C++ compiler")
    path = native.build(tmp_path_factory.mktemp("native"))
    return native.NativeLib.load(path)


@pytest.fixture
def native_prep(native_lib, monkeypatch):
    from vectorprep.core import VectorPreprocessor

    monkeypatch.setenv("VECTORPREP_LIB", str(native_lib.path))
    return VectorPreprocessor("native")
