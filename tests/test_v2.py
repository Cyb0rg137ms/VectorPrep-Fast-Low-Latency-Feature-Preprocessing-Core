import json

import numpy as np
import pytest

from vectorprep import (
    VectorPreprocessor,
    cosine_error,
    dequantize,
    explained_variance_ratio,
    pca_power,
    pca_svd,
    quantize_int8,
    sqnr_db,
)
from vectorprep.cli import main


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_numpy_does_not_mutate_input(rng):
    x = rng.standard_normal((5, 4)).astype(np.float32)
    x0 = x.copy()
    p = VectorPreprocessor("numpy")
    p.normalize_l2(x)
    p.scale_minmax(x[0])
    np.testing.assert_array_equal(x, x0)


def test_native_matches_numpy_and_does_not_mutate(native_prep, rng):
    ref = VectorPreprocessor("numpy")
    x = rng.standard_normal((300, 37)).astype(np.float32)
    x[3] = 0.0
    x0 = x.copy()
    np.testing.assert_allclose(native_prep.normalize_l2(x), ref.normalize_l2(x), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        native_prep.scale_minmax(x.ravel(), -1, 3), ref.scale_minmax(x.ravel(), -1, 3), atol=1e-5
    )
    np.testing.assert_array_equal(x, x0)  # v1 C++ path modified the caller's array
    assert np.all(native_prep.normalize_l2(x)[3] == 0)
    a, b = rng.standard_normal(1001), rng.standard_normal(1001)
    assert native_prep.dot_product(a, b) == pytest.approx(ref.dot_product(a, b), rel=1e-6)
    assert native_prep.backend == "native"


def test_native_inplace(native_prep, rng):
    x = rng.standard_normal((10, 8)).astype(np.float32)
    out = native_prep.normalize_l2(x, inplace=True)
    assert out is x
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, rtol=1e-6)
    with pytest.raises(ValueError):
        native_prep.normalize_l2(x.astype(np.float64), inplace=True)


def test_numpy_inplace(rng):
    x = rng.standard_normal((10, 8)).astype(np.float32)
    p = VectorPreprocessor("numpy")
    assert p.normalize_l2(x, inplace=True) is x
    v = np.array([1, 2, 3], np.float32)
    assert p.scale_minmax(v, inplace=True) is v
    np.testing.assert_allclose(v, [0, 0.5, 1])


def test_dot_accumulates_in_float64():
    a = np.full(10_000_000 // 10, 0.1, np.float32)
    exact = float(np.dot(a.astype(np.float64), a.astype(np.float64)))
    assert VectorPreprocessor("numpy").dot_product(a, a) == pytest.approx(exact, rel=1e-12)


def test_native_backend_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("VECTORPREP_LIB", str(tmp_path / "missing.so"))
    monkeypatch.setenv("VECTORPREP_CACHE", str(tmp_path))
    from vectorprep import native

    monkeypatch.setattr(native, "candidate_paths", lambda: [tmp_path / "missing.so"])
    with pytest.raises(RuntimeError):
        VectorPreprocessor("native")
    assert VectorPreprocessor().backend == "numpy"


def test_empty_and_shape_errors():
    p = VectorPreprocessor("numpy")
    assert p.scale_minmax(np.array([], np.float32)).size == 0
    with pytest.raises(ValueError):
        p.normalize_l2(np.ones(3))
    with pytest.raises(ValueError):
        p.dot_product(np.ones(3), np.ones(4))


def test_quantize_roundtrip_and_metrics(rng):
    x = rng.standard_normal((200, 64)).astype(np.float32)
    for per_row in (False, True):
        q, s = quantize_int8(x, per_row=per_row)
        assert q.dtype == np.int8 and np.abs(q).max() == 127
        xh = dequantize(q, s)
        assert np.abs(x - xh).max() <= s.max() / 2 + 1e-6
        assert 30 < sqnr_db(x, xh) < 50
        assert cosine_error(x, xh) < 0.01
    assert sqnr_db(x, x) == float("inf")
    with pytest.raises(ValueError):
        quantize_int8(np.ones(3), per_row=True)


def test_per_row_beats_per_tensor_on_varying_norms(rng):
    x = (rng.standard_normal((500, 64)) * rng.lognormal(0, 1, (500, 1))).astype(np.float32)
    s_t = sqnr_db(x, dequantize(*quantize_int8(x)))
    s_r = sqnr_db(x, dequantize(*quantize_int8(x, per_row=True)))
    assert s_r > s_t


def test_pca_power_matches_svd(rng):
    basis, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    x = (rng.standard_normal((2000, 30)) * np.sqrt(1 / np.arange(1, 31))) @ basis.T
    comps, evals, proj = pca_power(x, 5, seed=1)
    ref, evr = pca_svd(x, 5)
    assert np.abs((comps * ref).sum(1)).min() > 0.999
    np.testing.assert_allclose(explained_variance_ratio(x, evals), evr, rtol=1e-6)
    assert proj.shape == (2000, 5)
    c2, _, _ = pca_power(x, 5, seed=1)
    np.testing.assert_array_equal(comps, c2)  # seeded → deterministic


def test_v1_quantize_signature():
    q, scale = VectorPreprocessor("numpy").quantize_embeddings(np.array([-1.0, 1.0]))
    assert isinstance(scale, float) and list(q) == [-127, 127]


def test_bench_quick(tmp_path):
    pytest.importorskip("matplotlib")
    from vectorprep.benchmarks import run

    res = run(tmp_path, quick=True)
    assert (tmp_path / "benchmark.png").stat().st_size > 1000
    assert json.loads((tmp_path / "bench.json").read_text())["pca"][0]["min_axis_alignment"] > 0.99
    assert all(r["bytes_ratio"] > 3.9 for r in res["quantization"] if r["scheme"] == "per_tensor")


def test_cli(tmp_path, capsys, native_lib, monkeypatch):
    x = np.random.default_rng(0).standard_normal((20, 6)).astype(np.float32)
    np.save(tmp_path / "x.npy", x)
    for op in ("l2", "minmax", "pca"):
        assert main(["transform", op, str(tmp_path / "x.npy"), str(tmp_path / f"{op}.npy"), "--k", "2"]) == 0
    assert np.load(tmp_path / "pca.npy").shape == (20, 2)
    assert main(["transform", "int8", str(tmp_path / "x.npy"), str(tmp_path / "q.npz")]) == 0
    assert np.load(tmp_path / "q.npz")["q"].dtype == np.int8
    assert main(["build-native", "--out-dir", str(tmp_path / "b")]) == 0
    monkeypatch.setenv("VECTORPREP_LIB", str(native_lib.path))
    assert main(["info"]) == 0
    assert '"native"' in capsys.readouterr().out.splitlines()[-1]


def test_example_runs():
    import runpy

    runpy.run_path("examples/embedding_pipeline.py")
