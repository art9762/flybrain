import numpy as np
import pytest
from scipy import sparse

import flybrain.gpu as gpu


def signed_multigraph():
    # Two distinct edges from source 1 to target 0, plus inhibition and a
    # completely disconnected target. CSR data ordering must remain unchanged.
    return sparse.csr_matrix((
        np.array([1.5, -0.5, -2.0, 0.25], dtype=np.float32),
        np.array([1, 1, 0, 2], dtype=np.int32),
        np.array([0, 2, 3, 3, 4], dtype=np.int32),
    ), shape=(4, 4))


@pytest.mark.parametrize("backend", ["cpu", "mps"])
def test_signed_duplicate_edges_and_plastic_updates(backend):
    if backend == "mps" and not gpu.detect_gpu()["supported"]:
        pytest.skip("Metal GPU is unavailable to this test process")
    graph = signed_multigraph()
    operator = gpu.SparseOperator(graph, backend=backend)
    state = np.array([3.0, 2.0, -4.0, 9.0], dtype=np.float32)
    np.testing.assert_allclose(operator.matvec(state), [2.0, -6.0, 0.0, -1.0])
    operator.update_weights(np.array([1, 2]), np.array([2.0, -1.0]))
    np.testing.assert_allclose(operator.matvec(state), [7.0, -3.0, 0.0, -1.0])
    np.testing.assert_allclose(graph.data, [1.5, 2.0, -1.0, 0.25])
    # A later call must not mutate already returned products.
    previous = operator.matvec(state)
    operator.matvec(state * 2)
    np.testing.assert_allclose(previous, [7.0, -3.0, 0.0, -1.0])
    assert operator.status["backend"] == backend


def test_optional_torch_absence_falls_back_and_explicit_mps_reports_error(monkeypatch):
    monkeypatch.setattr(gpu, "_load_torch", lambda: None)
    operator = gpu.SparseOperator(signed_multigraph(), backend="auto")
    assert operator.status["backend"] == "cpu"
    assert not operator.status["supported"]
    assert "PyTorch" in operator.status["reason"]
    with pytest.raises(RuntimeError, match="PyTorch"):
        gpu.SparseOperator(signed_multigraph(), backend="mps")


def test_auto_falls_back_if_metal_kernel_cannot_initialize(monkeypatch):
    monkeypatch.setattr(gpu, "detect_gpu", lambda: {
        "backend": "mps", "device_name": "test", "supported": True, "reason": "test",
    })

    def fail(self):
        raise RuntimeError("unsupported kernel")

    monkeypatch.setattr(gpu.SparseOperator, "_initialize_mps", fail)
    operator = gpu.SparseOperator(signed_multigraph())
    assert operator.status["backend"] == "cpu"
    assert "unsupported kernel" in operator.status["reason"]


def test_invalid_weight_updates_do_not_change_graph():
    operator = gpu.SparseOperator(signed_multigraph(), "cpu")
    original = operator.matrix.data.copy()
    for indices, values in [([4], [1]), ([-1], [1]), ([0.5], [1]), ([0, 0], [1, 2]), ([0], [np.nan])]:
        with pytest.raises(ValueError):
            operator.update_weights(np.array(indices), np.array(values))
        np.testing.assert_array_equal(operator.matrix.data, original)
    with pytest.raises(ValueError, match="shape"):
        operator.matvec(np.zeros(3))
