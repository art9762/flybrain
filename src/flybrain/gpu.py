"""Optional Metal acceleration of the complete, sparse recurrent graph.

PyTorch's sparse matrix kernels are not available on MPS. This module instead
keeps edge source indices, target indices and weights on the GPU, gathers source
activity per edge and adds each weighted contribution to its target neuron.
No dense neuron-by-neuron matrix is constructed. CPU fallback uses SciPy CSR.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class GPUStatus:
    backend: str
    device_name: str
    supported: bool
    reason: str


@lru_cache(maxsize=1)
def _load_torch():
    try:
        import torch
    except (ImportError, OSError):
        return None
    return torch


def detect_gpu() -> dict:
    """Report what this process can use, without making a speedup claim."""
    torch = _load_torch()
    if torch is None:
        return asdict(GPUStatus("cpu", "CPU", False, "Optional PyTorch is not installed."))
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_built():
        return asdict(GPUStatus("cpu", "CPU", False, "PyTorch was built without Metal/MPS support."))
    if not torch.backends.mps.is_available():
        return asdict(GPUStatus(
            "cpu", "CPU", False,
            "Metal/MPS is unavailable to this process; a restricted sandbox can hide an available GPU.",
        ))
    return asdict(GPUStatus("mps", "Apple Metal GPU", True, "Metal/MPS is available."))


class SparseOperator:
    """Incoming CSR matrix-vector product with optional resident GPU edges.

    Rows are target neurons and columns are source neurons. Weight updates use
    positions in the original CSR ``data`` array, including duplicate edges.
    Float32 CSR input is shared with the CPU operator; use ``update_weights`` to
    keep the GPU copy synchronized. An operator is not safe for concurrent calls.

    ``auto`` compares real CPU and GPU products (including activity transfers)
    once at startup. ``mps`` explicitly selects Metal and raises if it is unusable.
    """

    def __init__(self, matrix: sparse.csr_matrix, backend: str = "auto"):
        if backend not in {"auto", "cpu", "mps"}:
            raise ValueError("backend must be auto, cpu, or mps")
        if not sparse.issparse(matrix) or len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square SciPy sparse matrix")
        self.matrix = matrix.tocsr(copy=False).astype(np.float32, copy=False)
        self.backend = "cpu"
        self._torch = None
        self._sources = self._targets = self._weights = self._output = None
        self._status = {
            **detect_gpu(),
            "backend": "cpu",
            "requested_backend": backend,
            "device_name": "CPU",
            "implementation": "scipy_csr",
            "neuron_count": int(self.matrix.shape[0]),
            "edge_count": int(self.matrix.nnz),
            "cpu_ms": None,
            "mps_ms": None,
        }
        if backend == "cpu":
            self._status["reason"] = "CPU was explicitly selected."
            return
        if not self._status["supported"]:
            if backend == "mps":
                raise RuntimeError(self._status["reason"])
            return
        try:
            self._initialize_mps()
            self.backend = "mps"
            if backend == "auto":
                self._benchmark_and_select()
            else:
                # Execute once so unsupported kernels fail at initialization.
                self._matvec_mps(np.zeros(self.matrix.shape[0], dtype=np.float32))
                self._status["reason"] = "Metal was explicitly selected; all graph edges are evaluated on GPU."
        except (RuntimeError, NotImplementedError) as error:
            self._release_mps()
            self.backend = "cpu"
            if backend == "mps":
                raise RuntimeError(f"Metal graph initialization failed: {error}") from error
            self._status["reason"] = f"Metal graph initialization failed; using CPU: {error}"
        self._status["backend"] = self.backend
        self._status["device_name"] = "Apple Metal GPU" if self.backend == "mps" else "CPU"
        self._status["implementation"] = "mps_edge_gather_scatter" if self.backend == "mps" else "scipy_csr"

    @property
    def status(self) -> dict:
        return dict(self._status)

    def _initialize_mps(self) -> None:
        self._torch = _load_torch()
        torch = self._torch
        sources = np.asarray(self.matrix.indices, dtype=np.int64)
        targets = np.repeat(np.arange(self.matrix.shape[0], dtype=np.int64), np.diff(self.matrix.indptr))
        self._sources = torch.from_numpy(sources).to("mps")
        self._targets = torch.from_numpy(targets).to("mps")
        self._weights = torch.from_numpy(self.matrix.data).to("mps")
        self._output = torch.empty(self.matrix.shape[0], dtype=torch.float32, device="mps")

    def _release_mps(self) -> None:
        self._sources = self._targets = self._weights = self._output = None
        if self._torch is not None:
            self._torch.mps.empty_cache()

    def _matvec_mps(self, state: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            device_state = torch.from_numpy(state).to("mps")
            self._output.zero_()
            contributions = self._weights * device_state[self._sources]
            self._output.index_add_(0, self._targets, contributions)
            # Returning CPU data synchronizes Metal and includes transfer costs.
            return self._output.cpu().numpy()

    def _benchmark_and_select(self) -> None:
        state = np.random.default_rng(731).normal(0, 0.1, self.matrix.shape[0]).astype(np.float32)
        expected = self.matrix @ state
        actual = self._matvec_mps(state)
        if not np.allclose(actual, expected, rtol=2e-4, atol=2e-5):
            raise RuntimeError("GPU and CPU graph products disagree.")
        timings = {"cpu": [], "mps": []}
        for _ in range(3):
            start = perf_counter()
            self.matrix @ state
            timings["cpu"].append((perf_counter() - start) * 1000)
            start = perf_counter()
            self._matvec_mps(state)
            timings["mps"].append((perf_counter() - start) * 1000)
        cpu_ms, mps_ms = (float(np.median(timings[name])) for name in ("cpu", "mps"))
        self._status.update(cpu_ms=round(cpu_ms, 3), mps_ms=round(mps_ms, 3))
        if mps_ms < cpu_ms:
            self._status["reason"] = "Startup graph benchmark selected Metal (including CPU/GPU activity transfers)."
        else:
            self.backend = "cpu"
            self._release_mps()
            self._status["reason"] = "Startup graph benchmark selected CPU; Metal was slower for this graph."

    def matvec(self, state: np.ndarray) -> np.ndarray:
        state = np.ascontiguousarray(state, dtype=np.float32)
        if state.shape != (self.matrix.shape[1],):
            raise ValueError(f"state must have shape ({self.matrix.shape[1]},)")
        if self.backend == "mps":
            return self._matvec_mps(state)
        return np.asarray(self.matrix @ state, dtype=np.float32)

    def update_weights(self, data_indices: np.ndarray, values: np.ndarray) -> None:
        raw_indices = np.asarray(data_indices)
        if not np.issubdtype(raw_indices.dtype, np.integer) and raw_indices.size:
            raise ValueError("weight indices must be integers")
        indices = np.ascontiguousarray(raw_indices, dtype=np.int64)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if indices.ndim != 1 or values.shape != indices.shape:
            raise ValueError("weight indices and values must be equal-length vectors")
        if indices.size == 0:
            return
        if indices.min() < 0 or indices.max() >= self.matrix.nnz:
            raise ValueError("weight index is outside the CSR data array")
        if np.unique(indices).size != indices.size:
            raise ValueError("weight update indices must be unique")
        if not np.isfinite(values).all():
            raise ValueError("weights must be finite")
        if self.backend == "mps":
            torch = self._torch
            with torch.inference_mode():
                # Metal index_copy_ scales with the full destination length:
                # ~0.5 s for 7k updates into 15m edges on an M4 Pro. index_put_
                # touches only the unique updated indices (~1 ms including
                # transfers), preserving the exact same assignment semantics.
                self._weights.index_put_(
                    (torch.from_numpy(indices).to("mps"),), torch.from_numpy(values).to("mps"),
                )
        self.matrix.data[indices] = values
