"""Resource metrics shared by the training benchmarks."""

import jax


def peak_gpu_memory_bytes() -> int | None:
    """Peak JAX allocation since process start; unavailable on CPU.

    Read after training/evaluation has completed, in a fresh process per run.
    This includes resident data, model, optimiser and temporary buffers.
    """
    device = jax.devices()[0]
    if device.platform != "gpu":
        return None
    try:
        stats = device.memory_stats() or {}
    except (RuntimeError, NotImplementedError):
        return None
    peak = stats.get("peak_bytes_in_use")
    return int(peak) if peak is not None else None
