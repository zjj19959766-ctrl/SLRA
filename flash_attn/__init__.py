"""Local shim for environments with a broken flash-attn install.

This repo benchmarks training with eager attention, so these symbols should
never actually be called. They only need to exist so Transformers can import
its LLaMA modeling module without crashing on a missing CUDA runtime from a
third-party flash-attn wheel.
"""

__all__ = ["flash_attn_func", "flash_attn_varlen_func"]


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "flash_attn shim was invoked unexpectedly. This benchmark is expected "
        "to run with eager attention, not flash-attn."
    )


flash_attn_func = _unsupported
flash_attn_varlen_func = _unsupported
