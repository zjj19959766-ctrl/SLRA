"""Local shim for Transformers' optional flash-attn imports.

These helpers should never be executed in our eager-attention benchmark path.
They only exist to satisfy import-time references from Transformers.
"""


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "flash_attn.bert_padding shim was invoked unexpectedly. "
        "This benchmark is expected to run with eager attention."
    )


index_first_axis = _unsupported
pad_input = _unsupported
unpad_input = _unsupported
