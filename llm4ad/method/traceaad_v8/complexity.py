"""V8 uses the same deterministic source metrics as TraceAAD V5."""

from ..traceaad_v5.complexity import (
    code_change_ratio,
    code_hash,
    nonempty_loc,
    normalized_source,
)

__all__ = ["code_change_ratio", "code_hash", "nonempty_loc", "normalized_source"]
