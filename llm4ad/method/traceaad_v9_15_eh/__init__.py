"""TraceAAD V9.15 with bounded error feedback and one repair attempt."""

from ..traceaad_v9_15 import RunArtifacts, TraceAADV915


class TraceAADV915EH(TraceAADV915):
    """V9.15 baseline with error-aware generation, without allocation changes."""

    def __init__(self, *args, error_retries: int = 1, **kwargs):
        kwargs.setdefault("error_retries", error_retries)
        kwargs.setdefault("error_handling", True)
        super().__init__(*args, **kwargs)


__all__ = ["RunArtifacts", "TraceAADV915EH"]
