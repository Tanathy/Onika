import sys
from contextlib import contextmanager

_LOG_SINK = None


def set_log_sink(sink):
    """Optional callable sink(line: str) to mirror console output."""
    global _LOG_SINK
    _LOG_SINK = sink


def _emit(line: str) -> None:
    if _LOG_SINK is None:
        return
    try:
        _LOG_SINK(line)
    except Exception:
        pass

def info(msg: str):
    line = f"[INFO] {msg}"
    print(line)
    _emit(line)

def warning(msg: str):
    line = f"[WARNING] {msg}"
    print(line, file=sys.stderr)
    _emit(line)

def error(msg: str):
    line = f"[ERROR] {msg}"
    print(line, file=sys.stderr)
    _emit(line)

def success(msg: str):
    line = f"[SUCCESS] {msg}"
    print(line)
    _emit(line)

@contextmanager
def progress_context(msg: str = None, total: int = None, desc: str = None, unit: str = None, **kwargs):
    description = desc or msg or "Processing"
    print(f"[START] {description}...")
    
    class DummyProgressBar:
        def update(self, n=1):
            pass
            
    try:
        yield DummyProgressBar()
    finally:
        print(f"[END] {description}")
