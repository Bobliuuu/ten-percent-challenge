"""Local stand-in for `modal` so tcq_run.py / vq8_run.py _impl() run on a plain GPU box (Vast). /vol is a plain directory."""
class _Any:
    def __init__(s, *a, **k): pass
    def __getattr__(s, n): return _Any()
    def __call__(s, *a, **k): return _Any()
    def commit(s): pass
    def reload(s): pass
class App(_Any):
    def function(s, *a, **k): return lambda f: f
class Volume:
    @staticmethod
    def from_name(*a, **k): return _Any()
Image = _Any()
