"""Python-level network guard (fallback isolation when the OS-level netns is unavailable). Loaded via PYTHONPATH.
Any socket connect / DNS lookup raises, and the attempt is appended to $NETGUARD_LOG."""
import os, socket
_log = os.environ.get("NETGUARD_LOG")
def _rec(what, a):
    if _log:
        try: open(_log, "a").write(f"{what} {a!r}\n")
        except Exception: pass
def _blocked(what):
    def f(*a, **k):
        _rec(what, a[1:] if what == "connect" else a); raise OSError(101, f"network blocked by netguard ({what})")
    return f
socket.socket.connect = _blocked("connect"); socket.socket.connect_ex = _blocked("connect_ex")
socket.getaddrinfo = _blocked("getaddrinfo"); socket.gethostbyname = _blocked("gethostbyname")
socket.create_connection = _blocked("create_connection")
