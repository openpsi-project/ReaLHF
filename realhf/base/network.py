import socket
from contextlib import closing


def find_free_port():
    """From, stackoverflow Issue 1365265."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def gethostname():
    return socket.gethostname()


def gethostip():
    return socket.gethostbyname(socket.gethostname())
