import os
import sys


def configure_console_encoding():
    """Keep accented French logs readable on Windows terminals."""
    if os.name == "nt":
        os.system("chcp 65001 > nul")

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
