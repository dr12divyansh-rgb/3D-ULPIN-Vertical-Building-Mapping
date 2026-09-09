"""Locate the Blender executable and run a Blender Python job headlessly."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_blender(explicit=None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Blender executable not found: {explicit}")

    # Common install locations (Windows).
    candidates = []
    pf = Path("C:/Program Files/Blender Foundation")
    if pf.exists():
        candidates.extend(sorted(pf.glob("*/blender.exe"), reverse=True))

    which = shutil.which("blender")
    if which:
        candidates.append(Path(which))

    for c in candidates:
        if Path(c).exists():
            return Path(c)

    raise FileNotFoundError(
        "Blender not found. Install Blender (e.g. 'winget install "
        "BlenderFoundation.Blender.LTS.4.2') or set blender.executable in config."
    )


def run_blender_script(blender: Path, script: Path, args, timeout=None):
    """Run ``blender --background --python <script> -- <args...>``.

    Returns a CompletedProcess. ``args`` is a list of strings forwarded to the
    script after the ``--`` separator.
    """
    cmd = [str(blender), "--background", "--factory-startup",
           "--python", str(script), "--"] + [str(a) for a in args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
