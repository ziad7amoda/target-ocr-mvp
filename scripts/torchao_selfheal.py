"""Self-heals a recurring Colab dependency conflict (see notebook cell 3).

Colab preinstalls torchao 0.10.0. Nothing in this project uses torchao, but
loading any LoRA-adapter model (e.g. NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct)
makes transformers hand off to peft, whose LoRA loader walks a list of layer
dispatchers. One of them, dispatch_torchao, calls is_torchao_available() -
which *raises* ImportError instead of returning False when torchao is
present but older than the version peft requires. That kills the model load
even though the dispatcher is completely irrelevant to the model being
loaded. Uninstalling torchao makes the availability check short-circuit on
"not installed" and return False, so the dispatcher is skipped.

This module holds the pure, unit-testable decision logic. The notebook cell
does the actual I/O (reading the installed version, running `pip uninstall`,
printing status) and imports these functions rather than reimplementing them
inline.
"""

from __future__ import annotations

import re

# peft requires torchao >0.16.0; anything below this is the broken case.
MIN_TORCHAO_VERSION: tuple[int, ...] = (0, 16)


def parse_version_prefix(version: str) -> tuple[int, ...]:
    """Extract the leading dotted numeric prefix of a version string as ints.

    Version strings in the wild carry suffixes plain int-splitting chokes on,
    e.g. "0.10.0+cu121" (local/build metadata) or "0.16.0.dev123" (a dev
    release). This walks components split on '.', '+', and '-', keeping only
    a leading run of pure-digit components and stopping at the first one
    that isn't (or at the end of the string).

    >>> parse_version_prefix("0.10.0+cu121")
    (0, 10, 0)
    >>> parse_version_prefix("0.16.0.dev123")
    (0, 16, 0)
    >>> parse_version_prefix("not-a-version")
    ()
    """
    nums: list[int] = []
    for part in re.split(r"[.+-]", version):
        if not re.fullmatch(r"\d+", part):
            break
        nums.append(int(part))
    return tuple(nums)


def is_version_below(version: str, threshold: tuple[int, ...] = MIN_TORCHAO_VERSION) -> bool:
    """True if `version` is numerically below `threshold` (default 0.16).

    Comparison is numeric and component-wise (so "0.9.0" < "0.16.0" even
    though that is false as a string comparison), with the shorter tuple
    zero-padded to the longer one's length. A version string with no
    parseable leading numeric prefix is treated conservatively as "too old",
    so the caller still attempts the uninstall rather than silently doing
    nothing.
    """
    parsed = parse_version_prefix(version)
    if not parsed:
        return True
    length = max(len(parsed), len(threshold))
    padded_parsed = parsed + (0,) * (length - len(parsed))
    padded_threshold = threshold + (0,) * (length - len(threshold))
    return padded_parsed < padded_threshold


def get_installed_version(package: str = "torchao") -> str | None:
    """Return the installed version string of `package`, or None if absent.

    Uses importlib.metadata only - never imports the package itself, so this
    is safe to call even for packages (like torchao) we don't want loaded.

    Caches are invalidated first. importlib.metadata memoises its filesystem
    scan, and this function is called again immediately after a `pip
    uninstall` in the same process to check whether the uninstall actually
    took effect - a stale cache there would report the package as still
    present and send the user chasing a problem that no longer exists.
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, version

    importlib.invalidate_caches()
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def uninstall_commands(executable: str) -> list[list[str]]:
    """Commands to try, in order, to remove `torchao`.

    Two of them, because `pip uninstall` exits 0 when the package is not
    installed in the environment that particular pip manages:

        $ pip uninstall -y torchao
        WARNING: Skipping torchao as it is not installed.
        $ echo $?
        0

    A zero exit therefore proves nothing, and the caller MUST re-check with
    get_installed_version() rather than trusting the return code. On Colab
    the kernel's sys.executable and the `pip` on PATH do not reliably
    resolve to the same site-packages, which is precisely the case where the
    first command reports success and removes nothing.
    """
    return [
        [executable, "-m", "pip", "uninstall", "-y", "torchao"],
        ["pip", "uninstall", "-y", "torchao"],
    ]


def decide_action(installed_version: str | None) -> str:
    """Decide what the notebook cell should do, without touching pip.

    Returns one of:
      "uninstall"    - installed and below MIN_TORCHAO_VERSION
      "skip-absent"  - not installed
      "skip-ok"      - installed and already >= MIN_TORCHAO_VERSION
    """
    if installed_version is None:
        return "skip-absent"
    if is_version_below(installed_version):
        return "uninstall"
    return "skip-ok"
