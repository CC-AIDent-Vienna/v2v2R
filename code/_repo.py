"""Where the repo root is, and where the code groups are.

Every module under code/ is flat-imported by name -- `import postprocess_pred`,
not `from pipeline.postprocess import postprocess_pred`. That is on purpose: it
keeps `python3 code/<group>/<module>.py` working as a plain script invocation,
which is how every job script and every doc example calls this code, and it
means the grouping below can be rearranged without touching a single import.

The cost is that sys.path has to be told where the groups are, which is what
add_code_paths() does. Call it once, at the top of any module that imports a
sibling from another group.

REPO_ROOT is resolved from this file's own location rather than counted in
`.parent.parent` hops, so nothing breaks the next time a file moves. That
matters more than it sounds: the hop-counting version failed SILENTLY -- a
default path pointed one directory too shallow and only surfaced when someone
omitted the flag that normally overrides it.
"""

from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent

#: Directories under code/ that hold flat-importable modules.
GROUPS = (
    "pipeline/preprocess",
    "pipeline/infer",
    "pipeline/postprocess",
    "ground_truth",
    "eval",
    "train",
)

DEFAULT_SCHEMA = REPO_ROOT / "schema" / "schema.json"


def add_code_paths() -> None:
    """Put every code group on sys.path, nearest-first, without duplicates."""
    for group in GROUPS:
        d = CODE_ROOT / group
        if d.is_dir():
            s = str(d)
            if s not in sys.path:
                sys.path.insert(0, s)


def check() -> None:
    """Fail loudly if the tree is not shaped the way REPO_ROOT assumes."""
    if not DEFAULT_SCHEMA.is_file():
        raise RuntimeError(
            f"repo root resolved to {REPO_ROOT}, which has no "
            f"schema/schema.json -- code/_repo.py has been moved out of code/")
