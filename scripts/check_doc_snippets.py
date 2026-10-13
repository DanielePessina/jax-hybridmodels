"""Check that every ``<<<`` code include in the docs resolves.

VitePress fails soft on snippet includes: a missing file renders as a line of
text, and a region name that is not in the file silently includes the whole
file. This script makes both a hard error, and also flags regions defined in
an example script that no page includes (usually a renamed include).

Regions are marked in Python as ``# region name`` / ``# endregion name`` and
included with ``<<< @/../examples/<dir>/<file>.py#name``.

Run::

    uv run python scripts/check_doc_snippets.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

INCLUDE = re.compile(r"^<<<\s+(\S+?)(?:#([\w*-]+))?(?:\{[^}]*\})?(?:\s|$)", re.MULTILINE)
REGION_START = re.compile(r"^\s*# ?region ([\w*-]+)\s*$", re.MULTILINE)
REGION_END = re.compile(r"^\s*# ?endregion ([\w*-]+)\s*$", re.MULTILINE)


def _resolve(page: Path, raw: str) -> Path:
    # Same rule as VitePress: a leading @ means the docs source root.
    if raw.startswith("@"):
        return (DOCS / raw[1:].lstrip("/")).resolve()
    return (page.parent / raw).resolve()


def main() -> int:
    errors: list[str] = []
    used: dict[Path, set[str]] = {}
    for page in sorted(DOCS.rglob("*.md")):
        if "node_modules" in page.parts or ".vitepress" in page.parts:
            continue
        for match in INCLUDE.finditer(page.read_text()):
            raw, region = match.group(1), match.group(2)
            target = _resolve(page, raw)
            where = f"{page.relative_to(ROOT)}: <<< {raw}" + (f"#{region}" if region else "")
            if not target.is_file():
                errors.append(f"{where}: file not found")
                continue
            if region is None:
                continue
            text = target.read_text()
            starts = REGION_START.findall(text)
            ends = REGION_END.findall(text)
            if region not in starts:
                errors.append(f"{where}: region not in {target.relative_to(ROOT)}")
            elif region not in ends:
                errors.append(f"{where}: region has no endregion")
            used.setdefault(target, set()).add(region)

    for script in sorted((ROOT / "examples").rglob("*.py")):
        defined = set(REGION_START.findall(script.read_text()))
        for region in sorted(defined - used.get(script.resolve(), set())):
            name = script.relative_to(ROOT)
            errors.append(f"{name}: region {region!r} is not included by any page")

    for error in errors:
        print(f"error: {error}")
    if not errors:
        print("Doc snippets resolve.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
