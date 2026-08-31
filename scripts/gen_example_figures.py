"""Regenerate the example figures committed under ``docs/examples/assets/``.

The VitePress pages under ``docs/examples/`` embed the figures the example
scripts produce, so those PNGs are committed. This script reruns every
example with ``--plot-dir`` pointed into ``docs/examples/assets/<scenario>/``
to keep the committed figures in sync with the scripts.

The batch-reactor RL example needs the frozen trunk artefact that
``train_hybrid.py --save-predictors`` writes, so this script produces it
first (with ``--no-plot``; the artefact directory is gitignored).

Run::

    uv run python scripts/gen_example_figures.py            # full training budgets
    uv run python scripts/gen_example_figures.py --quick    # reduced budgets, for iteration
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "examples" / "assets"
TRUNK_ARTEFACT = ROOT / "examples" / "batch_reactor" / "artefacts" / "trunk_fresh.eqx"

# (script, scenario assets subdirectory). The plot filenames are fixed by the
# scripts; only the destination directory changes.
EXAMPLES: tuple[tuple[str, str], ...] = (
    ("examples/custom_loop/train_custom_loop.py", "custom-loop"),
    ("examples/supersaturation_poly/train_supersaturation_poly.py", "supersaturation-poly"),
    ("examples/custom_predictor/train_custom_predictor.py", "custom-predictor"),
    ("examples/hybrid_ode/train_hybrid_ode.py", "hybrid-ode"),
    ("examples/sbml_hybrid/train_sbml_hybrid.py", "sbml-hybrid"),
    ("examples/crystallisation/train_kinetic.py", "crystallisation"),
    ("examples/batch_reactor/train_hybrid.py", "batch-reactor"),
    ("examples/batch_reactor/train_rl_deactivation.py", "batch-reactor-rl"),
)

# Reduced budgets for --quick: small enough to iterate, large enough that the
# figures still read.
QUICK_ARGS: dict[str, tuple[str, ...]] = {
    "examples/crystallisation/train_kinetic.py": ("--steps", "120"),
    "examples/hybrid_ode/train_hybrid_ode.py": ("--steps", "100"),
    "examples/custom_predictor/train_custom_predictor.py": ("--steps", "150"),
    "examples/batch_reactor/train_hybrid.py": ("--num-generations", "20", "--steps", "200"),
    "examples/batch_reactor/train_rl_deactivation.py": (
        "--updates",
        "120",
        "--rollouts",
        "16",
        "--baseline-steps",
        "150",
    ),
}


def _run(*args: str) -> None:
    print(f"\n>>> {Path(args[0]).name} " + " ".join(f"'{a}'" for a in args[1:]))
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{args[0]} failed with exit code {result.returncode}")
    print(f"<<< {Path(args[0]).name} done in {time.monotonic() - t0:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced training budgets (see QUICK_ARGS); the committed figures "
        "should be regenerated with full budgets.",
    )
    args = parser.parse_args()

    for script, scenario in EXAMPLES:
        plot_dir = ASSETS / scenario
        plot_dir.mkdir(parents=True, exist_ok=True)

        extra: tuple[str, ...] = QUICK_ARGS.get(script, ()) if args.quick else ()
        if script == "examples/batch_reactor/train_hybrid.py":
            # The RL example loads this artefact as its frozen trunk; produce
            # it first (plots off), then run the script again with plotting on
            # so the two figures are generated from the same fit.
            trunk_args = ("--no-plot", "--save-predictors", str(TRUNK_ARTEFACT))
            _run(script, "--plot-dir", str(plot_dir), *trunk_args, *extra)
            _run(script, "--plot-dir", str(plot_dir), *extra)
        else:
            _run(script, "--plot-dir", str(plot_dir), *extra)

    print("\nAll example figures regenerated under", ASSETS)


if __name__ == "__main__":
    main()