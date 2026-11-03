"""The text behind ``FitResult.summary`` and ``SavedRun.summary``.

Kept out of :mod:`jaxhybridmodels.results`, which the trainers import, because
this module reads predictors and penalties. Nothing here changes a result: it
only reports what the run already holds, plus a forward pass over each
network's input box and, when asked, a prediction on the data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jaxhybridmodels.metrics import _format_metrics, _worst_first, compute_metrics
from jaxhybridmodels.penalties import _bounded_leaves_with_paths, _latents_at, box_grid
from jaxhybridmodels.predictors import Parameter
from jaxhybridmodels.predictors.base import _is_parameter
from jaxhybridmodels.results import FitHistory
from jaxhybridmodels.trainable import _path_to_dotted, count_trainable_params

_STOP_TEXT = {
    "budget_exhausted": "ran its full budget",
    "patience": "stopped early on patience",
    "diverged": "diverged; the model holds the best parameters before the failure",
}

# The sweep of a network's input box is a tensor product, so its size is
# capped: 9 points per axis in low dimensions, and fewer as inputs are added.
_MAX_POINTS_PER_AXIS = 9
_MAX_BOX_POINTS = 4096


_MIN_TREND_WINDOW = 2


def format_fit_summary(run: Any, *, data: Any = None) -> str:
    """Build the report for a ``FitResult`` or ``SavedRun``; see ``FitResult.summary``."""
    lines = _run_lines(run.history, run.best_step, run.stop_reason)
    lines += _phase_lines(run.history)
    lines += _parameter_lines(run.model.predictors)
    lines += _network_lines(run.model.predictors)
    if data is not None:
        lines += _data_lines(run.model.predict(data))
    return "\n".join(lines)


def _run_lines(history: FitHistory, best_step: int | None, stop_reason: str | None) -> list[str]:
    if not history.loss:
        return ["Run: no fit recorded"]
    # Only Optax records a phase per step; Evosax leaves it empty.
    unit = "steps" if history.phase else "generations"
    returned = "the last one" if best_step is None else f"{unit[:-1]} {best_step}"
    reason = _STOP_TEXT.get(stop_reason or "", str(stop_reason))
    at = -1 if best_step is None else best_step
    lines = [
        f"Run: {reason}",
        f"  {len(history.loss)} {unit} recorded; returned {returned}",
        f"  loss {history.loss[0]:.4e} at the start, {history.loss[at]:.4e} at the returned point",
    ]
    trend = _loss_trend(history)
    if trend is not None:
        lines.append(f"  loss change over the last 10% of {unit}: {trend:+.2%}")
    if history.validation_loss:
        lines.append(f"  validation loss {history.validation_loss[at]:.4e} at the returned point")
    if any(history.penalty):
        lines.append(f"  bound penalty {history.penalty[at]:.4e} at the returned point")
    failed = sum(history.failed_candidates)
    if failed:
        generations = sum(1 for n in history.failed_candidates if n)
        lines.append(
            f"  {failed} candidates failed to solve, in {generations} of "
            f"{len(history.failed_candidates)} generations, and ranked last"
        )
    if history.tournament_scores:
        lines.append(f"  {_tournament_line(history.tournament_scores)}")
    return lines


def _tournament_line(scores: Sequence[float], *, kept: int | None = None) -> str:
    """One line on a tournament: its size, best and worst score, and failures.

    With ``kept``, the line also names the attempts kept, numbered from 1 in
    the order the tournament ranks them (ties to the earlier attempt). The
    trainer reports that as the tournament ends. A saved history does not
    record which attempt a seed-ensemble member came from, so the summary
    leaves it out.
    """
    finite = [s for s in scores if math.isfinite(s)]
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    if kept is None:
        line = f"tournament: {len(scores)} attempts"
    else:
        which = "attempt" if kept == 1 else "attempts"
        named = ", ".join(str(i + 1) for i in order[:kept])
        line = f"tournament: kept {which} {named} of {len(scores)}"
    if not finite:
        # The trainer falls back to the starting predictors then.
        return f"tournament: all {len(scores)} attempts failed, kept the starting values"
    line += f", best score {scores[order[0]]:.4g}, worst {max(finite):.4g}"
    failed = len(scores) - len(finite)
    return line + (f", {failed} failed" if failed else "")


def _loss_trend(history: FitHistory) -> float | None:
    """Relative change of the median loss between the last two tenths of the final phase.

    Budget exhaustion says nothing about whether a fit had settled, and this
    shows if it was still moving. Medians, because a raw loss can jump from one
    step to the next. Only the final phase, because a new horizon changes what
    the loss measures. None when there are too few steps to have two windows.
    """
    final = history.phase[-1] if history.phase else None
    losses = [
        loss
        for i, loss in enumerate(history.loss)
        if not history.phase or history.phase[i] == final
    ]
    window = len(losses) // 10
    if window < _MIN_TREND_WINDOW:
        return None
    before = float(np.median(losses[-2 * window : -window]))
    after = float(np.median(losses[-window:]))
    if before == 0.0 or not (math.isfinite(before) and math.isfinite(after)):
        return None
    return (after - before) / abs(before)


def _phase_lines(history: FitHistory) -> list[str]:
    """The loss at the start and end of each phase, and the gap term where segmented.

    A segmented phase's loss is the data term on the assembled segments, so
    the table adds the gap term at the phase's last step: how far its
    segments were from joining up when it handed over.
    """
    phases = sorted(set(history.phase))
    if len(phases) < 2:
        return []
    gaps = history.continuity if len(history.continuity) == len(history.loss) else ()
    header = f"  {'phase':>5} {'steps':>6} {'first loss':>13} {'last loss':>13}"
    lines = ["Phases", header + (f" {'last gap term':>14}" if gaps else "")]
    for phase in phases:
        steps = [i for i, owner in enumerate(history.phase) if owner == phase]
        first, last = history.loss[steps[0]], history.loss[steps[-1]]
        line = f"  {phase:>5} {len(steps):>6} {first:>13.4e} {last:>13.4e}"
        if gaps:
            gap = gaps[steps[-1]]
            line += f" {gap:>14.4e}" if any(gaps[i] for i in steps) else f" {'-':>14}"
        lines.append(line)
    return lines


def _parameter_lines(predictors: Any) -> list[str]:
    rows: list[tuple[str, str, str, str]] = []
    for path, leaf in jtu.tree_flatten_with_path(predictors, is_leaf=_is_parameter)[0]:
        if _is_parameter(leaf):
            rows.extend(_parameter_rows(_path_to_dotted(path), leaf))
    if not rows:
        return []
    width = max(len("parameter"), *(len(row[0]) for row in rows))
    lines = ["Parameters", f"  {'parameter':<{width}} {'value':>12}  bounds"]
    for name, value, bounds, note in rows:
        lines.append(f"  {name:<{width}} {value:>12}  {bounds}{note}".rstrip())
    return lines


def _parameter_rows(path: str, parameter: Parameter) -> list[tuple[str, str, str, str]]:
    """One row per element, flagging an element past its squash's knee.

    A boxed element is flagged past the knee on either side. A one-sided one
    is flagged when its latent is below ``-knee`` times its temperature,
    within about 5% of the temperature of its edge: a note to look, not a
    charge, since a one-sided element is log-like there and still trains.
    """
    scaler = parameter.scaler
    latent = jnp.ravel(parameter.latent)
    values = jnp.ravel(scaler.from_latent(parameter.latent))
    # The middle of the latent range decides the side, whatever the warp or squash.
    midpoints = jnp.ravel(scaler.from_latent(jnp.zeros_like(parameter.latent)))
    temperature = jnp.broadcast_to(scaler.temperature, latent.shape)
    # A warped one-sided element is exp-like all the way to its edge, never stuck.
    one_sided = jnp.array([math.isfinite(lo) != math.isfinite(hi) for lo, hi in scaler.bounds])
    if scaler.warp != "linear":
        one_sided = jnp.zeros_like(one_sided)
    pinned = jnp.where(
        one_sided,
        -latent / temperature > scaler.z_knee,
        scaler._saturation_terms(latent) > 0.0,
    )
    rows = []
    for i, (low, high) in enumerate(scaler.bounds):
        note = ""
        if pinned[i]:
            note = f"  near {'upper' if values[i] > midpoints[i] else 'lower'} bound"
        name = path if latent.size == 1 else f"{path}[{i}]"
        rows.append((name, f"{float(values[i]):.5g}", f"({low:g}, {high:g})", note))
    return rows


def _network_lines(predictors: Any) -> list[str]:
    leaves = _bounded_leaves_with_paths(predictors)
    if not leaves:
        return []
    width = max(len("network"), *(len(path) for path, _ in leaves))
    lines = ["Networks", f"  {'network':<{width}} {'params':>8}  output saturated"]
    for path, leaf in leaves:
        lines.append(
            f"  {path:<{width}} {count_trainable_params(leaf):>8}  {_saturated_share(leaf)}"
        )
    return lines


def _saturated_share(leaf: Any) -> str:
    """The share of a network's input box where any output is past the squash's knee."""
    n_inputs = len(leaf.in_scaler.bounds)
    if any(not all(map(math.isfinite, pair)) for pair in leaf.in_scaler.bounds):
        return "not checked: an input box is open"
    per_axis = next(
        (p for p in range(_MAX_POINTS_PER_AXIS, 1, -1) if p**n_inputs <= _MAX_BOX_POINTS), None
    )
    if per_axis is None:
        return f"not checked: {n_inputs} inputs is too many to sweep"
    latents = _latents_at(leaf, box_grid(leaf.in_scaler, n_per_dim=per_axis))
    share = float(jnp.mean(jax.vmap(leaf.out_scaler.saturation)(latents) > 0.0))
    return f"{share:.1%} of the input box"


def _data_lines(predictions: Any) -> list[str]:
    lines = ["Metrics", *_format_metrics(compute_metrics(predictions))]
    if len(predictions) < 2:
        return lines
    by_experiment = compute_metrics(predictions, by="experiment")
    lines.append("Worst experiment per channel, by RMSE")
    for channel in next(iter(by_experiment.values())):
        rows = _worst_first(by_experiment, channel)
        if rows:
            exp_id, worst = rows[0]
            lines.append(f"  {channel}: {exp_id} ({float(worst.rmse):.4e})")
    return lines
