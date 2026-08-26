# Bucketed-irregular is the only data interface

The framework supports exactly one in-memory data shape: experiments grouped into buckets by `len(union_ts)`, each bucket a `NamedTuple` of stacked `[N, T, ...]` arrays. The source package's parallel `UnscaledBatchedExperiments` (padded-to-T_max) pathway is dropped. Users load their own raw data into `Experiment` objects (per-channel sparse), and `make_dataset` does the union/mask/bucket work once.

## Why this is non-obvious

A future reader may notice that "padded everything to T_max" is the simpler implementation and try to reintroduce it as an alternative. Don't. The padded pathway forces every loss/sim function to handle masks anyway (because variable trajectory lengths exist), wastes compute on padding, and gives users two containers to choose between. The bucketed pathway is sufficient and uniform.

## Considered alternatives

- Padded + bucketed (both supported). Rejected for the extra public API and maintenance cost; users were never sure which to pick.
- A single global pad to `T_max`. Rejected for compute waste on jagged data.
- Per-experiment vmap (no bucketing). Rejected because each experiment shape would re-trace.
