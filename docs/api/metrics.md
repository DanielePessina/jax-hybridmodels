# Metrics: Per-Channel Evaluation

Per-channel evaluation of trained predictions, mirroring the loss mask discipline: only cells the bucket mask marks as real measurements count. `compute_metrics(predictions, dataset)` returns one `ChannelMetrics` per output channel with MSE, RMSE, MAE and R^2; `print_metrics` renders them as a table.

## Quick links

- [`ChannelMetrics`](#channelmetrics)
- [`compute_metrics`](#compute_metrics)
- [`print_metrics`](#print_metrics)

---

<a id="channelmetrics"></a>

### `ChannelMetrics`

<small>`from hybridmodels.metrics import ChannelMetrics` &nbsp;·&nbsp; also re-exported as `hybridmodels.ChannelMetrics`</small>

```python
ChannelMetrics(
    name: 'str',
    n: "Int[Array, '']",
    mse: "Float[Array, '']",
    rmse: "Float[Array, '']",
    mae: "Float[Array, '']",
    r2: "Float[Array, '']",
) -> None
```

Metrics for a single output channel.

**Attributes**

| Field | Type | Description |
| --- | --- | --- |
| `name` | `str` | Channel name from ``dataset.output_channel_names``. |
| `n` | `Int[Array, ""]` | Scalar number of observed (mask=True) cells behind the stats. It is a JAX scalar so the complete metrics result can pass through ``jit``. |
| `mse, rmse, mae` | `Float[Array, ""]` | Error of ``predicted - observed`` over the masked cells. |
| `r2` | `Float[Array, ""]` | ``1 - SS_res/SS_tot``; ``nan`` when the observations are constant and ``SS_tot`` is zero. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/metrics.py#L33)</small>

---

<a id="compute_metrics"></a>

### `compute_metrics()`

<small>`from hybridmodels.metrics import compute_metrics` &nbsp;·&nbsp; also re-exported as `hybridmodels.compute_metrics`</small>

```python
compute_metrics(
    predictions: "tuple[Float[Array, 'N T D'], ...]",
    dataset: 'Dataset',
) -> dict[str, ChannelMetrics]
```

Collapse bucketed predictions into per-channel summary stats.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictions` | `tuple of arrays, one per bucket` | From :func:`hybridmodels.prediction.predict_dataset`; each entry is ``[N_b, T_b, D]`` matching its ``BucketPayload``. |
| `dataset` | `Dataset` | The dataset that produced ``predictions``. Read for masks, observations, and channel names. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `dict[str, ChannelMetrics]` |  | Keyed by channel, in ``dataset.output_channel_names`` order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/metrics.py#L64)</small>

---

<a id="print_metrics"></a>

### `print_metrics()`

<small>`from hybridmodels.metrics import print_metrics` &nbsp;·&nbsp; also re-exported as `hybridmodels.print_metrics`</small>

```python
print_metrics(
    metrics: 'dict[str, ChannelMetrics]',
    header: 'str | None' = None,
) -> None
```

Print a metrics table, one row per channel.

Format::

    {header}
      channel        n          MSE         RMSE          MAE       R^2
      conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
      d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

Scientific notation throughout, so one template stays readable across
the example suite's scales, from ``omega ~ O(1)`` to nucleation rates
spanning nine decades.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/metrics.py#L146)</small>
