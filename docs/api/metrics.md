# Metrics: Per-Channel Evaluation

Per-channel evaluation of trained predictions, mirroring the loss mask discipline: only cells the bucket mask marks as real measurements count. `compute_metrics(predictions, dataset)` returns one `ChannelMetrics` per output channel with MSE, RMSE, MAE and R^2; `print_metrics` renders them as a table.

## Quick links

- [`ChannelMetrics`](#channelmetrics)
- [`compute_metrics`](#compute_metrics)
- [`print_metrics`](#print_metrics)

---

<a id="channelmetrics"></a>

### `ChannelMetrics`

<small>`from jaxhybridmodels.metrics import ChannelMetrics` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.ChannelMetrics`</small>

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
| `name` | `str` | Channel name, one of the model's ``output_channel_names``. |
| `n` | `Int[Array, ""]` | Scalar number of observed (mask=True) cells behind the stats. It is a JAX scalar so the complete metrics result can pass through ``jit``. |
| `mse, rmse, mae` | `Float[Array, ""]` | Error of ``predicted - observed`` over the masked cells. |
| `r2` | `Float[Array, ""]` | ``1 - SS_res/SS_tot``; ``nan`` when the observations are constant and ``SS_tot`` is zero. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/metrics.py#L36)</small>

---

<a id="compute_metrics"></a>

### `compute_metrics()`

<small>`from jaxhybridmodels.metrics import compute_metrics` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.compute_metrics`</small>

```python
compute_metrics(predictions: 'Mapping[str, Any]') -> 'dict[str, ChannelMetrics]'
```

Per-channel summary stats of a prediction against the data it was made on.

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `predictions` | `PredictionResult` | From ``HybridModel.predict`` or ``PredictionResult.from_buckets``, or any mapping from experiment ID to a record with ``outputs``, ``observed`` and ``observation_mask`` dicts keyed by channel. Only cells the mask marks as measured count. Channels are read from the first experiment's ``outputs``. |

**Returns**

| Item | Type | Description |
| --- | --- | --- |
| `dict[str, ChannelMetrics]` |  | Keyed by channel, in the model's ``output_channel_names`` order. |

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/metrics.py#L67)</small>

---

<a id="print_metrics"></a>

### `print_metrics()`

<small>`from jaxhybridmodels.metrics import print_metrics` &nbsp;·&nbsp; also re-exported as `jaxhybridmodels.print_metrics`</small>

```python
print_metrics(metrics: 'Mapping[str, Any]', *, header: 'str | None' = None) -> 'None'
```

Print a metrics table, one row per channel.

Takes the output of [`compute_metrics`](/api/metrics#compute_metrics), or a ``PredictionResult``,
which is passed through [`compute_metrics`](/api/metrics#compute_metrics) first.

Format::

    {header}
      channel        n          MSE         RMSE          MAE       R^2
      conc          42   1.2345e-03   3.5135e-02   2.7012e-02    0.9876
      d43           17   1.4321e+00   1.1967e+00   8.9120e-01    0.4231

Scientific notation throughout, so one template stays readable across
the example suite's scales, from ``omega ~ O(1)`` to nucleation rates
spanning nine decades.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/jaxhybridmodels/metrics.py#L133)</small>
