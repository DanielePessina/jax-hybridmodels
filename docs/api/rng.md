# RNG: Named-Fold Keys

Reproducibility is built on named folds: every random operation derives its key from a single root `key` via [`fold(root, name)`](#fold). Reordering operations or reorganising code doesn't change the keys downstream of unchanged names — compare to `jax.random.split`, which is positional and very fragile under refactors.

Names used internally: `"tournament"`, `"tournament_attempt_{i}"`, `"evosax_init"`, `"evosax_ask_{gen}"`, `"evosax_tell_{gen}"`. User code can fold its own names off the same root without collisions.

## Quick links

- [`fold`](#fold)

---

<a id="fold"></a>

### `fold()`

<small>`from hybridmodels.rng import fold` &nbsp;·&nbsp; also re-exported as `hybridmodels.fold`</small>

```python
fold(root_key: 'Array', name: 'str') -> 'Array'
```

Derive a stable subkey from ``root_key`` named ``name``.

Equivalent to ``jr.fold_in(root_key, crc32(name.encode("utf-8")))``.
``fold(root, "tournament")`` always returns the same key for the same root,
and two different names return different keys unless their CRC32 values
collide, which none of the framework's fixed set of names do.

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/rng.py#L32)</small>
