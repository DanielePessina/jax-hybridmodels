# RNG: Named-Fold Keys

Reproducibility is built on named folds: every random operation derives its key from a single root `key` via [`fold(root, name)`](#fold). Reordering operations or reorganising code doesn't change the keys downstream of unchanged names — compare to `jax.random.split`, which is positional and very fragile under refactors.

Names used internally: `"init"`, `"tournament"`, `"phase_{i}"`, `"evosax_init"`, `"evosax_ask_{gen}"`. User code can fold its own names off the same root without collisions.

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
Calling ``fold(root, "init")`` always yields the same key for the same
root, and different names always yield different keys (collisions only
on CRC32 collisions across the framework's small constant set of names).

<small>[Source](https://github.com/DanielePessina/jax-hybridmodels/blob/main/src/hybridmodels/rng.py#L29)</small>
