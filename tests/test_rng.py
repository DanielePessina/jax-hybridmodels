import jax.numpy as jnp
import jax.random as jr
import numpy as np

from hybridmodels.rng import fold


def test_fold_returns_jax_key_shape_and_dtype() -> None:
    key = fold(jr.PRNGKey(0), "init")
    assert key.dtype == jnp.uint32
    assert key.shape == (2,)


def test_fold_is_stable_for_same_root_and_name() -> None:
    root = jr.PRNGKey(0)
    first = fold(root, "init")
    second = fold(root, "init")
    assert jnp.array_equal(first, second)


def test_fold_distinct_names_give_distinct_keys() -> None:
    root = jr.PRNGKey(0)
    assert not jnp.array_equal(fold(root, "a"), fold(root, "b"))


def test_fold_distinct_roots_give_distinct_keys() -> None:
    assert not jnp.array_equal(fold(jr.PRNGKey(0), "x"), fold(jr.PRNGKey(1), "x"))


def test_fold_is_independent_of_call_order() -> None:
    root = jr.PRNGKey(7)
    names = ("init", "tournament", "phase_0", "phase_1", "evosax_init", "evosax_ask_0")
    forward = {name: fold(root, name) for name in names}
    reversed_keys = {name: fold(root, name) for name in reversed(names)}
    for name in names:
        assert jnp.array_equal(forward[name], reversed_keys[name])


def test_fold_pinned_bytes_anchor() -> None:
    expected = np.array([3020537540, 2946840664], dtype=np.uint32)
    actual = np.asarray(fold(jr.PRNGKey(0), "init"))
    assert actual.dtype == expected.dtype
    assert np.array_equal(actual, expected)
