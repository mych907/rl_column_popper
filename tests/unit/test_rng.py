import numpy as np


def test_make_rng_and_determinism():
    from column_popper.utils.rng import make_rng

    g1 = make_rng(123)
    g2 = make_rng(123)
    a = g1.integers(0, 1_000_000, size=10)
    b = g2.integers(0, 1_000_000, size=10)
    np.testing.assert_array_equal(a, b)


def test_rng_pool_different_keys_but_stable():
    from column_popper.utils.rng import RngPool

    pool1 = RngPool(seed=999)
    pool2 = RngPool(seed=999)
    g1a = pool1.get("board")
    g1b = pool1.get("env")
    g2a = pool2.get("board")
    g2b = pool2.get("env")

    a1 = g1a.integers(0, 1000, size=5)
    a2 = g2a.integers(0, 1000, size=5)
    b1 = g1b.integers(0, 1000, size=5)
    b2 = g2b.integers(0, 1000, size=5)

    # Same key sequences match across pools with same root seed
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    # Different keys should produce different sequences (very high probability)
    assert not np.array_equal(a1, b1)


def test_split_generators_independence():
    from column_popper.utils.rng import split_generators

    gens = split_generators(12345, 3)
    seqs = [g.integers(0, 1000, size=5) for g in gens]
    assert len(seqs) == 3
    assert not np.array_equal(seqs[0], seqs[1])
    assert not np.array_equal(seqs[1], seqs[2])
    assert not np.array_equal(seqs[0], seqs[2])

