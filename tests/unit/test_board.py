import numpy as np


def test_board_spawn_constraints_and_pops():
    """
    Board rules from data-model/spec:
    - Grid is 8x3 int32, 0=empty, values from number_pool
    - Spawn generator avoids creating an immediate three-in-a-column from two
      identical cells beneath
    - Pops only remove vertical triples and only in the acted column
    """
    from column_popper.core.board import Board

    rng_seed = 123
    board = Board(height=12, width=3, number_pool=(1, 2, 3), seed=rng_seed)
    assert board.grid.shape == (12, 3)
    assert board.grid.dtype == np.int32

    # Fill a column with a pattern and test pop logic
    col = 1
    # Clear and set specific values bottom-up
    board.grid[:, col] = 0
    board.grid[-3:, col] = [2, 2, 2]

    # Act on this column to trigger pop removal
    popped = board.pop_triples_in_column(col)
    assert popped == 3
    assert board.grid[-3:, col].sum() == 0

    # Test spawn avoids immediate triple on top of two identical beneath
    board.grid[:, col] = 0
    board.grid[-2:, col] = [1, 1]
    # Simulate a spawn at the top row; value must not be 1
    v = board.spawn_value_for_column(col)
    assert v in (1, 2, 3)
    assert v != 1, "Spawn must avoid forming instant triple"
