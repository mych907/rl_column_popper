from __future__ import annotations

import os
import shutil
from typing import Any, Dict


class AnsiRenderer:
    def __init__(self) -> None:
        self._supports_color = os.environ.get("TERM") not in (None, "dumb")

    def clear(self) -> None:
        print("\x1b[2J\x1b[H", end="")  # clear screen + home

    def draw(self, obs: Dict[str, Any], info: Dict[str, Any]) -> None:
        board = obs["board"]
        selection = obs["selection"]

        cols, _ = shutil.get_terminal_size((80, 24))
        sep = "\n"

        header = f"Column Popper  |  Score: {info.get('score', 0)}  |  Time Left: {int(info.get('time_left', 0))}s"
        print(header[: cols])
        print("=" * min(len(header), cols))

        # Draw grid top->bottom
        for r in range(board.shape[0]):
            row_vals = [str(int(x)) if int(x) != 0 else "." for x in board[r, :]]
            print("  ".join(row_vals))

        sel = int(selection[0])
        sval = int(selection[1])
        if sel:
            print(f"\nHolding: {sval}")
        else:
            print("\nHolding: none")

        print("\nControls: a=col0  s=col1  d=col2  f=fall  q=quit")

