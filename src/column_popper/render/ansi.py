from __future__ import annotations

import os
import shutil
from typing import Any


class AnsiRenderer:
    def __init__(self) -> None:
        self._supports_color = os.environ.get("TERM") not in (None, "dumb")

    def clear(self) -> None:
        print("\x1b[2J\x1b[H", end="")  # clear screen + home

    def draw(self, obs: dict[str, Any], info: dict[str, Any]) -> None:
        board = obs["board"]
        selection = obs["selection"]
        sel_pos = obs.get("sel_pos", [-1, -1])
        sel_r, sel_c = int(sel_pos[0]), int(sel_pos[1])

        cols, _ = shutil.get_terminal_size((80, 24))

        header = (
            f"Column Popper  |  Score: {info.get('score', 0)}  |  Time Left: "
            f"{int(info.get('time_left', 0))}s"
        )
        print(header[: cols])
        print("=" * min(len(header), cols))

        # Draw grid top->bottom
        for r in range(board.shape[0]):
            cells = []
            for c in range(board.shape[1]):
                v = int(board[r, c])
                if v == 0:
                    s = "."
                else:
                    s = str(v)
                if r == sel_r and c == sel_c and int(selection[0]) == 1:
                    s = f"\x1b[34m{s}\x1b[0m"
                cells.append(s)
            print("  ".join(cells))

        sel = int(selection[0])
        sval = int(selection[1])
        if sel:
            print(f"\nHolding: {sval}")
        else:
            print("\nHolding: none")

        print("\nControls: a=col0  s=col1  d=col2  f=fall  q=quit")
