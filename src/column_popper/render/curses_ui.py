from __future__ import annotations

from typing import Any, Dict


def _draw_board(stdscr, obs: Dict[str, Any], info: Dict[str, Any]) -> None:
    board = obs["board"]
    selection = obs["selection"]

    stdscr.clear()
    header = f"Column Popper  |  Score: {int(info.get('score', 0))}  |  Time Left: {int(info.get('time_left', 0))}s"
    stdscr.addstr(0, 0, header)
    stdscr.addstr(1, 0, "=" * len(header))

    # Top -> bottom
    for r in range(board.shape[0]):
        row_vals = [str(int(x)) if int(x) != 0 else "." for x in board[r, :]]
        stdscr.addstr(2 + r, 0, "  ".join(row_vals))

    holding = "none" if int(selection[0]) == 0 else str(int(selection[1]))
    stdscr.addstr(2 + board.shape[0] + 1, 0, f"Holding: {holding}")
    stdscr.addstr(2 + board.shape[0] + 3, 0, "Controls: a=col0  s=col1  d=col2  f=fall  q=quit")
    stdscr.refresh()


def run(stdscr, env) -> None:
    stdscr.nodelay(False)
    stdscr.keypad(True)

    obs, info = env.reset()
    _draw_board(stdscr, obs, info)

    keymap = {
        ord("a"): 0,
        ord("s"): 1,
        ord("d"): 2,
        ord("f"): 3,
    }

    while True:
        ch = stdscr.getch()
        if ch in (ord("q"), 27):  # q or ESC
            break
        if ch not in keymap:
            continue
        action = keymap[ch]
        obs, reward, terminated, truncated, info = env.step(action)
        _draw_board(stdscr, obs, info)
        if terminated or truncated:
            break

