from __future__ import annotations

from typing import Any


def _draw_board(stdscr: Any, obs: dict[str, Any], info: dict[str, Any]) -> None:
    board = obs["board"]
    selection = obs["selection"]

    stdscr.clear()
    header1 = f"Column Popper  |  Score: {int(info.get('score', 0))}"
    header2 = f"Time Left: {int(info.get('time_left', 0))}s"
    stdscr.addstr(0, 0, header1)
    stdscr.addstr(1, 0, header2)
    stdscr.addstr(2, 0, "=" * max(len(header1), len(header2)))

    # Colors
    try:
        import curses

        has_colors = curses.has_colors()
        if has_colors:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_BLUE, -1)
            blue = curses.color_pair(1)
        else:
            blue = 0
    except Exception:
        has_colors = False
        blue = 0

    sel_pos = obs.get("sel_pos", [-1, -1])
    sel_r, sel_c = int(sel_pos[0]), int(sel_pos[1])

    # Top -> bottom
    for r in range(board.shape[0]):
        parts = []
        for c in range(board.shape[1]):
            v = int(board[r, c])
            s = "." if v == 0 else str(v)
            parts.append((s, blue if (r == sel_r and c == sel_c and int(selection[0]) == 1) else 0))
        # write with coloring
        x = 0
        stdscr.move(3 + r, 0)
        for s, attr in parts:
            if attr:
                stdscr.addstr(s, attr)
            else:
                stdscr.addstr(s)
            stdscr.addstr("  ")

    holding = "none" if int(selection[0]) == 0 else str(int(selection[1]))
    stdscr.addstr(3 + board.shape[0] + 1, 0, f"Holding: {holding}")
    stdscr.addstr(3 + board.shape[0] + 3, 0, "Controls: a=col0  s=col1  d=col2  f=fall  q=quit")
    stdscr.refresh()


def run(stdscr: Any, env: Any) -> None:
    stdscr.nodelay(True)
    stdscr.keypad(True)

    obs, info = env.reset()
    _draw_board(stdscr, obs, info)

    import time
    last_draw = time.perf_counter()
    draw_interval = 1.0

    keymap = {
        ord("a"): 0,
        ord("s"): 1,
        ord("d"): 2,
        ord("f"): 3,
    }

    base = getattr(env, "unwrapped", env)

    while True:
        ch = stdscr.getch()
        if ch == -1:
            # No key: advance wall time and redraw periodically
            if hasattr(base, "wall_time_tick"):
                base.wall_time_tick()
            now = time.perf_counter()
            if now - last_draw >= draw_interval:
                obs, info = base.peek() if hasattr(base, "peek") else (obs, info)
                _draw_board(stdscr, obs, info)
                last_draw = now
            # Check for end conditions
            if getattr(base, "_terminated", False) or getattr(base, "schedule").truncated:
                break
            time.sleep(0.02)
            continue
        if ch in (ord("q"), 27):  # q or ESC
            break
        if ch not in keymap:
            continue
        action = keymap[ch]
        obs, reward, terminated, truncated, info = env.step(action)
        _draw_board(stdscr, obs, info)
        if terminated or truncated:
            break
