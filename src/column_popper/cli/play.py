from __future__ import annotations

import argparse
import sys
from typing import Any

import gymnasium as gym

import column_popper.envs  # noqa: F401 ensure registration
from column_popper.render.ansi import AnsiRenderer


def _run_curses(env: gym.Env[dict[str, Any], int]) -> int:
    try:
        import curses
        from column_popper.render import curses_ui
    except Exception:
        return 1

    def _wrapped(stdscr: Any) -> None:
        return curses_ui.run(stdscr, env)

    curses.wrapper(_wrapped)
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description="Play Column Popper")
    parser.add_argument("--mode", choices=["play", "rollout", "stream"], default="play")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ui", choices=["auto", "curses", "ansi"], default="auto")
    args = parser.parse_args(argv)

    # Human mode uses wall time for falling schedule; start at 3s per fall and ramp faster
    env: gym.Env[dict[str, Any], int] = gym.make(
        "SpecKitAI/ColumnPopper-v1",
        disable_env_checker=True,
        seed=args.seed,
        use_wall_time=True,
        initial_fall_interval=3.0,
        schedule_curve=[(20.0, 2.0), (40.0, 1.0)],
    )
    try:
        if args.mode == "play":
            # Choose UI
            use_curses = False
            if args.ui == "curses":
                use_curses = True
            elif args.ui == "auto":
                try:
                    import curses  # noqa: F401

                    use_curses = sys.stdin.isatty() and sys.stdout.isatty()
                except Exception:
                    use_curses = False

            if use_curses:
                return _run_curses(env)
            else:
                # ANSI fallback with simple input loop
                renderer = AnsiRenderer()
                obs, info = env.reset(seed=args.seed)
                while True:
                    renderer.clear()
                    renderer.draw(obs, info)
                    try:
                        s = input("> action [a/s/d=f cols, f=fall, q=quit]: ")
                    except EOFError:
                        break
                    if not s:
                        continue
                    ch = s.strip().lower()[0]
                    if ch == "q":
                        break
                    keymap = {"a": 0, "s": 1, "d": 2, "f": 3}
                    if ch not in keymap:
                        continue
                    action = keymap[ch]
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        renderer.clear()
                        renderer.draw(obs, info)
                        print("\nGame over.")
                        break
                return 0
        else:
            # Dispatch to subcommands for headless modes
            if args.mode == "rollout":
                from .rollout import main as rollout_main

                return rollout_main([f"--seed={args.seed}", "--episodes=1"])  # defer arg parsing
            if args.mode == "stream":
                from .protocol import main as protocol_main

                return protocol_main([f"--seed={args.seed}", "--episodes=1"])  # defer parsing
            return 0
    finally:
        from typing import Any as _Any, cast

        cast(_Any, env).close()  # cast to Any to avoid strict typing on gym close


if __name__ == "__main__":
    raise SystemExit(main())
