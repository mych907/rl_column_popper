from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import column_popper.envs  # noqa: F401 ensure registration
from column_popper.render.ansi import AnsiRenderer


def _run_curses(env) -> int:
    try:
        import curses  # type: ignore
        from column_popper.render import curses_ui
    except Exception:
        return 1

    def _wrapped(stdscr):
        return curses_ui.run(stdscr, env)

    curses.wrapper(_wrapped)  # type: ignore[attr-defined]
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play Column Popper")
    parser.add_argument("--mode", choices=["play", "rollout", "stream"], default="play")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ui", choices=["auto", "curses", "ansi"], default="auto")
    args = parser.parse_args(argv)

    env = gym.make("SpecKitAI/ColumnPopper-v1", disable_env_checker=True, seed=args.seed)
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
            # Placeholder for rollout/stream modes
            obs, info = env.reset(seed=args.seed)
            for _ in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
