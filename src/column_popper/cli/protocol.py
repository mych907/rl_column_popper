from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

import gymnasium as gym

import column_popper.envs  # noqa: F401


def _jsonable(x: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in x.items():
        try:
            out[k] = v.tolist()
        except Exception:
            out[k] = v
    return out


def _read_action(stdin: TextIO) -> int:
    line = stdin.readline()
    s = (line or "").strip()
    try:
        a = int(s)
    except Exception:
        a = 0
    if a < 0 or a > 3:
        a = 0
    return a


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactive streaming protocol over JSONL")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-time", action="store_true")
    args = parser.parse_args(argv)

    env: gym.Env[dict[str, Any], int] = gym.make(
        "SpecKitAI/ColumnPopper-v1",
        disable_env_checker=True,
        seed=args.seed,
        include_time_left_norm=args.include_time,
    )
    try:
        meta = {
            "type": "meta",
            "env_id": "SpecKitAI/ColumnPopper-v1",
            "action_space_n": 4,
        }
        sys.stdout.write(json.dumps(meta) + "\n")
        sys.stdout.flush()

        for epi in range(args.episodes):
            obs, info = env.reset(seed=args.seed + epi)
            sys.stdout.write(
                json.dumps({
                    "type": "reset",
                    "episode": epi,
                    "obs": _jsonable(obs),
                    "info": info,
                }) + "\n"
            )
            sys.stdout.flush()

            step = 0
            while True:
                # Request action
                sys.stdout.write(
                    json.dumps({
                        "type": "step_request",
                        "episode": epi,
                        "step": step,
                        "obs": _jsonable(obs),
                        "info": info,
                    }) + "\n"
                )
                sys.stdout.flush()

                action = _read_action(sys.stdin)
                obs, reward, terminated, truncated, info = env.step(action)
                sys.stdout.write(
                    json.dumps({
                        "type": "step_result",
                        "episode": epi,
                        "step": step,
                        "action": action,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info,
                        "obs": _jsonable(obs),
                    }) + "\n"
                )
                sys.stdout.flush()
                step += 1
                if terminated or truncated:
                    sys.stdout.write(
                        json.dumps({"type": "done", "episode": epi}) + "\n"
                    )
                    sys.stdout.flush()
                    break
        return 0
    finally:
        from typing import cast

        cast(Any, env).close()


if __name__ == "__main__":
    raise SystemExit(main())
