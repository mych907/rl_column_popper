from __future__ import annotations

import argparse
import json
import sys
from typing import Any, TextIO

import gymnasium as gym

import column_popper.envs  # noqa: F401


def _read_action(stdin: TextIO) -> int | None:
    line = stdin.readline()
    if line == "":
        return None  # EOF
    s = line.strip()
    if not s:
        return None
    try:
        a = int(s)
        if 0 <= a <= 3:
            return a
    except Exception:
        pass
    return None


def _to_jsonable(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obs.items():
        try:
            out[k] = v.tolist()
        except Exception:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless rollout that streams JSONL frames")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--format", choices=["jsonl"], default="jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-time", action="store_true", help="Include time_left_norm in obs")
    args = parser.parse_args(argv)

    env: gym.Env[dict[str, Any], int] = gym.make(
        "SpecKitAI/ColumnPopper-v1",
        disable_env_checker=True,
        seed=args.seed,
        include_time_left_norm=args.include_time,
    )
    try:
        for epi in range(args.episodes):
            obs, info = env.reset(seed=args.seed + epi)
            step_idx = 0
            while True:
                act = _read_action(sys.stdin)
                if act is None:
                    act = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(act)

                frame = {
                    "episode": epi,
                    "step": step_idx,
                    "action": act,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                    "obs": _to_jsonable(obs),
                }
                sys.stdout.write(json.dumps(frame) + "\n")
                sys.stdout.flush()
                step_idx += 1
                if terminated or truncated:
                    break
        return 0
    finally:
        from typing import cast

        cast(Any, env).close()


if __name__ == "__main__":
    raise SystemExit(main())
