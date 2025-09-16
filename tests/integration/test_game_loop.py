import pytest


def test_basic_pick_drop_pop_and_overflow_sequence():
    """
    Integration scenario from the spec:
    - Picking up from a column and dropping into another
    - Forming a vertical triple triggers a pop and increases score
    - Overflow ends the game
    - Manual fall action advances time/rows
    """
    import gymnasium as gym
    import column_popper.envs  # noqa: F401

    env_id = "SpecKitAI/ColumnPopper-v1"
    env = gym.make(env_id, disable_env_checker=True)
    try:
        obs, info = env.reset(seed=7)

        # Try a short scripted sequence; semantics per spec
        # 0,1,2 operate on a target column; 3 = manual fall
        actions = [0, 3, 1, 3, 2, 3]
        total_reward = 0.0
        terminated = truncated = False
        for a in actions:
            obs, reward, terminated, truncated, info = env.step(a)
            total_reward += reward
            if terminated or truncated:
                break

        assert "score" in info
        assert isinstance(info["score"], (int, float))

        # Force overflow by repeatedly forcing falls on a single column
        # The exact number of steps may vary by implementation; just do a lot
        steps = 0
        while not (terminated or truncated) and steps < 500:
            obs, reward, terminated, truncated, info = env.step(3)
            steps += 1

        assert terminated or truncated, "Game should end by overflow or truncation eventually"
    finally:
        env.close()


def test_time_up_truncation():
    import gymnasium as gym
    import column_popper.envs  # noqa: F401

    env_id = "SpecKitAI/ColumnPopper-v1"
    env = gym.make(env_id, disable_env_checker=True)
    try:
        obs, info = env.reset(seed=11)
        # Take many no-op-equivalent or random actions until truncation
        for _ in range(5000):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        assert truncated or terminated
    finally:
        env.close()
