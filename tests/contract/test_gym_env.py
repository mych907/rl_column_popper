import pytest


def test_env_contract_spaces_and_info():
    import gymnasium as gym
    # Ensure env is registered by importing package envs
    import column_popper.envs  # noqa: F401

    env_id = "SpecKitAI/ColumnPopper-v1"

    # Expect env to be registered and constructible
    env = gym.make(env_id, disable_env_checker=True, seed=123)
    try:
        # Observation space
        assert hasattr(env, "observation_space"), "Env must expose observation_space"
        assert hasattr(env, "action_space"), "Env must expose action_space"

        # Validate observation space structure
        from gymnasium.spaces import Dict, Box, Discrete
        import numpy as np

        assert isinstance(env.observation_space, Dict), "Observation must be a Dict"
        obs_space = env.observation_space
        assert "board" in obs_space.spaces, "Observation must include 'board'"
        assert "selection" in obs_space.spaces, "Observation must include 'selection'"

        board_space = obs_space["board"]
        sel_space = obs_space["selection"]
        assert isinstance(board_space, Box)
        assert isinstance(sel_space, Box)
        assert tuple(board_space.shape) == (8, 3)
        assert sel_space.shape == (2,)
        assert board_space.dtype == np.int32
        assert sel_space.dtype == np.int32

        # Validate action space
        assert isinstance(env.action_space, Discrete)
        assert env.action_space.n == 4

        # Reset/step contract and info keys
        obs, info = env.reset(seed=123)
        assert isinstance(info, dict)
        for k in [
            "score",
            "time_left",
            "pops_this_step",
            "fall_interval",
            "seed",
            "version",
        ]:
            assert k in info, f"Missing info key: {k}"

        # One random step should return the 5-tuple
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()


def test_env_determinism_with_seed():
    import gymnasium as gym
    import column_popper.envs  # noqa: F401
    import numpy as np

    env_id = "SpecKitAI/ColumnPopper-v1"
    actions = [0, 1, 2, 3] * 5

    env1 = gym.make(env_id, disable_env_checker=True)
    env2 = gym.make(env_id, disable_env_checker=True)
    try:
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        frames1 = []
        frames2 = []
        for a in actions:
            o1, r1, t1, tr1, _ = env1.step(a)
            o2, r2, t2, tr2, _ = env2.step(a)
            frames1.append((o1, r1, t1, tr1))
            frames2.append((o2, r2, t2, tr2))

        # Compare arrays and scalars element-wise
        for (o1, r1, t1, tr1), (o2, r2, t2, tr2) in zip(frames1, frames2):
            # Observations are dicts containing numpy arrays
            assert set(o1.keys()) == set(o2.keys())
            for k in o1:
                if hasattr(o1[k], "dtype"):
                    np.testing.assert_array_equal(o1[k], o2[k])
                else:
                    assert o1[k] == o2[k]
            assert r1 == r2
            assert t1 == t2
            assert tr1 == tr2
    finally:
        env1.close()
        env2.close()
