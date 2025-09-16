try:
    import pytest_benchmark  # noqa: F401
except Exception:  # pragma: no cover
    import pytest

    pytest.skip("pytest-benchmark not installed", allow_module_level=True)


def test_env_step_benchmark(benchmark):
    import gymnasium as gym
    import column_popper.envs  # noqa: F401

    env = gym.make("SpecKitAI/ColumnPopper-v1", disable_env_checker=True, seed=123)
    try:
        env.reset()

        def _do_step():
            # Fixed action to reduce variance
            env.step(3)

        benchmark(_do_step)
    finally:
        env.close()

