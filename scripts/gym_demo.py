import gymnasium as gym
import column_popper.envs  # ensure env registration


def main() -> None:
    env = gym.make("SpecKitAI/ColumnPopper-v1", seed=42)
    obs, info = env.reset()
    print("Obs keys:", list(obs.keys()))
    print("Action space:", env.action_space)

    total_reward = 0.0
    steps = 0
    try:
        for steps in range(1, 101):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"Steps: {steps}  Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()

