"""Training script for PPO and DQN agents on grid environments."""
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from helper_utils import make_env, make_expanded_env


def main():
    """Main training function."""
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment parameters
    grid_size = (6, 6)
    num_obstacles = 5
    max_steps = 50
    num_bonus = 2
    num_obstacles_exp = 3
    
    # Training parameters
    n_envs = 8
    total_timesteps = 10_000_000

    # Choose which environment to train on
    print("\nCreating environments...")
    print(f"Grid size: {grid_size}, Obstacles: {num_obstacles}, Max steps: {max_steps}")
    
    # Create vectorized environments
    env = make_vec_env(
        lambda: make_env(grid_size, num_obstacles, max_steps),
        n_envs=n_envs
    )
    
    # Uncomment to train on expanded environment instead:
    # env = make_vec_env(
    #     lambda: make_expanded_env(grid_size, num_obstacles_exp, max_steps, num_bonus),
    #     n_envs=n_envs
    # )

    # ===== DQN Configuration =====
    policy_kwargs_dqn = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.LeakyReLU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-2)
    )

    model_dqn = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=10000,
        batch_size=64,
        tau=1e-3,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=8,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        policy_kwargs=policy_kwargs_dqn,
        device=device
    )

    # ===== PPO Configuration =====
    policy_kwargs_ppo = dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network
            vf=[256, 256]   # Value network
        ),
        activation_fn=torch.nn.LeakyReLU,
        optimizer_class=torch.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-2)
    )

    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        verbose=1,
        policy_kwargs=policy_kwargs_ppo,
        device=device
    )

    # ===== Training =====
    print("\n" + "="*60)
    print("Starting PPO training...")
    print("="*60)
    model_ppo.learn(total_timesteps=total_timesteps)
    model_ppo.save("ppo_gridenv")
    print("\n✓ PPO training completed and model saved!")

    print("\n" + "="*60)
    print("Starting DQN training...")
    print("="*60)
    model_dqn.learn(total_timesteps=total_timesteps)
    model_dqn.save("dqn_gridenv")
    print("\n✓ DQN training completed and model saved!")

    print("\n" + "="*60)
    print("All training completed successfully!")
    print("="*60)

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
