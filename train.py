# train_and_eval.py
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from helper_utils import make_env, make_expanded_env, test_trained_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available Device : {device}")


grid_size = (6,6)
num_obstacles = 5
max_steps = 50
num_bonus = 2 # valid for the expanded env 
num_obstacles_exp = 3 # valid for the expanded env
# pick environment for the agents to be trained on
expanded_env = make_vec_env(lambda: make_expanded_env(grid_size, num_obstacles_exp, max_steps, num_bonus), n_envs=8)
env = make_vec_env(lambda: make_env(grid_size, num_obstacles, max_steps, num_bonus), n_envs=8)

policy_kwargs_dqn = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.LeakyReLU,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=dict(weight_decay=1e-2)
)

# Define DQN model with appropriate hyperparameters
model_dqn = DQN(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100_000,            # Size of replay buffer
    learning_starts=1000,           # Learn after collecting this many samples
    batch_size=64,                  # Mini-batch size
    tau=1e-3,                       # Target network update rate
    gamma=0.99,                     # Discount factor
    train_freq=4,                   # Update model every 4 steps
    gradient_steps=1,               # How many gradient steps after each update
    target_update_interval=8,       # Update target network every 8 steps
    exploration_fraction=0.4,       # Fraction of training for exploration decay
    exploration_initial_eps=1.0,    # Initial exploration rate
    exploration_final_eps=0.05,     # Final exploration rate
    verbose=1, 
    policy_kwargs=policy_kwargs_dqn,
    device=device
)

policy_kwargs_ppo = dict(
    net_arch=dict(
        pi=[256, 256],  # Policy network architecture
        vf=[256, 256]   # Value network architecture
    ),
    activation_fn=torch.nn.LeakyReLU,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=dict(weight_decay=1e-2)
)

model_ppo = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,     # encourages exploration,
    verbose=1,
    device = device, 
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs_ppo
)


print("Starting DQN training...")
model_ppo.learn(total_timesteps= 10_000_000)
# Save the final model
model_ppo.save("ppo_gridenv")


model_dqn.learn(total_timesteps= 10_000_000)
# Save the final model
model_dqn.save("dqn_gridenv")


print("Training completed!")
