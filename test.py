# train_and_eval.py
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from env import *
from helper_utils import test_trained_model, make_env

# Create vectorized environment
grid_size = 6
num_obstacles = 5
max_steps = 50
env = make_vec_env(lambda: make_env(grid_size, num_obstacles, max_steps), n_envs=8)

# Define DQN model with appropriate hyperparameters

# correct
model1 = PPO.load("ppo_gridworld", env = env)
model2 = DQN.load("dqn_gridworld", env = env)
print("Starting PPO evaluation...")

# Save the final model
test_trained_model(model1, grid_size, num_obstacles, max_steps, num_episodes=1)
test_trained_model(model2, grid_size, num_obstacles, max_steps, num_episodes=1)
