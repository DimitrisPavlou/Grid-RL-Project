# train_and_eval.py
from stable_baselines3 import PPO, DQN
from helper_utils import test_trained_model

# Create vectorized environment
grid_size = (6,6)
max_steps = 50


num_obstacles = 5
# load models
model1 = PPO.load("agent_parameters/ppo_gridenv")
model2 = DQN.load("agent_parameters/dqn_gridenv")
print("Starting evaluation...")

# Save the final model
print("PPO Results")
test_trained_model(model1, grid_size, num_obstacles, max_steps, expanded=False, num_episodes=100, render = False, print_every=10)
print("DQN results")
test_trained_model(model2, grid_size, num_obstacles, max_steps, expanded=False, num_episodes=100, render = False, print_every=10)

# load the expanded env models
model1_exp = PPO.load("agent_parameters/ppo_expanded_gridenv")
model2_exp = DQN.load("agent_parameters/dqn_expanded_gridenv")

num_bonus = 2
num_obstacles_expanded = 3
print("PPO Results")
test_trained_model(model1_exp, grid_size, num_obstacles_expanded, max_steps, num_bonus=num_bonus, num_episodes=100, render=False, expanded=True)
print("DQN Results")
test_trained_model(model2_exp, grid_size, num_obstacles_expanded, max_steps, num_bonus=num_bonus, num_episodes=100, render=False, expanded=True)
