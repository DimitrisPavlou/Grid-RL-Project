# train_and_eval.py
import os
import time
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from env import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class DictToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.size = env.size 
        # Flatten observation: agent(2) + target(2) + obstacles(num_obstacles*2)
        obs_size = 2 + 2 + env.num_obstacles * 2
        self.observation_space = gym.spaces.Box(
            low=0,
            high=env.size - 1,
            shape=(obs_size,),
            dtype=int
        )

    def observation(self, observation):
        # Flatten everything into 1D
        return np.concatenate([
            observation["agent"],
            observation["target"],
            observation["obstacles"].flatten()
        ])

        
def make_env(grid_size, num_obstacles, max_steps):
    env = GridEnv(size=grid_size, max_steps=max_steps, num_obstacles=num_obstacles)
    env = DictToArrayWrapper(env)  # Convert dict to array
    return env

#def test_trained_model(model, grid_size, num_obstacles, max_steps, num_episodes=10):
#    """Test the trained model - CORRECTED VERSION"""
#    print(f"\nTesting trained model for {num_episodes} episodes...")
#    
#    successes = 0
#    total_steps = 0
#    
#    for episode in range(num_episodes):
#        # Create a fresh environment for this episode
#        original_env = SimpleGridWorldEnv(size=grid_size, max_steps=max_steps, num_obstacles=num_obstacles, render_mode="ansi")
#        wrapped_env = DictToArrayWrapper(original_env)
#        
#        # Reset the wrapped environment (this also resets the original_env internally)
#        obs, _ = wrapped_env.reset()
#        terminated = False
#        truncated = False
#        steps = 0
#        
#        print(f"\n--- Episode {episode + 1} ---")
#        wrapped_env.render_frame()  # Show initial state
#        
#        while not (terminated or truncated):
#            # Get action from model using the wrapped observation
#            action, _ = model.predict(obs, deterministic=True)
#            
#            # Step the wrapped environment ONCE
#            obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
#            steps += 1
#            total_steps += 1
#            
#            # The step above also updated the original_env, so we can render it
#            wrapped_env.render_frame()
#            
#            if terminated:
#                successes += 1
#                print(f"🎉 Success! Reached target in {steps} steps")
#                break
#            elif truncated:
#                print(f"❌ Failed! Max steps reached")
#                break
#        
#        print(f"Episode completed in {steps} steps")
#        wrapped_env.close()
#    
#    success_rate = (successes / num_episodes) * 100
#    avg_steps = total_steps / num_episodes
#    
#    print(f"\n📊 Final Results:")
#    print(f"Success rate: {success_rate:.1f}%")
#    print(f"Average steps per episode: {avg_steps:.1f}")



def test_trained_model(model, grid_size, num_obstacles, max_steps, num_episodes=10):
    """Test the trained model - FIXED VERSION"""
    print(f"\nTesting trained model for {num_episodes} episodes...")
    
    successes = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        # Create a fresh environment for this episode
        original_env = GridEnv(size=grid_size, max_steps=max_steps, num_obstacles=num_obstacles, render_mode="ansi")
        wrapped_env = DictToArrayWrapper(original_env)
        
        # Reset the environment
        obs, _ = wrapped_env.reset()
        terminated = False
        truncated = False
        steps = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        original_env.render_frame()  # Render from ORIGINAL environment
        
        while not (terminated or truncated):
            # Get action from model using the wrapped observation
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the wrapped environment
            obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
            steps += 1
            total_steps += 1
            
            # Render from ORIGINAL environment (not the wrapper)
            original_env.render_frame()
            
            if terminated:
                # Check if termination was due to SUCCESS or COLLISION
                if reward > 0:  # Success reward
                    successes += 1
                    print(f"🎉 Success! Reached target in {steps} steps")
                else:  # Collision penalty
                    print(f"💥 Collision! Hit obstacle in {steps} steps")
                break
            elif truncated:
                print(f"❌ Failed! Max steps reached")
                break
        
        print(f"Episode completed in {steps} steps")
        wrapped_env.close()
    
    success_rate = (successes / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    
    print(f"\n📊 Final Results:")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps per episode: {avg_steps:.1f}")






# Create vectorized environment
grid_size = 6
num_obstacles = 1 
max_steps = 50
env = make_vec_env(lambda: make_env(grid_size, num_obstacles, max_steps), n_envs=8)

# Define DQN model with appropriate hyperparameters

# correct
model = PPO.load("ppo_gridworld", env = env)

print("Starting PPO evaluation...")

# Save the final model
test_trained_model(model, grid_size, num_obstacles, max_steps, num_episodes=100)