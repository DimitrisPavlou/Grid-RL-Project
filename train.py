# train_and_eval.py
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
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



# Define DQN model with appropriate hyperparameters
#model = DQN(
#    "MlpPolicy",
#    env,
#    learning_rate=1e-3,
#    buffer_size=50000,        # Size of replay buffer
#    learning_starts=1000,     # Learn after collecting this many samples
#    batch_size=32,           # Mini-batch size
#    tau=1.0,                 # Target network update rate
#    gamma=0.99,              # Discount factor
#    train_freq=4,            # Update model every 4 steps
#    gradient_steps=1,        # How many gradient steps after each update
#    target_update_interval=100,  # Update target network every 100 steps
#    exploration_fraction=0.4,    # Fraction of training for exploration decay
#    exploration_initial_eps=1.0, # Initial exploration rate
#    exploration_final_eps=0.2,  # Final exploration rate
#    verbose=1,
#    policy_kwargs=policy_kwargs, 
#    device=device
#)

grid_size = 6
num_obstacles = 5 
max_steps = 50
env = make_vec_env(lambda: make_env(grid_size, num_obstacles, max_steps), n_envs=8)


policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],  # Policy network architecture
        vf=[256, 256]   # Value network architecture
    ),
    activation_fn=torch.nn.LeakyReLU,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=dict(weight_decay=1e-2)
)


model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=512,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,   # encourages exploration,
    verbose=1,
    device = device, 
    max_grad_norm=1.0,
    policy_kwargs=policy_kwargs
)


print("Starting PPO training...")
model.learn(total_timesteps= 5_000_000)

# Save the final model
model.save("ppo_gridworld")
print("Training completed!")

# Save the final model
test_trained_model(model, grid_size, num_obstacles, max_steps, num_episodes=100)