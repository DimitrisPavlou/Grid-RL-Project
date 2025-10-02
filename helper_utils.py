# train_and_eval.py
import numpy as np
import gymnasium as gym
from env import GridEnv

class DictToArrayWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Flatten observation: agent(2) + target(2) + obstacles(num_obstacles*2)
        obs_size = 2 + 2 + env.num_obstacles * 2
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(env.grid_size) - 1,  # Use max dimension for upper bound
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
    env = GridEnv(grid_size=grid_size, max_steps=max_steps, num_obstacles=num_obstacles)
    env = DictToArrayWrapper(env)  # Convert dict to array
    return env


def test_trained_model(model, grid_size, num_obstacles, max_steps, num_episodes=10, render=True):
    """Test the trained model"""
    print(f"\nTesting trained model for {num_episodes} episodes...")
    
    successes = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        # Create a fresh environment for this episode
        original_env = GridEnv(grid_size=grid_size, max_steps=max_steps, num_obstacles=num_obstacles, render_mode="human")
        wrapped_env = DictToArrayWrapper(original_env)
        
        # Reset the environment
        obs, _ = wrapped_env.reset()
        terminated = False
        truncated = False
        steps = 0

        if render:
            original_env.enable_rendering()

        print(f"\n--- Episode {episode + 1} ---")
        if render:
            original_env.render_frame()# Render from ORIGINAL environment 
        
        while not (terminated or truncated):
            # Get action from model using the wrapped observation
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the wrapped environment
            obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
            steps += 1
            total_steps += 1
            
            # Render from ORIGINAL environment (not the wrapper)
            if render: 
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

