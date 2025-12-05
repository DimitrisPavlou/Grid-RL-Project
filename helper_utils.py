"""Helper utilities for training and testing RL agents."""
import numpy as np
import gymnasium as gym
from env import GridEnv
from expanded_env import ExpandedGridEnv


class DictToArrayWrapper(gym.ObservationWrapper):
    """Wrapper to flatten dictionary observations into 1D arrays."""
    
    def __init__(self, env):
        """
        Initialize wrapper.
        
        Args:
            env: Environment with dictionary observations
        """
        super().__init__(env)
        
        # Calculate flattened observation size
        obs_size = 2 + 2 + env.num_obstacles * 2
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(env.grid_size) - 1,
            shape=(obs_size,),
            dtype=np.int64
        )

    def observation(self, observation):
        """
        Flatten dictionary observation to 1D array.
        
        Args:
            observation: Dictionary with 'agent', 'target', 'obstacles' keys
            
        Returns:
            Flattened numpy array
        """
        return np.concatenate([
            observation["agent"],
            observation["target"],
            observation["obstacles"].flatten()
        ]).astype(np.int64)


class ExpandedDictToArrayWrapper(gym.ObservationWrapper):
    """Wrapper to flatten expanded environment observations into 1D arrays."""
    
    def __init__(self, env):
        """
        Initialize wrapper.
        
        Args:
            env: Expanded environment with dictionary observations
        """
        super().__init__(env)
        
        # Calculate flattened observation size
        obs_size = (
            2 +                      # agent position
            2 +                      # target position
            env.num_obstacles * 2 +  # obstacle positions
            env.num_bonus * 2 +      # bonus positions
            env.num_bonus            # bonus collected flags
        )
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=max(env.grid_size) - 1,
            shape=(obs_size,),
            dtype=np.int64
        )

    def observation(self, observation):
        """
        Flatten dictionary observation to 1D array.
        
        Args:
            observation: Dictionary with agent, target, obstacles, bonuses
            
        Returns:
            Flattened numpy array
        """
        return np.concatenate([
            observation["agent"],
            observation["target"],
            observation["obstacles"].flatten(),
            observation["bonus_positions"].flatten(),
            observation["bonus_collected"].astype(np.int64)
        ]).astype(np.int64)


def make_env(grid_size, num_obstacles, max_steps):
    """
    Create basic grid environment with observation wrapper.
    
    Args:
        grid_size: Tuple of (rows, cols)
        num_obstacles: Number of obstacles
        max_steps: Maximum steps per episode
        
    Returns:
        Wrapped environment
    """
    env = GridEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        num_obstacles=num_obstacles
    )
    env = DictToArrayWrapper(env)
    return env


def make_expanded_env(grid_size, num_obstacles, max_steps, num_bonus=2):
    """
    Create expanded grid environment with observation wrapper.
    
    Args:
        grid_size: Tuple of (rows, cols)
        num_obstacles: Number of obstacles
        max_steps: Maximum steps per episode
        num_bonus: Number of bonus items
        
    Returns:
        Wrapped environment
    """
    env = ExpandedGridEnv(
        grid_size=grid_size,
        max_steps=max_steps,
        num_obstacles=num_obstacles,
        num_bonus=num_bonus
    )
    env = ExpandedDictToArrayWrapper(env)
    return env


def test_trained_model(
    model,
    grid_size,
    num_obstacles,
    max_steps,
    expanded=False,
    num_bonus=0,
    num_episodes=10,
    render=False,
    print_every=10,
    render_delay=0.3
):
    """
    Test a trained RL model on the grid environment.
    
    Args:
        model: Trained RL model with predict() method
        grid_size: Tuple of (rows, cols)
        num_obstacles: Number of obstacles
        max_steps: Maximum steps per episode
        expanded: Whether to use expanded environment
        num_bonus: Number of bonus items (for expanded env)
        num_episodes: Number of test episodes
        render: Whether to render episodes visually
        print_every: Print status every N episodes
        render_delay: Seconds to sleep after each rendered frame
    """
    print(f"\nTesting trained model for {num_episodes} episodes...")
    
    successes = 0
    total_steps = 0
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        # Create fresh environment for this episode
        if expanded:
            original_env = ExpandedGridEnv(
                grid_size=grid_size,
                max_steps=max_steps,
                num_obstacles=num_obstacles,
                num_bonus=num_bonus,
                render_mode="human" if render else None,
                render_delay=render_delay
            )
            wrapped_env = ExpandedDictToArrayWrapper(original_env)
        else:
            original_env = GridEnv(
                grid_size=grid_size,
                max_steps=max_steps,
                num_obstacles=num_obstacles,
                render_mode="human" if render else None,
                render_delay=render_delay
            )
            wrapped_env = DictToArrayWrapper(original_env)

        # Reset environment
        obs, _ = wrapped_env.reset()
        terminated = False
        truncated = False
        steps = 0
        episode_reward = 0.0

        if render:
            original_env.enable_rendering()

        if episode % print_every == 0:
            print(f"\n--- Episode {episode} ---")

        if render:
            original_env.render_frame()
        
        # Run episode
        while not (terminated or truncated):
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
            steps += 1
            total_steps += 1
            episode_reward += reward
            
            # Render if enabled
            if render:
                original_env.render_frame()
            
            # Check termination reason
            if terminated:
                if reward > 0:  # Success
                    successes += 1
                    if episode % print_every == 0:
                        print(f"✓ Success! Reached target in {steps} steps")
                else:  # Collision or premature target
                    if episode % print_every == 0:
                        print(f"✗ Collision/Failure in {steps} steps")
            elif truncated:
                if episode % print_every == 0:
                    print(f"⏱ Timeout! Max steps reached")

        if episode % print_every == 0:
            print(f"Episode reward: {episode_reward:.2f}")
        
        episode_rewards.append(episode_reward)
        wrapped_env.close()

    # Calculate and display statistics
    success_rate = (successes / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n{'='*60}")
    print(f"Final Results over {num_episodes} episodes:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Average steps per episode: {avg_steps:.1f}")
    print(f"  Average reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"{'='*60}")
