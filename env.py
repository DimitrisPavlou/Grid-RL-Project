"""Basic grid environment for reinforcement learning."""
import numpy as np
import gymnasium as gym
import time


class GridEnv(gym.Env):
    """Grid environment where agent must reach target while avoiding obstacles."""
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self, 
        render_mode=None, 
        grid_size=(5, 5), 
        max_steps=100, 
        num_obstacles=3,
        render_delay=0.3
    ):
        """
        Initialize the grid environment.
        
        Args:
            render_mode: Rendering mode ('human', 'ansi', or None)
            grid_size: Tuple of (rows, cols) defining grid dimensions
            max_steps: Maximum steps per episode
            num_obstacles: Number of obstacle cells
            render_delay: Seconds to sleep after rendering (for visual clarity)
        """
        self.grid_size = np.array(grid_size, dtype=int)
        self.rows, self.cols = self.grid_size
        self.max_steps = max_steps
        self.step_count = 0
        self.num_obstacles = num_obstacles
        self._should_render = False
        self.render_delay = render_delay

        # Observation space
        max_coord = max(self.rows, self.cols) - 1
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, max_coord, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, max_coord, shape=(2,), dtype=int),
            "obstacles": gym.spaces.Box(
                0, max_coord, 
                shape=(self.num_obstacles, 2), 
                dtype=int
            )
        })
        
        # Action space: 4 cardinal directions
        self.action_space = gym.spaces.Discrete(4)
        
        # Action mappings
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up  
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
        
        self.render_mode = render_mode
        self._agent_location = None
        self._target_location = None
        self._obstacles = None

    def _get_obs(self):
        """Get current observation."""
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "obstacles": self._obstacles.copy(),
        }

    def _get_info(self):
        """Get auxiliary information."""
        return {
            "distance": float(np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )),
            "steps_remaining": self.max_steps - self.step_count
        }

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self.step_count = 0

        # Place agent randomly
        self._agent_location = self.np_random.integers(
            0, self.grid_size, size=2, dtype=int
        )
        
        # Place target (different from agent)
        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.grid_size, size=2, dtype=int
            )

        # Place obstacles (no overlap with agent or target)
        self._obstacles = []
        max_attempts = self.rows * self.cols * 10
        attempts = 0
        
        while len(self._obstacles) < self.num_obstacles and attempts < max_attempts:
            pos = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            if (not np.array_equal(pos, self._agent_location) and 
                not np.array_equal(pos, self._target_location) and
                not any(np.array_equal(pos, o) for o in self._obstacles)):
                self._obstacles.append(pos)
            attempts += 1
            
        if len(self._obstacles) < self.num_obstacles:
            raise RuntimeError(
                f"Could not place {self.num_obstacles} obstacles in "
                f"{self.rows}x{self.cols} grid. Grid may be too small."
            )
            
        self._obstacles = np.array(self._obstacles, dtype=int)

        if self._should_render:
            self.render_frame()

        return self._get_obs(), self._get_info()

    def _is_out_of_bounds(self, position):
        """Check if position is outside grid bounds."""
        return (
            position[0] < 0 or position[0] >= self.cols or 
            position[1] < 0 or position[1] >= self.rows
        )

    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0-3)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        action = int(action)
        direction = self._action_to_direction[action]

        # Calculate new position
        new_pos = self._agent_location + direction
        
        # Check bounds
        if self._is_out_of_bounds(new_pos):
            reward = -1.0
            terminated = True
        # Check obstacle collision
        elif any(np.array_equal(new_pos, o) for o in self._obstacles):
            reward = -1.0
            terminated = True
            self._agent_location = new_pos
        else:
            # Valid move
            old_dist = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
            self._agent_location = new_pos
            new_dist = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
            
            # Check if target reached
            terminated = np.array_equal(self._agent_location, self._target_location)
            
            if terminated:
                reward = 1.0
            elif new_dist < old_dist:
                reward = 0.01
            else:
                reward = -0.05

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def enable_rendering(self):
        """Enable rendering for this environment instance."""
        self._should_render = True

    def disable_rendering(self):
        """Disable rendering for this environment instance."""
        self._should_render = False

    def set_render_delay(self, delay):
        """
        Set the delay between rendered frames.
        
        Args:
            delay: Seconds to sleep after rendering
        """
        self.render_delay = delay

    def render_frame(self):
        """Render current state to console with optional delay."""
        print("\n" + "=" * 50)
        print(
            f"Grid World ({self.rows}x{self.cols}) | "
            f"Step: {self.step_count}/{self.max_steps}"
        )
        print("A = Agent, T = Target, X = Obstacle")
        print()
        
        for y in range(self.rows - 1, -1, -1):
            row = "|"
            for x in range(self.cols):
                pos = np.array([x, y])
                if np.array_equal(pos, self._agent_location):
                    row += " A |"
                elif np.array_equal(pos, self._target_location):
                    row += " T |"
                elif any(np.array_equal(pos, o) for o in self._obstacles):
                    row += " X |"
                else:
                    row += " . |"
            print(row)

        obs = self._get_obs()
        info = self._get_info()
        print(f"\nAgent: {obs['agent']}, Target: {obs['target']}")
        print(f"Obstacles: {obs['obstacles'].tolist()}")
        print(
            f"Distance: {info['distance']:.1f}, "
            f"Steps remaining: {info['steps_remaining']}"
        )
        print("=" * 50)
        
        # Sleep for visual clarity
        if self.render_delay > 0:
            time.sleep(self.render_delay)

    def close(self):
        """Clean up resources."""
        pass
