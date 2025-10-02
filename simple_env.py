import numpy as np
import gymnasium as gym

# a simpler environment for experimentation

class SimpleGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, max_steps=100):
        self.size = size
        self.max_steps = max_steps
        self.step_count = 0
        
        # Observation: agent and target positions
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        
        # Action: 4 directions
        self.action_space = gym.spaces.Discrete(4)
        
        # Movement directions: right, up, left, down
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, 1]),   # up  
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
        
        self.render_mode = render_mode
        self._agent_location = None
        self._target_location = None

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "steps_remaining": self.max_steps - self.step_count
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Random positions for agent and target
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        
        # Ensure target is different from agent
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        if self.render_mode == "human":
            self.render_frame()
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Increment step counter
        self.step_count += 1
        
        direction = self._action_to_direction[action]
        
        # Move agent (clipping to stay in bounds)
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        
        # Check if reached target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        
        # Check if max steps reached (truncated)
        truncated = self.step_count >= self.max_steps
        
        if self.render_mode == "human":
            self.render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render_frame(self):
        """Human-readable rendering to console"""
        print("\n" + "=" * 50) 
        print(f"Grid World (size: {self.size}x{self.size}) | Step: {self.step_count}/{self.max_steps}")
        print("A = Agent, T = Target")
        print()
        
        # Create visual grid
        for y in range(self.size - 1, -1, -1):  # Top to bottom
            row = "|"
            for x in range(self.size):
                if np.array_equal([x, y], self._agent_location):
                    row += " A |"
                elif np.array_equal([x, y], self._target_location):
                    row += " T |"
                else:
                    row += " . |"
            print(row)
        
        # Print info
        obs = self._get_obs()
        info = self._get_info()
        print(f"\nAgent: {obs['agent']}, Target: {obs['target']}")
        print(f"Distance: {info['distance']}, Steps remaining: {info['steps_remaining']}")
        print("=" * 50)

    def close(self):
        pass
