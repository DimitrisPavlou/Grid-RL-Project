# grid_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5, max_steps=100, num_obstacles=3):
        self.size = size
        self.max_steps = max_steps
        self.step_count = 0
        self.num_obstacles = num_obstacles
        
        # Observation: agent, target, obstacles
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "obstacles": gym.spaces.Box(0, size - 1, shape=(self.num_obstacles, 2), dtype=int)
        })
        
        # Action: 4 directions
        self.action_space = gym.spaces.Discrete(4)
        
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
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "obstacles": self._obstacles,
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "steps_remaining": self.max_steps - self.step_count
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Random agent and target
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Random obstacles (no overlap with agent or target)
        self._obstacles = []
        while len(self._obstacles) < self.num_obstacles:
            pos = self.np_random.integers(0, self.size, size=2, dtype=int)
            if (not np.array_equal(pos, self._agent_location) and 
                not np.array_equal(pos, self._target_location) and
                not any(np.array_equal(pos, o) for o in self._obstacles)):
                self._obstacles.append(pos)
        self._obstacles = np.array(self._obstacles, dtype=int)

        if self.render_mode == "human":
            self.render_frame()
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        action = int(action)
        direction = self._action_to_direction[action]

        new_pos = np.clip(self._agent_location + direction, 0, self.size - 1)

        # Check collision with obstacles
        if any(np.array_equal(new_pos, o) for o in self._obstacles):
            reward = -1
            terminated = True  # Episode ends immediately
        else:
            self._agent_location = new_pos
            terminated = np.array_equal(self._agent_location, self._target_location)
            if terminated: 
                reward = 1.0 
            else: 
                old_dist = np.linalg.norm(self._agent_location - self._target_location, ord = 1)
                new_dist = np.linalg.norm(new_pos - self._target_location, ord=1)
                if new_dist < old_dist: 
                    reward = 0.01 
                else: 
                    reward = -0.05


        truncated = self.step_count >= self.max_steps

        if self.render_mode == "human":
            self.render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render_frame(self):
        print("\n" + "=" * 50) 
        print(f"Grid World (size: {self.size}x{self.size}) | Step: {self.step_count}/{self.max_steps}")
        print("A = Agent, T = Target, X = Obstacle")
        print()
        
        for y in range(self.size - 1, -1, -1):
            row = "|"
            for x in range(self.size):
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
        print(f"\nAgent: {obs['agent']}, Target: {obs['target']}, Obstacles: {obs['obstacles']}")
        print(f"Distance: {info['distance']}, Steps remaining: {info['steps_remaining']}")
        print("=" * 50)
