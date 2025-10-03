import numpy as np
import gymnasium as gym

class ExpandedGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, grid_size=(5, 5), max_steps=100, num_obstacles=3, num_bonus=2):
        # grid_size is now a tuple (rows, columns) or (height, width)
        self.grid_size = np.array(grid_size)
        self.rows = self.grid_size[0]
        self.cols = self.grid_size[1]
        self.max_steps = max_steps
        self.step_count = 0
        self.num_obstacles = num_obstacles
        self.num_bonus = num_bonus  # Number of bonus positions
        self._should_render = False  

        # Observation: agent, target, obstacles, bonus positions, and collected status
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, max(self.grid_size) - 1, shape=(2,), dtype=int),
            "target": gym.spaces.Box(0, max(self.grid_size) - 1, shape=(2,), dtype=int),
            "obstacles": gym.spaces.Box(0, max(self.grid_size) - 1, shape=(self.num_obstacles, 2), dtype=int),
            "bonus_positions": gym.spaces.Box(0, max(self.grid_size) - 1, shape=(self.num_bonus, 2), dtype=int),
            "bonus_collected": gym.spaces.MultiBinary(self.num_bonus)  # Track which bonuses are collected
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
        self._bonus_positions = None
        self._bonus_collected = None  # Boolean array tracking collected bonuses

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "obstacles": self._obstacles,
            "bonus_positions": self._bonus_positions,
            "bonus_collected": self._bonus_collected
        }

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1),
            "steps_remaining": self.max_steps - self.step_count,
            "bonus_remaining": self.num_bonus - np.sum(self._bonus_collected),
            "all_bonus_collected": np.all(self._bonus_collected)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Random agent
        self._agent_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
        
        # Random target (different from agent)
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.grid_size, size=2, dtype=int)

        # Random obstacles (no overlap with agent or target)
        self._obstacles = []
        while len(self._obstacles) < self.num_obstacles:
            pos = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            if (not np.array_equal(pos, self._agent_location) and 
                not np.array_equal(pos, self._target_location) and
                not any(np.array_equal(pos, o) for o in self._obstacles)):
                self._obstacles.append(pos)
        self._obstacles = np.array(self._obstacles, dtype=int)

        # Random bonus positions (no overlap with agent, target, or obstacles)
        self._bonus_positions = []
        while len(self._bonus_positions) < self.num_bonus:
            pos = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            if (not np.array_equal(pos, self._agent_location) and 
                not np.array_equal(pos, self._target_location) and
                not any(np.array_equal(pos, o) for o in self._obstacles) and
                not any(np.array_equal(pos, b) for b in self._bonus_positions)):
                self._bonus_positions.append(pos)
        self._bonus_positions = np.array(self._bonus_positions, dtype=int)
        
        # Initialize bonus collected status (all False at start)
        self._bonus_collected = np.zeros(self.num_bonus, dtype=bool)

        if self._should_render:
            self.render_frame()

        return self._get_obs(), self._get_info()

    def _is_out_of_bounds(self, position):
        """Check if position is out of grid bounds"""
        return (position[0] < 0 or position[0] >= self.cols or 
                position[1] < 0 or position[1] >= self.rows)

    def step(self, action):
        self.step_count += 1
        action = int(action)
        direction = self._action_to_direction[action]

        new_pos = np.clip(self._agent_location + direction, [0, 0], self.grid_size - 1)

        if self._is_out_of_bounds(new_pos):
            reward = -1
            terminated = True  # Episode ends immediately when stepping out of bounds
            self._agent_location = new_pos  # Still update position for rendering
        else:
            # Check collision with obstacles
            if any(np.array_equal(new_pos, o) for o in self._obstacles):
                reward = -1
                terminated = True  # Episode ends immediately
                self._agent_location = new_pos  # Still update position for rendering
            else:
                self._agent_location = new_pos
                
                # Check if agent collected any bonus
                bonus_collected_this_step = False
                for i, bonus_pos in enumerate(self._bonus_positions):
                    if (not self._bonus_collected[i] and 
                        np.array_equal(self._agent_location, bonus_pos)):
                        self._bonus_collected[i] = True
                        bonus_collected_this_step = True
                
                # Check if reached target
                reached_target = np.array_equal(self._agent_location, self._target_location)
                
                # Only terminate if reached target AND collected all bonuses
                all_bonus_collected = np.all(self._bonus_collected)
                terminated = reached_target and all_bonus_collected
                
                # Calculate reward
                if terminated:
                    reward = 1.0  # Successfully reached target after collecting all bonuses
                elif reached_target and not all_bonus_collected:
                    reward = -1.0  # Reached target prematurely without collecting all bonuses
                    terminated = True  # End episode if reached target without all bonuses
                elif bonus_collected_this_step:
                    reward = 0.5  # Reward for collecting a bonus
                else:
                    # Distance-based reward
                    old_dist = np.linalg.norm(self._agent_location - direction - self._target_location, ord=1)
                    new_dist = np.linalg.norm(new_pos - self._target_location, ord=1)
                    if new_dist < old_dist:
                        reward = 0.01
                    else:
                        reward = -0.05

        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def enable_rendering(self):
        """Enable rendering for this environment instance"""
        self._should_render = True

    def disable_rendering(self):
        """Disable rendering for this environment instance"""
        self._should_render = False

    def render_frame(self):
        print("\n" + "=" * 50) 
        print(f"Grid World (size: {self.rows}x{self.cols}) | Step: {self.step_count}/{self.max_steps}")
        print("A = Agent, T = Target, X = Obstacle, B = Bonus (uncollected)")
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
                    # Check if this is an uncollected bonus position
                    is_bonus = False
                    for i, bonus_pos in enumerate(self._bonus_positions):
                        if (np.array_equal(pos, bonus_pos) and 
                            not self._bonus_collected[i]):
                            row += " B |"
                            is_bonus = True
                            break
                    if not is_bonus:
                        row += " . |"
            print(row)

        obs = self._get_obs()
        info = self._get_info()
        print(f"\nAgent: {obs['agent']}, Target: {obs['target']}")
        print(f"Obstacles: {obs['obstacles']}")
        print(f"Bonus positions: {obs['bonus_positions']}")
        print(f"Bonus collected: {obs['bonus_collected']}")
        print(f"Distance: {info['distance']}, Steps remaining: {info['steps_remaining']}")
        print(f"Bonus remaining: {info['bonus_remaining']}, All bonus collected: {info['all_bonus_collected']}")
        print("=" * 50)