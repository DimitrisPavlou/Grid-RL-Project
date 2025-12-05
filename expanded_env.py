"""Expanded grid environment with bonus collection mechanic."""
import numpy as np
import gymnasium as gym
import time


class ExpandedGridEnv(gym.Env):
    """
    Grid environment where agent must collect all bonuses before reaching target.
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        grid_size=(5, 5),
        max_steps=100,
        num_obstacles=3,
        num_bonus=2,
        render_delay=0.3
    ):
        """
        Initialize the expanded grid environment.
        
        Args:
            render_mode: Rendering mode ('human', 'ansi', or None)
            grid_size: Tuple of (rows, cols) defining grid dimensions
            max_steps: Maximum steps per episode
            num_obstacles: Number of obstacle cells
            num_bonus: Number of bonus items to collect
            render_delay: Seconds to sleep after rendering (for visual clarity)
        """
        self.grid_size = np.array(grid_size, dtype=int)
        self.rows, self.cols = self.grid_size
        self.max_steps = max_steps
        self.step_count = 0
        self.num_obstacles = num_obstacles
        self.num_bonus = num_bonus
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
            ),
            "bonus_positions": gym.spaces.Box(
                0, max_coord,
                shape=(self.num_bonus, 2),
                dtype=int
            ),
            "bonus_collected": gym.spaces.MultiBinary(self.num_bonus)
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
        self._bonus_positions = None
        self._bonus_collected = None

    def _get_obs(self):
        """Get current observation."""
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
            "obstacles": self._obstacles.copy(),
            "bonus_positions": self._bonus_positions.copy(),
            "bonus_collected": self._bonus_collected.copy()
        }

    def _get_info(self):
        """Get auxiliary information."""
        return {
            "distance": float(np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )),
            "steps_remaining": self.max_steps - self.step_count,
            "bonus_remaining": int(self.num_bonus - np.sum(self._bonus_collected)),
            "all_bonus_collected": bool(np.all(self._bonus_collected))
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

        # Verify grid is large enough
        total_entities = 2 + self.num_obstacles + self.num_bonus  # agent + target + rest
        if total_entities > self.rows * self.cols:
            raise ValueError(
                f"Grid size {self.rows}x{self.cols} too small for "
                f"{total_entities} entities (agent, target, {self.num_obstacles} "
                f"obstacles, {self.num_bonus} bonuses)"
            )

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

        # Place obstacles (no overlap)
        occupied = [self._agent_location, self._target_location]
        self._obstacles = self._place_entities(
            self.num_obstacles, occupied, "obstacles"
        )
        occupied.extend(self._obstacles)

        # Place bonuses (no overlap)
        self._bonus_positions = self._place_entities(
            self.num_bonus, occupied, "bonuses"
        )
        
        # Initialize bonus collection status
        self._bonus_collected = np.zeros(self.num_bonus, dtype=bool)

        if self._should_render:
            self.render_frame()

        return self._get_obs(), self._get_info()

    def _place_entities(self, count, occupied, entity_name):
        """
        Place entities on grid avoiding occupied positions.
        
        Args:
            count: Number of entities to place
            occupied: List of already occupied positions
            entity_name: Name for error messages
            
        Returns:
            numpy array of entity positions
        """
        entities = []
        max_attempts = self.rows * self.cols * 10
        attempts = 0
        
        while len(entities) < count and attempts < max_attempts:
            pos = self.np_random.integers(0, self.grid_size, size=2, dtype=int)
            if not any(np.array_equal(pos, occ) for occ in occupied + entities):
                entities.append(pos)
            attempts += 1
            
        if len(entities) < count:
            raise RuntimeError(
                f"Could not place {count} {entity_name} in {self.rows}x{self.cols} "
                f"grid after {max_attempts} attempts. Grid may be too crowded."
            )
            
        return np.array(entities, dtype=int)

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
            
            # Check bonus collection
            bonus_collected_this_step = False
            for i, bonus_pos in enumerate(self._bonus_positions):
                if (not self._bonus_collected[i] and 
                    np.array_equal(self._agent_location, bonus_pos)):
                    self._bonus_collected[i] = True
                    bonus_collected_this_step = True
            
            # Check target arrival
            reached_target = np.array_equal(
                self._agent_location, self._target_location
            )
            all_bonus_collected = np.all(self._bonus_collected)
            
            # Calculate reward and termination
            if reached_target and all_bonus_collected:
                reward = 1.0
                terminated = True
            elif reached_target and not all_bonus_collected:
                reward = -1.0  # Penalty for reaching target prematurely
                terminated = True
            elif bonus_collected_this_step:
                reward = 0.5
                terminated = False
            else:
                # Distance-based reward
                if new_dist < old_dist:
                    reward = 0.01
                else:
                    reward = -0.05
                terminated = False

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
                    # Check for uncollected bonus
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
        print(f"Obstacles: {obs['obstacles'].tolist()}")
        print(f"Bonus positions: {obs['bonus_positions'].tolist()}")
        print(f"Bonus collected: {obs['bonus_collected'].tolist()}")
        print(
            f"Distance: {info['distance']:.1f}, "
            f"Steps remaining: {info['steps_remaining']}"
        )
        print(
            f"Bonus remaining: {info['bonus_remaining']}, "
            f"All collected: {info['all_bonus_collected']}"
        )
        print("=" * 50)
        
        # Sleep for visual clarity
        if self.render_delay > 0:
            time.sleep(self.render_delay)

    def close(self):
        """Clean up resources."""
        pass
