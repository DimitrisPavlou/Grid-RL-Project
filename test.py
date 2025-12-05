"""Testing script for trained PPO and DQN agents."""
from stable_baselines3 import PPO, DQN
from helper_utils import test_trained_model


def main():
    """Main testing function."""
    # Environment parameters
    grid_size = (6, 6)
    max_steps = 50
    num_obstacles = 5

    # Testing parameters
    num_test_episodes = 1
    render_mode = True  # Set to True to see visual rendering
    print_frequency = 1  # Print every N episodes
    render_delay = 1  # Seconds to pause between frames (0.3-1.0 recommended)

    print("="*60)
    print("Loading trained models...")
    print("="*60)

    # Load trained models
    try:
        model_ppo = PPO.load("agent_parameters/ppo_gridenv")
        print("✓ PPO model loaded successfully")
    except FileNotFoundError:
        print("✗ PPO model not found at 'agent_parameters/ppo_gridenv'")
        return

    try:
        model_dqn = DQN.load("agent_parameters/dqn_gridenv")
        print("✓ DQN model loaded successfully")
    except FileNotFoundError:
        print("✗ DQN model not found at 'agent_parameters/dqn_gridenv'")
        return

    # ===== Test PPO =====
    print("\n" + "="*60)
    print("TESTING PPO AGENT")
    print("="*60)
    test_trained_model(
        model_ppo,
        grid_size,
        num_obstacles,
        max_steps,
        expanded=False,
        num_episodes=num_test_episodes,
        render=render_mode,
        print_every=print_frequency,
        render_delay=render_delay
    )

    # ===== Test DQN =====
    print("\n" + "="*60)
    print("TESTING DQN AGENT")
    print("="*60)
    test_trained_model(
        model_dqn,
        grid_size,
        num_obstacles,
        max_steps,
        expanded=False,
        num_episodes=num_test_episodes,
        render=render_mode,
        print_every=print_frequency,
        render_delay=render_delay
    )

    # ===== Test on Expanded Environment (Optional) =====
    # Uncomment to test on expanded environment:
    """
    print("\n" + "="*60)
    print("TESTING ON EXPANDED ENVIRONMENT")
    print("="*60)
    
    num_bonus = 2
    num_obstacles_expanded = 3
    
    try:
        model_ppo_exp = PPO.load("agent_parameters/ppo_expanded_gridenv")
        print("\n--- PPO on Expanded Environment ---")
        test_trained_model(
            model_ppo_exp,
            grid_size,
            num_obstacles_expanded,
            max_steps,
            num_bonus=num_bonus,
            expanded=True,
            num_episodes=num_test_episodes,
            render=False,
            print_every=print_frequency
        )
    except FileNotFoundError:
        print("✗ Expanded PPO model not found")
    
    try:
        model_dqn_exp = DQN.load("agent_parameters/dqn_expanded_gridenv")
        print("\n--- DQN on Expanded Environment ---")
        test_trained_model(
            model_dqn_exp,
            grid_size,
            num_obstacles_expanded,
            max_steps,
            num_bonus=num_bonus,
            expanded=True,
            num_episodes=num_test_episodes,
            render=False,
            print_every=print_frequency
        )
    except FileNotFoundError:
        print("✗ Expanded DQN model not found")
    """

    print("\n" + "="*60)
    print("All testing completed!")
    print("="*60)


if __name__ == "__main__":
    main()
