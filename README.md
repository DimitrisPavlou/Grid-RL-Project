# Grid-RL-Project
This is a repository for a Reinforcement Learning project with a focus on environment design, specifically a grid environment

To setup the project, install the packages with the required versions found in the requirements.txt file.

To train an agent I used the stable-baselines3 library. I opted for a Proximal Policy Optimization (PPO) agent and DQN agent. The train.py script has the agents already set up with their respective hyperparameters that were found beneficial for the convergence of the agents 

To test the performance of the trained agents, load the correscponding parameters of the agents from the agent_parameters folder and run the test.py script. There is an option for rendering the enviroment in the console to see the trained agent (marked as A) reach its target (marked as T) while avoiding the obstacles (marked as X). At the end of the n_episodes run the script prints out the average steps per episode that the agent needs to reach the target, the avergae reward and the success rate of the agent i.e. the percentage of solved enviroments.  


The main environment files are the env.py where the main environment is implemented and the expanded_env.py where the expanded environment with bonus positions is implemented. The simple_env.py is a simpler environment and is there for demonstration purposes. For further information on the design choices of the environment, read the assossiated pdf explaining the choices made, the RL algorithms picked and the challenges faced during this project
