import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from singleSAC import SAC
from singleSAC import TwoVehiclesEnv
# Make sure to import your SAC, TwoVehiclesEnv, and any other dependencies.
# from your_module import SAC, TwoVehiclesEnv  # Uncomment if you have them in a separate module

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Setup the Environment
# ---------------------------
env = TwoVehiclesEnv(scenario=1)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Calculate action scaling parameters
action_scale = torch.tensor(
    (env.action_space.high - env.action_space.low) / 2.0, 
    dtype=torch.float32
).to(device)
action_bias = torch.tensor(
    (env.action_space.high + env.action_space.low) / 2.0, 
    dtype=torch.float32
).to(device)

# ---------------------------
# Instantiate the SAC Agent
# ---------------------------
sac_agent = SAC(state_dim, action_dim, action_scale, action_bias)

# ---------------------------
# Load the Model Checkpoint
# ---------------------------
checkpoint_path = "sac_checkpoint.pth"  # Adjust the path if necessary
checkpoint = torch.load(checkpoint_path, map_location=device)

sac_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
sac_agent.qf1.load_state_dict(checkpoint['qf1_state_dict'])
sac_agent.qf2.load_state_dict(checkpoint['qf2_state_dict'])
sac_agent.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
sac_agent.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])
sac_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
sac_agent.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
sac_agent.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
sac_agent.log_alpha = checkpoint['log_alpha']
sac_agent.alpha = sac_agent.log_alpha.exp().item()
global_steps = checkpoint.get('global_steps', 0)

print("Checkpoint loaded successfully.")

# ---------------------------
# Inference Loop
# ---------------------------
plt.ion()  # Turn on interactive mode for dynamic updating
fig_rewards, ax_rewards = plt.subplots()
ax_rewards.set_xlabel("Episode")
ax_rewards.set_ylabel("Total Reward")
ax_rewards.set_title("Dynamic Plot of Total Reward per Episode")

# Real-time agent movement visualization setup
# fig_agents, ax_agents = plt.subplots(figsize=(6, 6))
# ax_agents.set_xlim(0.1, 1)
# ax_agents.set_ylim(0.1, 1)
# ax_agents.set_title("Agent Simulation During Training")
# ax_agents.set_xlabel("X")
# ax_agents.set_ylabel("Y")

# num_episodes = 5  # Run inference for 5 episodes
# for episode in range(num_episodes):
#     state, goal_state = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         # Use torch.no_grad() for inference to save memory and computations
#         with torch.no_grad():
#             action = sac_agent.select_action(state)
#         next_state, reward, done, _ = env.step(action)
#         total_reward += reward
#         state = next_state

#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# # ---------------------------
# # Optionally, Visualize a Single Episode
# # ---------------------------
# state, goal_state = env.reset()
# positions = [[state[0]], [state[1]]]

# plt.figure(figsize=(6, 6))
# plt.xlim(0.3, 1)
# plt.ylim(0.3, 1)
# plt.title("Inference: Agent Trajectory")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.scatter(goal_state[0], goal_state[1], color='green', marker='X', s=100, label="Goal")

# done = False
# while not done:
#     with torch.no_grad():
#         action = sac_agent.select_action(state)
#     next_state, reward, done, _ = env.step(action)
#     positions[0].append(next_state[0])
#     positions[1].append(next_state[1])
#     state = next_state

# plt.plot(positions[0], positions[1], color='red', label="Agent Path")
# plt.scatter(positions[0][-1], positions[1][-1], color='red', s=50, label="Final Position")
# plt.legend()
# plt.show()


# Setup for real-time visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0.1, 1)
ax.set_ylim(0.1, 1)
ax.set_title("Real-Time Agent Trajectory")
ax.set_xlabel("X")
ax.set_ylabel("Y")

num_episodes = 5  # Number of episodes to visualize
for episode in range(num_episodes):
    state, goal_state = env.reset()
    positions = [[state[0]], [state[1]]]

    # Plot goal position
    ax.scatter(goal_state[0], goal_state[1], color='green', marker='X', s=100, label="Goal" if episode == 0 else "")

    done = False
    while not done:
        with torch.no_grad():
            action = sac_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        positions[0].append(next_state[0])
        positions[1].append(next_state[1])
        state = next_state

        # Clear and update the plot dynamically
        ax.clear()
        ax.set_xlim(0.1, 1)
        ax.set_ylim(0.1, 1)
        ax.set_title(f"Episode {episode + 1}: Real-Time Agent Trajectory")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Replot the goal and agent's path
        ax.scatter(goal_state[0], goal_state[1], color='green', marker='X', s=100, label="Goal")
        ax.plot(positions[0], positions[1], color='red', label="Agent Path")
        ax.scatter(positions[0][-1], positions[1][-1], color='red', s=50, label="Current Position")

        ax.legend()
        plt.pause(0.01)  # Small pause to show the update

    print(f"Episode {episode + 1} completed.")

plt.show()  # Ensure the plot stays visible at the end
