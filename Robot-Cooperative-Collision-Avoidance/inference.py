import torch
import matplotlib.pyplot as plt
from singleSAC import MultiAgentEnv, SAC
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Environment
env = MultiAgentEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(device)
action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(device)

# Load trained agents
agent1 = SAC(state_dim, action_dim, action_scale, action_bias, "agent1_checkpoint.pth")
agent2 = SAC(state_dim, action_dim, action_scale, action_bias, "agent2_checkpoint.pth")

# Visualization setup
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Agent Inference Visualization")

def run_single_episode(ep_num):
    state1, state2 = env.reset()
    done1, done2 = False, False

    pos1_x, pos1_y = [state1[0]], [state1[1]]
    pos2_x, pos2_y = [state2[0]], [state2[1]]

    for t in range(200):
        ax.clear()
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Inference Episode {ep_num+1} - Step {t+1}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        action1 = agent1.select_action(state1)
        action2 = agent2.select_action(state2)

        (state1, state2), (_, _), (done1, done2) = env.step(action1, action2)

        pos1_x.append(state1[0])
        pos1_y.append(state1[1])
        pos2_x.append(state2[0])
        pos2_y.append(state2[1])

        ax.plot(pos1_x, pos1_y, color="red", label="Agent 1 Path")
        ax.plot(pos2_x, pos2_y, color="blue", label="Agent 2 Path")
        ax.scatter(env.agent1_goal[0], env.agent1_goal[1], color="red", marker="x", label="Agent 1 Goal")
        ax.scatter(env.agent2_goal[0], env.agent2_goal[1], color="blue", marker="x", label="Agent 2 Goal")
        ax.scatter(state1[0], state1[1], color="red", s=50)
        ax.scatter(state2[0], state2[1], color="blue", s=50)
        ax.legend()

        plt.pause(0.05)

        if done1 or done2:
            break

    print(f"Episode {ep_num + 1} completed in {t+1} steps.")
    time.sleep(0.5)

# Main loop to run multiple inference episodes
episode_counter = 0
while True:
    run_single_episode(episode_counter)
    episode_counter += 1

    user_input = input("Run another episode? (y/n): ").strip().lower()
    if user_input != 'y':
        break

plt.ioff()
plt.close()
