import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import signal
import sys
import matplotlib.pyplot as plt
from collections import deque


torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, buffer_size=int(8000)):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, action_bias):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mean = nn.Linear(64, action_dim)
        self.fc_logstd = nn.Linear(64, action_dim)

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        # with torch.no_grad():
        #     mean_of_mean = mean.mean()
        #     mean_of_std = std.mean()
        #     # print(f"Mean : {mean_of_mean.item():.2f}")
            # print(f"Standard Deviation : {mean_of_std.item():.2f}")


        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class SAC:
    def __init__(self, state_dim, action_dim, action_scale, action_bias):
        self.actor = SACActor(state_dim, action_dim, action_scale, action_bias).to(device).float()
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(device).float()

        # Load initial weights into target networks
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.replay_buffer = ReplayBuffer()

        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Typically -dim(A) for continuous SAC
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_freq = 2

    def select_action(self, state):
        with torch.no_grad():  # No need to track gradients during inference
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _ = self.actor.get_action(state)
            # print(f"angular velocity  : f{action}")
        return action.detach().cpu().numpy().flatten()


    def soft_update(self):
        # Soft update target networks using the τ coefficient
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, global_steps, batch_size=128):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = (
            torch.FloatTensor(states).to(device),
            torch.FloatTensor(actions).to(device),
            torch.FloatTensor(rewards).reshape(-1, 1).to(device),
            torch.FloatTensor(next_states).to(device),
            torch.FloatTensor(dones).reshape(-1, 1).to(device),
        )

        # Compute target Q values
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_states)
            target_q1 = self.qf1_target(next_states, next_actions)
            target_q2 = self.qf2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q_value = rewards + (1 - dones) * self.gamma * min_target_q

        # Compute Q losses
        q1 = self.qf1(states, actions)
        q2 = self.qf2(states, actions)
        q_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Policy and entropy updates
        if global_steps % self.policy_freq == 0:
            for _ in range(self.policy_freq):
                actions, log_pi = self.actor.get_action(states)
                q1_pi = self.qf1(states, actions)
                q2_pi = self.qf2(states, actions)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (self.alpha * log_pi - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Automatic entropy tuning
                with torch.no_grad():
                    actions, log_pi = self.actor.get_action(states)
                alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            # Perform the soft update of target networks
        self.soft_update()


# ---- Custom Environment: TwoVehiclesEnv ---- #
class TwoVehiclesEnv(gym.Env):
    def __init__(self, scenario=1):
        super(TwoVehiclesEnv, self).__init__()
        self.observation_space = Box(low=np.array([0.3, 0.3, 0]),
                                     high=np.array([1, 1, 2 * np.pi]),
                                     dtype=np.float32)
        self.action_space = Box(low=np.array([-0.98]),
                                high=np.array([0.98]),
                                dtype=np.float32)
        self.v = 0.05
        self.dt = 0.1
        self.alpha = 10
        self.gamma = 0
        self.epsilon2 = 0.07
        self.scenario = scenario
        self.start_state, self.goal_state = self._get_scenario(scenario)
        self.state = None

    def _get_scenario(self, scenario):
        if scenario == 1:
            start_state = np.array([
                np.random.uniform(0.45, 0.85),
                np.random.uniform(0.45, 0.85),
                np.random.uniform(0, 2 * np.pi)
            ])
            goal_state = np.array([0.89, 0.89])
        elif scenario == 2:
                # Scenario 2: Opposite diagonal crossing
                start_state = np.array([
                        np.random.uniform(0.45, 0.85),  # Random x1
                        np.random.uniform(0.45, 0.85),  # Random y1
                        np.random.uniform(0, 2 * np.pi)  # Random θ1
                    ])
                goal_state = np.array([0.93, 0.90])
        else:
            raise ValueError("Invalid scenario! Choose scenario 1 or 2.")
        
        return start_state, goal_state

    def reset(self):
        self.start_state, self.goal_state = self._get_scenario(self.scenario)
        self.state = self.start_state.copy()
        return self.state , self.goal_state

    def step(self, action):
        x1, y1, θ1 = self.state
        delta_x = self.v * np.cos(θ1) * self.dt
        delta_y = self.v * np.sin(θ1) * self.dt
        new_x1 = max(0.30, min(1.0, x1 + delta_x))
        new_y1 = max(0.30, min(1.0, y1 + delta_y))
        θ1 = (θ1 + float(action) * self.dt) % (2 * np.pi)
        self.state = np.array([new_x1, new_y1, θ1])
        reward = self._compute_reward()
        done = reward == 250
        # done = reward == 250 or reward==
        return self.state, reward, done, {}

    def _compute_reward(self):
        x1, y1, θ1 = self.state
        gx1, gy1 = self.goal_state
        distance_to_goal = np.sqrt((gx1 - x1) ** 2 + (gy1 - y1) ** 2)
        if distance_to_goal <= self.epsilon2:
            return 250
        if x1 <= 0.3 or x1 >= 1.0 or y1 <= 0.3 or y1 >= 1.0:
            return -200  # Heavy Penalty for going out of bounds
        θ_goal = math.atan2(gy1 - y1, gx1 - x1)
        angular_diff = abs(((θ_goal - θ1 + np.pi) % (2 * np.pi)) - np.pi )
        scaled_angular_diff = round(angular_diff / np.pi, 2)
        reward = (
            - self.alpha * scaled_angular_diff # Scaled orientation reward
            - self.gamma                         # Step penalty
        )
        return reward


# ---- Main SAC Training Loop ---- #
env = TwoVehiclesEnv(scenario=1)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(device)
action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(device)

sac_agent = SAC(state_dim, action_dim, action_scale, action_bias)
global_steps = 0

def signal_handler(sig, frame):
    print("\nInterrupt received! Cleaning up...")
    plt.close('all')  # Close all open figures
    sys.exit(0)  # Exit the program cleanly

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
episode_rewards = []

plt.ion()  # Turn on interactive mode for dynamic updating
fig_rewards, ax_rewards = plt.subplots()
ax_rewards.set_xlabel("Episode")
ax_rewards.set_ylabel("Total Reward")
ax_rewards.set_title("Dynamic Plot of Total Reward per Episode")

# Real-time agent movement visualization setup
fig_agents, ax_agents = plt.subplots(figsize=(6, 6))
ax_agents.set_xlim(0.3, 1)
ax_agents.set_ylim(0.3, 1)
ax_agents.set_title("Agent Simulation During Training")
ax_agents.set_xlabel("X")
ax_agents.set_ylabel("Y")


for episode in range(500):
    state , goal_state = env.reset()
    episode_reward = 0

    ax_agents.clear()
    ax_agents.set_xlim(0.3, 1)
    ax_agents.set_ylim(0.3, 1)
    ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
    ax_agents.set_xlabel("X")
    ax_agents.set_ylabel("Y")
    ax_agents.scatter(goal_state[0], goal_state[1], color='green', label="Goal 1", marker="X", s=100)
    positions1 = [[state[0]], [state[1]]] 
    print("------------------------------------------------------------------------------------------------------------")
    print("epsiode" + str(episode))
   
    for t in range(200):
        action = sac_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        sac_agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        episode_reward += reward
        global_steps += 1
        sac_agent.train(global_steps)
        
        positions1[0].append(next_state[0])
        positions1[1].append(next_state[1])

        ax_agents.clear()
        ax_agents.set_xlim(0.3, 1)
        ax_agents.set_ylim(0.3, 1)
        ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
        ax_agents.set_xlabel("X")
        ax_agents.set_ylabel("Y")
        ax_agents.scatter(goal_state[0], goal_state[1], color='green', label="Goal 1", marker="X", s=100)
        ax_agents.plot(positions1[0], positions1[1], color='red', label="Vehicle 1 Path")
        ax_agents.scatter(next_state[0], next_state[1], color='red', label="Vehicle 1", s=50)
        ax_agents.legend()

        try:
            plt.pause(0.01)  # Pause to refresh the plot dynamically
        except KeyboardInterrupt:
            signal_handler(None, None)
        
        
        if done:
            break

    episode_rewards.append(episode_reward)
    ax_rewards.clear()
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Total Reward")
    ax_rewards.set_title("Dynamic Plot of Total Reward per Episode")
    ax_rewards.plot(range(1, len(episode_rewards) + 1), episode_rewards, color="blue")
    
    try:
        plt.pause(0.01)  # Pause to refresh the plot dynamically
    except KeyboardInterrupt:
        signal_handler(None, None)
    print(f"Episode {episode + 1}: Reward = {episode_reward}")

plt.ioff()
while True:
    user_input = input("Press 1 to close the figures: ")
    if user_input == "1":
        plt.close('all')
        break

# At some point after training, save your checkpoint:
checkpoint = {
    'actor_state_dict': sac_agent.actor.state_dict(),
    'qf1_state_dict': sac_agent.qf1.state_dict(),
    'qf2_state_dict': sac_agent.qf2.state_dict(),
    'qf1_target_state_dict': sac_agent.qf1_target.state_dict(),
    'qf2_target_state_dict': sac_agent.qf2_target.state_dict(),
    'actor_optimizer_state_dict': sac_agent.actor_optimizer.state_dict(),
    'q_optimizer_state_dict': sac_agent.q_optimizer.state_dict(),
    'a_optimizer_state_dict': sac_agent.a_optimizer.state_dict(),
    'log_alpha': sac_agent.log_alpha,  # Note: log_alpha is a torch.Tensor with grad enabled
    'global_steps': global_steps,
}

torch.save(checkpoint, "sac_checkpoint.pth")
print("Checkpoint saved.")


