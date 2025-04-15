# Dual-Agent SAC in Rectangular Grid Environment

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
import os

torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
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

# Q-Network
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

# Actor
LOG_STD_MAX = 2
LOG_STD_MIN = -5

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
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, action_scale, action_bias , filename):
        self.actor = SACActor(state_dim, action_dim, action_scale, action_bias).to(device).float()
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(device).float()

        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.replay_buffer = ReplayBuffer()

        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_freq = 2
        self.load_checkpoint(filename)
        self.alpha = self.log_alpha.exp().item()


    def save_checkpoint(self, filename, global_steps):
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf1.state_dict(),
            'qf2_state_dict': self.qf2.state_dict(),
            'qf1_target_state_dict': self.qf1_target.state_dict(),
            'qf2_target_state_dict': self.qf2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'a_optimizer_state_dict': self.a_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'global_steps': global_steps,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.qf1.load_state_dict(checkpoint['qf1_state_dict'])
            self.qf2.load_state_dict(checkpoint['qf2_state_dict'])
            self.qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
            self.qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])

            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
            self.a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
            self.log_alpha = checkpoint['log_alpha']
            print(f"Checkpoint loaded from {filename}")
        else:
            print(f"No checkpoint found at {filename}, starting from scratch.")

    def select_action(self, state):
        # print(f"alpha values : {self.alpha}")
        with torch.no_grad():  # No need to track gradients during inference
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _ = self.actor.get_action(state)
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

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_states)
            target_q1 = self.qf1_target(next_states, next_actions)
            target_q2 = self.qf2_target(next_states, next_actions)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q_value = rewards + (1 - dones) * self.gamma * min_target_q

        q1 = self.qf1(states, actions)
        q2 = self.qf2(states, actions)
        q_loss = F.mse_loss(q1, target_q_value) + F.mse_loss(q2, target_q_value)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        if global_steps % self.policy_freq == 0:
            actions, log_pi = self.actor.get_action(states)
            q1_pi = self.qf1(states, actions)
            q2_pi = self.qf2(states, actions)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actions, log_pi = self.actor.get_action(states)
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        self.soft_update()

# MultiAgent Environment
class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()
        self.observation_space = Box(low=np.array([0, 0, 0, -np.pi, -np.pi]),
                                     high=np.array([0.5, 1.0, 2*np.pi, np.pi, np.pi]),
                                     dtype=np.float32)
        self.action_space = Box(low=np.array([-0.98]), high=np.array([0.98]), dtype=np.float32)

        self.v = 0.05
        self.dt = 0.1
        self.epsilon = 0.04
        self.alpha = 12
        self.gamma = 0
        self.reset()

    def reset(self):
        x1 = np.random.uniform(0.1, 0.5)
        x2 = np.random.uniform(0.1, 0.5)

        # y1=np.random.uniform(0.1,1)
        # y2=np.random.uniform(0.1,1)

        θ1 = np.random.uniform(0, 2*np.pi)
        θ2 = np.random.uniform(0, 2*np.pi)

        self.agent1_pos = np.array([x1, 1, θ1])
        self.agent2_pos = np.array([x2, 0.1, θ2])

        self.agent1_goal = np.copy(self.agent2_pos)
        self.agent2_goal = np.copy(self.agent1_pos)

        self._update_states()
        return self.agent1_state, self.agent2_state

    def _update_states(self):
        def get_state(agent_pos, goal, other_pos):
            θ_goal = math.atan2(goal[1] - agent_pos[1], goal[0] - agent_pos[0])
            θ_other = math.atan2(other_pos[1] - agent_pos[1], other_pos[0] - agent_pos[0])
            θ = agent_pos[2]
            angular_diff = ((θ_goal - θ + np.pi) % (2*np.pi)) - np.pi
            angular_diff2 = ((θ_other - θ + np.pi) % (2*np.pi)) - np.pi
            return np.array([agent_pos[0], agent_pos[1], θ, angular_diff, angular_diff2])

        self.agent1_state = get_state(self.agent1_pos, self.agent1_goal, self.agent2_pos)
        self.agent2_state = get_state(self.agent2_pos, self.agent2_goal, self.agent1_pos)

    def collision(self, a_pos, b_pos):
        for step in range(5):
            future_a = a_pos[:2] + self.v * np.array([np.cos(a_pos[2]), np.sin(a_pos[2])]) * step * self.dt
            future_b = b_pos[:2] + self.v * np.array([np.cos(b_pos[2]), np.sin(b_pos[2])]) * step * self.dt
            if np.linalg.norm(future_a - future_b) < 0.05:
                return True
        return False

    def _compute_reward(self, pos, goal, other_pos, angular_diff):
        dist = np.linalg.norm(pos[:2] - goal[:2])
        distbtwagents=np.linalg.norm(pos[:2]-other_pos[:2])
        if dist <= self.epsilon:
            return 150, True
        if pos[0] < 0 or pos[0] > 0.5 or pos[1] < 0 or pos[1] > 1.0:
            return -20, True
        reward = -self.alpha * abs(angular_diff) / np.pi
        if self.collision(pos, other_pos) and distbtwagents<0.05:
            reward -= 50
        reward+=-1 #penalising for every time step taken so that we minimise the time steps to reach the goal..
        return reward, False

    def step(self, action1, action2):
        def move(pos, action):
            dx = self.v * np.cos(pos[2]) * self.dt
            dy = self.v * np.sin(pos[2]) * self.dt
            pos[0] = np.clip(pos[0] + dx, 0.1, 0.5)
            pos[1] = np.clip(pos[1] + dy, 0.1, 1.0)
            pos[2] = (pos[2] + float(action) * self.dt) % (2*np.pi)
            return pos

        self.agent1_pos = move(self.agent1_pos, action1)
        self.agent2_pos = move(self.agent2_pos, action2)

        self._update_states()

        r1, d1 = self._compute_reward(self.agent1_pos, self.agent1_goal, self.agent2_pos, self.agent1_state[3])
        r2, d2 = self._compute_reward(self.agent2_pos, self.agent2_goal, self.agent1_pos, self.agent2_state[3])

        return (self.agent1_state, self.agent2_state), (r1, r2), (d1, d2)

# --- Training Loop ---
if __name__ == "__main__":
    env = MultiAgentEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(device)
    action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(device)

    agent1 = SAC(state_dim, action_dim, action_scale, action_bias , "agent1_checkpoint.pth")
    agent2 = SAC(state_dim, action_dim, action_scale, action_bias , "agent2_checkpoint.pth")

    global_steps = 0
    episode_rewards_1 = []
    episode_rewards_2 = []

    def signal_handler(sig, frame):
        print("Interrupt received! Cleaning up...")
        plt.close('all')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    plt.ion()
    fig_rewards, ax_rewards = plt.subplots()
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Total Reward")
    ax_rewards.set_title("Reward per Episode for Both Agents")

    fig_agents, ax_agents = plt.subplots(figsize=(6, 6))
    ax_agents.set_xlim(0, 0.5)
    ax_agents.set_ylim(0, 1.0)
    ax_agents.set_title("Agent Positions")
    ax_agents.set_xlabel("X")
    ax_agents.set_ylabel("Y")

    for episode in range(5001):
        state1, state2 = env.reset()
        episode_reward1 = 0
        episode_reward2 = 0

        pos1_x = [state1[0]]
        pos1_y = [state1[1]]
        pos2_x = [state2[0]]
        pos2_y = [state2[1]]

        print("-" * 120)
        print(f"Episode {episode + 1}")

        for t in range(200):
            action1 = agent1.select_action(state1)
            action2 = agent2.select_action(state2)

            (next_state1, next_state2), (reward1, reward2), (done1, done2) = env.step(action1, action2)

            agent1.replay_buffer.add(state1, action1, reward1, next_state1, done1)
            agent2.replay_buffer.add(state2, action2, reward2, next_state2, done2)

            state1, state2 = next_state1, next_state2
            episode_reward1 += reward1
            episode_reward2 += reward2
            global_steps += 1

            agent1.train(global_steps)
            agent2.train(global_steps)

            pos1_x.append(state1[0])
            pos1_y.append(state1[1])
            pos2_x.append(state2[0])
            pos2_y.append(state2[1])

            ax_agents.clear()
            ax_agents.set_xlim(0, 0.5)
            ax_agents.set_ylim(0, 1.0)
            ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
            ax_agents.set_xlabel("X")
            ax_agents.set_ylabel("Y")
            ax_agents.plot(pos1_x, pos1_y, color="red", label="Agent 1 Path")
            ax_agents.plot(pos2_x, pos2_y, color="blue", label="Agent 2 Path")
            ax_agents.scatter(env.agent1_goal[0], env.agent1_goal[1], color="red", marker="x", label="Agent 1 Goal")
            ax_agents.scatter(env.agent2_goal[0], env.agent2_goal[1], color="blue", marker="x", label="Agent 2 Goal")
            ax_agents.scatter(state1[0], state1[1], color="red", s=50)
            ax_agents.scatter(state2[0], state2[1], color="blue", s=50)
            ax_agents.legend()

            try:
                plt.pause(0.01)
            except KeyboardInterrupt:
                signal_handler(None, None)

            if done1 or done2:
                break

        episode_rewards_1.append(episode_reward1)
        episode_rewards_2.append(episode_reward2)

        ax_rewards.clear()
        ax_rewards.set_title("Reward per Episode for Both Agents")
        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Reward")
        ax_rewards.plot(episode_rewards_1, label="Agent 1")
        ax_rewards.plot(episode_rewards_2, label="Agent 2")
        ax_rewards.legend()

        try:
            plt.pause(0.01)
        except KeyboardInterrupt:
            signal_handler(None, None)

        print(f"Episode {episode + 1}: Agent1 Reward = {episode_reward1}, Agent2 Reward = {episode_reward2}")
        if (episode + 1) % 100 == 0:
            agent1.save_checkpoint("agent1_checkpoint.pth", global_steps)
            agent2.save_checkpoint("agent2_checkpoint.pth", global_steps)

    plt.ioff()
    while True:
        user_input = input("Press 1 to close the figures: ")
        if user_input == "1":
            plt.close('all')
            break
