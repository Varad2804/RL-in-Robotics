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

# Set random seed and device
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Replay Buffer ---------------- #
class ReplayBuffer:
    def __init__(self, buffer_size=int(8000)):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))

# ---------------- Network Definitions ---------------- #
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
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

# ---------------- SAC Agent Definition ---------------- #
class SAC:
    def __init__(self, state_dim, action_dim, action_scale, action_bias, checkpoint_file=None):
        self.actor = SACActor(state_dim, action_dim, action_scale, action_bias).to(device).float()
        self.qf1 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2 = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf1_target = SoftQNetwork(state_dim, action_dim).to(device).float()
        self.qf2_target = SoftQNetwork(state_dim, action_dim).to(device).float()

        # Initialize target networks with same weights
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.replay_buffer = ReplayBuffer()

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.a_optimizer = optim.Adam([self.log_alpha], lr=1e-3)

        self.gamma = 0.99
        self.tau = 0.005
        self.policy_freq = 2

        if checkpoint_file:
            self.load_checkpoint(checkpoint_file)
            self.alpha = self.log_alpha.exp().item()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _ = self.actor.get_action(state)
        return action.detach().cpu().numpy().flatten()

    def soft_update(self):
        for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, global_steps, batch_size=128):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(device)

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
            for _ in range(self.policy_freq):
                actions_pi, log_pi = self.actor.get_action(states)
                q1_pi = self.qf1(states, actions_pi)
                q2_pi = self.qf2(states, actions_pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (self.alpha * log_pi - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                with torch.no_grad():
                    _, log_pi = self.actor.get_action(states)
                alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            self.soft_update()

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
        print("Checkpoint saved.")

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
            global_steps = checkpoint['global_steps']
            print("Checkpoint loaded.")
        else:
            print("No checkpoint found. Starting training from scratch.")

def check_out_of_bounds(x, y):
        if x <= 0.1 or x >= 0.5 or y <= 0.1 or y >= 1.0:
            return -20  # Heavy Penalty for going out of bounds
        return 0
# ---------------- Modified Two-Agents Environment with Collision Triangle ---------------- #
class TwoVehiclesEnv(gym.Env):
    """
    In this environment, there are two agents.
    - Agent 1’s goal is to approach Agent 2.
    - Agent 2’s goal is to approach Agent 1.
    Each agent’s observation is a 4D vector:
        [x, y, theta, angular_diff]
    where angular_diff is the difference between the agent’s heading and the angle toward the other agent.
    
    A collision triangle is checked by predicting future positions. If a collision triangle is detected,
    an extra negative reward is applied.
    """
    def __init__(self):
        super(TwoVehiclesEnv, self).__init__()
        low = np.array([0.1, 0.1, 0.0, -np.pi], dtype=np.float32)
        high = np.array([0.5, 1.0, 2*np.pi, np.pi], dtype=np.float32)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)
        self.action_space = Box(low=np.array([-0.98]), high=np.array([0.98]), dtype=np.float32)

        self.dt = 0.1
        self.v1 = 0.05  # Velocity for agent1
        self.v2 = 0.05  # Velocity for agent2
        self.alpha = 10
        self.epsilon = 0.04      # Success threshold (agents are considered to have reached each other)
        self.collision_penalty = 50  # Penalty applied if a collision triangle is detected

    def reset(self):
        # Sample initial states ensuring agents are not too close initially
        while True:
            self.agent1_state = np.array([
                np.random.uniform(0.1, 0.45),
                np.random.uniform(0.1, 1.0),
                np.random.uniform(0, 2 * np.pi)
            ], dtype=np.float32)
            self.agent2_state = np.array([
                np.random.uniform(0.1, 0.45),
                np.random.uniform(0.1, 1.0),
                np.random.uniform(0, 2 * np.pi)
            ], dtype=np.float32)
            if np.linalg.norm(self.agent1_state[:2] - self.agent2_state[:2]) > 0.3:
                break

        obs1 = self._get_obs(agent=1)
        obs2 = self._get_obs(agent=2)
        return obs1, obs2

    def _get_obs(self, agent=1):
        if agent == 1:
            x, y, theta = self.agent1_state
            goal_x, goal_y = self.agent2_state[:2]
        else:
            x, y, theta = self.agent2_state
            goal_x, goal_y = self.agent1_state[:2]

        desired_heading = math.atan2(goal_y - y, goal_x - x)
        angular_diff = ((desired_heading - theta + np.pi) % (2 * np.pi)) - np.pi
        return np.array([x, y, theta, angular_diff], dtype=np.float32)

    def collision_triangle(self, horizon=2.0, collision_threshold=0.05):
        """
        Predict future positions for both agents over the given horizon.
        If at any predicted step the distance between agents is less than collision_threshold,
        return True to indicate a collision triangle.
        """
        steps = int(horizon / self.dt)
        # Use at least 5 steps for prediction
        steps = max(steps, 5)
        x1, y1, theta1 = self.agent1_state
        x2, y2, theta2 = self.agent2_state

        for step in range(1, steps + 1):
            future_x1 = x1 + self.v1 * math.cos(theta1) * step * self.dt
            future_y1 = y1 + self.v1 * math.sin(theta1) * step * self.dt
            future_x2 = x2 + self.v2 * math.cos(theta2) * step * self.dt
            future_y2 = y2 + self.v2 * math.sin(theta2) * step * self.dt
            distance = math.sqrt((future_x1 - future_x2)**2 + (future_y1 - future_y2)**2)
            if distance < collision_threshold:
                return True
        return False
    
    def step(self, actions):
        """
        actions: tuple of two actions (action1 for agent1, action2 for agent2)
        """
        action1, action2 = actions

        # Update headings based on actions (steering)
        self.agent1_state[2] = (self.agent1_state[2] + float(action1) * self.dt) % (2 * np.pi)
        self.agent2_state[2] = (self.agent2_state[2] + float(action2) * self.dt) % (2 * np.pi)

        # Update positions using constant velocities
        for state, v in zip([self.agent1_state, self.agent2_state], [self.v1, self.v2]):
            delta_x = v * math.cos(state[2]) * self.dt
            delta_y = v * math.sin(state[2]) * self.dt
            state[0] = np.clip(state[0] + delta_x, 0.1, 0.5)
            state[1] = np.clip(state[1] + delta_y, 0.1, 1.0)

        # Compute observations for both agents
        obs1 = self._get_obs(agent=1)
        obs2 = self._get_obs(agent=2)

        # Compute Euclidean distance between the agents
        pos1 = self.agent1_state[:2]
        pos2 = self.agent2_state[:2]
        distance = np.linalg.norm(pos1 - pos2)

        x1, y1 = self.agent1_state[0], self.agent1_state[1]
        penalty1 = check_out_of_bounds(x1, y1)
        
        x2, y2 = self.agent2_state[0], self.agent2_state[1]
        penalty2=check_out_of_bounds(x2,y2)
        # Base reward: if within success threshold, give high positive reward;
        # otherwise penalize based on angular error.
        reward1 = 150 if distance <= self.epsilon else - self.alpha * (abs(obs1[3]) / np.pi)
        reward2 = 150 if distance <= self.epsilon else - self.alpha * (abs(obs2[3]) / np.pi)

        # Check for collision triangle. If detected, subtract collision penalty.
        done = False
        if self.collision_triangle():
            reward1 -= self.collision_penalty
            reward2 -= self.collision_penalty
            done = True  # End the episode if a collision triangle is formed

        # End episode if agents have reached each other
        if distance <= self.epsilon:
            done = True
        reward1+=penalty1
        reward2+=penalty2
        return (obs1, reward1, done), (obs2, reward2, done)

# ---------------- Main Training Loop ---------------- #
if __name__ == "__main__":
    env = TwoVehiclesEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = torch.tensor((env.action_space.high - env.action_space.low) / 2.0,
                                dtype=torch.float32).to(device)
    action_bias = torch.tensor((env.action_space.high + env.action_space.low) / 2.0,
                               dtype=torch.float32).to(device)

    # Two independent SAC agents
    sac_agent1 = SAC(state_dim, action_dim, action_scale, action_bias,"sac_agent1_checkpoint.pth")
    sac_agent2 = SAC(state_dim, action_dim, action_scale, action_bias,"sac_agent2_checkpoint.pth")
    global_steps = 0

    def signal_handler(sig, frame):
        print("\nInterrupt received! Cleaning up...")
        plt.close('all')
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    episode_rewards_agent1 = []
    episode_rewards_agent2 = []

    plt.ion()
    fig_rewards, ax_rewards = plt.subplots()
    ax_rewards.set_xlabel("Episode")
    ax_rewards.set_ylabel("Total Reward")
    ax_rewards.set_title("Total Rewards per Episode (Agent1 & Agent2)")

    fig_agents, ax_agents = plt.subplots(figsize=(6, 6))
    ax_agents.set_xlim(0.1, 0.5)
    ax_agents.set_ylim(0.1, 1)
    ax_agents.set_title("Agent Simulation During Training")
    ax_agents.set_xlabel("X")
    ax_agents.set_ylabel("Y")

    num_episodes = 2000
    for episode in range(num_episodes):
        obs1, obs2 = env.reset()
        episode_reward1 = 0
        episode_reward2 = 0

        positions1 = [[obs1[0]], [obs1[1]]]
        positions2 = [[obs2[0]], [obs2[1]]]

        ax_agents.clear()
        ax_agents.set_xlim(0.1, 0.5)
        ax_agents.set_ylim(0.1, 1)
        ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
        ax_agents.set_xlabel("X")
        ax_agents.set_ylabel("Y")

        print("---------------------------------------------------------")
        print("Episode", episode + 1)

        for t in range(200):
            action1 = sac_agent1.select_action(obs1)
            action2 = sac_agent2.select_action(obs2)

            (next_obs1, reward1, done), (next_obs2, reward2, _) = env.step((action1, action2))
            sac_agent1.replay_buffer.add(obs1, action1, reward1, next_obs1, done)
            sac_agent2.replay_buffer.add(obs2, action2, reward2, next_obs2, done)

            obs1 = next_obs1
            obs2 = next_obs2

            episode_reward1 += reward1
            episode_reward2 += reward2
            global_steps += 1

            sac_agent1.train(global_steps)
            sac_agent2.train(global_steps)

            positions1[0].append(obs1[0])
            positions1[1].append(obs1[1])
            positions2[0].append(obs2[0])
            positions2[1].append(obs2[1])

            # Update dynamic plot of agents
            ax_agents.clear()
            ax_agents.set_xlim(0.1, 0.5)
            ax_agents.set_ylim(0.1, 1)
            ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
            ax_agents.set_xlabel("X")
            ax_agents.set_ylabel("Y")
            ax_agents.plot(positions1[0], positions1[1], color='red', label="Agent 1 Path")
            ax_agents.plot(positions2[0], positions2[1], color='blue', label="Agent 2 Path")
            ax_agents.scatter(positions1[0][-1], positions1[1][-1], color='red', s=50)
            ax_agents.scatter(positions2[0][-1], positions2[1][-1], color='blue', s=50)
            ax_agents.legend()
            try:
                plt.pause(0.01)
            except KeyboardInterrupt:
                signal_handler(None, None)
            if done:
                break

        episode_rewards_agent1.append(episode_reward1)
        episode_rewards_agent2.append(episode_reward2)

        ax_rewards.clear()
        ax_rewards.set_xlabel("Episode")
        ax_rewards.set_ylabel("Total Reward")
        ax_rewards.set_title("Total Rewards per Episode (Agent1 & Agent2)")
        ax_rewards.plot(range(1, len(episode_rewards_agent1) + 1), episode_rewards_agent1, color="red", label="Agent 1")
        ax_rewards.plot(range(1, len(episode_rewards_agent2) + 1), episode_rewards_agent2, color="blue", label="Agent 2")
        ax_rewards.legend()
        try:
            plt.pause(0.01)
        except KeyboardInterrupt:
            signal_handler(None, None)

        print(f"Episode {episode + 1}: Agent1 Reward = {episode_reward1}, Agent2 Reward = {episode_reward2}")

        # Save checkpoints periodically (e.g., every 20 episodes)
        if (episode + 1) % 20 == 0:
            sac_agent1.save_checkpoint("sac_agent1_checkpoint.pth", global_steps)
            sac_agent2.save_checkpoint("sac_agent2_checkpoint.pth", global_steps)

    while True:
        user_input = input("Press 1 to close the figures: ")
        if user_input == "1":
            plt.close('all')
            break
