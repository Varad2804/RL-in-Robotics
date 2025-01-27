import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
import signal
import sys



# Custom environment for the problem
class TwoVehiclesEnv(gym.Env):
    def __init__(self, scenario=1):
        super(TwoVehiclesEnv, self).__init__()

        # State space: [x1, y1, θ1, x2, y2, θ2]
        self.observation_space = Box(low=np.array([0.5, 0.5, 0, 0.5, 0.5, 0]),
                                     high=np.array([1, 1, 2 * np.pi, 1, 1, 2 * np.pi]),
                                     dtype=np.float32)

        # Action space: [angular_velocity1, angular_velocity2]
        self.action_space = Box(low=np.array([-0.78, -0.78]),
                                high=np.array([0.78, 0.78]),
                                dtype=np.float32)

        # Constant speed and minimum distance for collision
        self.v = 0.04  # Constant speed
        self.d_min = 0.1  # Minimum allowed distance between vehicles
        self.dt = 0.1  # Time step

        # Set scenario
        self.scenario = scenario
        self.start_states, self.goal_states = self._get_scenario(scenario)
        self.state = None

    def _get_scenario(self, scenario):
        """
        Defines different start and goal states for various scenarios.
        """
        if scenario == 1:
            # Scenario 1: Challenging diagonal crossing
            start_states = np.array([0.6, 0.6, np.pi/4, 0.89, 0.89, (5*np.pi)/4])
            goal_states = np.array([0.89, 0.89, 0.6, 0.6])
        elif scenario == 2:
            # Scenario 2: Opposite diagonal crossing
            start_states = np.array([0.64, 0.64, np.pi/4, 0.64, 0.86, (7*np.pi)/4])
            goal_states = np.array([0.93, 0.90, 0.9, 0.6])
        else:
            raise ValueError("Invalid scenario! Choose scenario 1 or 2.")
        return start_states, goal_states

    def reset(self):
        """
        Resets the environment to the start state.
        """
        self.state = self.start_states.copy()
        return self.state

    def step(self, action):
        """
        Executes one step of simulation.
        """
        x1, y1, θ1, x2, y2, θ2 = self.state
        a1, a2 = action

        # Update positions using old θ
        new_x1 = x1 + self.v * np.cos(θ1) * self.dt
        new_y1 = y1 + self.v * np.sin(θ1) * self.dt
        new_x2 = x2 + self.v * np.cos(θ2) * self.dt
        new_y2 = y2 + self.v * np.sin(θ2) * self.dt

        # Handle boundary constraints for both vehicles
        new_x1 = max(0.50, min(1, new_x1))
        new_y1 = max(0.50, min(1, new_y1))
        new_x2 = max(0.50, min(1, new_x2))
        new_y2 = max(0.50, min(1, new_y2))

        # Check for collision
        distance = np.sqrt((new_x1 - new_x2)**2 + (new_y1 - new_y2)**2)
        if distance <= self.d_min:
            # Collision detected: Freeze positions
            new_x1, new_y1 = x1, y1
            new_x2, new_y2 = x2, y2

        # Update orientations
        θ1 = (θ1 + a1 * self.dt) % (2 * np.pi)
        θ2 = (θ2 + a2 * self.dt) % (2 * np.pi)

        # Update the state
        self.state = np.array([new_x1, new_y1, θ1, new_x2, new_y2, θ2])

        # Compute reward
        reward = self._compute_reward()

        # Check termination condition
        done = reward == -100 or reward == 100

        return self.state, reward, done, {}

    def _compute_reward(self):
        """
        Computes reward based on current state.
        """
        x1, y1, θ1, x2, y2, θ2 = self.state
        gx1, gy1, gx2, gy2 = self.goal_states

        # Compute distance between vehicles
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # Forbidden state (collision)
        if d <= self.d_min:
            return -100

        # Goal state (both vehicles on goal positions)
        goal1 = np.sqrt((x1 - gx1)**2 + (y1 - gy1)**2) < 0.05
        goal2 = np.sqrt((x2 - gx2)**2 + (y2 - gy2)**2) < 0.05
        if goal1 and goal2:
            return 100

        # Default penalty
        return -1


class OUNoise:
    def __init__(self, action_dim, action_low, action_high, mu=0.0, theta=0.15, sigma=0.2, decay_rate=0.99):
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.decay_rate = decay_rate
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        scaled_noise = self.action_range * self.state
        return np.clip(scaled_noise, self.action_low, self.action_high)

    def decay_sigma(self):
        self.sigma *= self.decay_rate


class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)

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

    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.layer(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state, action):
        return self.layer(torch.cat([state, action], dim=1))


class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).float()
        self.actor_target = Actor(state_dim, action_dim, max_action).float()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).float()
        self.critic_target = Critic(state_dim, action_dim).float()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer()

        self.noise = OUNoise(action_dim, -max_action, max_action)

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise:
            action += self.noise.sample()
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64, discount=0.99, tau=0.005):
        if self.replay_buffer.size() < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).reshape(-1, 1)

        # Compute target Q
        target_Q = self.critic_target(next_states, self.actor_target(next_states))
        target_Q = rewards + (1 - dones) * discount * target_Q.detach()

        # Critic loss
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Main training loop
scenario = 1  # Choose scenario 1 or 2
env = TwoVehiclesEnv(scenario=scenario)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG(state_dim, action_dim, max_action)



def signal_handler(sig, frame):
    print("\nInterrupt received! Cleaning up...")
    plt.close('all')  # Close all open figures
    sys.exit(0)  # Exit the program cleanly

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Reward visualization setup
episode_rewards = []
plt.ion()  # Turn on interactive mode for dynamic updating
fig_rewards, ax_rewards = plt.subplots()
ax_rewards.set_xlabel("Episode")
ax_rewards.set_ylabel("Total Reward")
ax_rewards.set_title("Dynamic Plot of Total Reward per Episode")

# Real-time agent movement visualization setup
fig_agents, ax_agents = plt.subplots(figsize=(6, 6))
ax_agents.set_xlim(0.5, 1)
ax_agents.set_ylim(0.5, 1)
ax_agents.set_title("Agent Simulation During Training")
ax_agents.set_xlabel("X")
ax_agents.set_ylabel("Y")

# Main training loop
for episode in range(20):
    state = env.reset()
    agent.noise.reset()  # Reset noise at the start of each episode
    episode_reward = 0

    # Initialize the plot for agents
    ax_agents.clear()
    ax_agents.set_xlim(0.5, 1)
    ax_agents.set_ylim(0.5, 1)
    ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
    ax_agents.set_xlabel("X")
    ax_agents.set_ylabel("Y")

    # Extract goal states
    goal_states = env.goal_states
    ax_agents.scatter(goal_states[0], goal_states[1], color='green', label="Goal 1", marker="X", s=100)
    ax_agents.scatter(goal_states[2], goal_states[3], color='purple', label="Goal 2", marker="X", s=100)

    # Track positions of both vehicles
    positions1 = [[state[0]], [state[1]]]  # Vehicle 1
    positions2 = [[state[3]], [state[4]]]  # Vehicle 2

    for t in range(200):
        # Select action using the agent's policy
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Train the agent
        agent.train()

        # Update positions
        positions1[0].append(next_state[0])
        positions1[1].append(next_state[1])
        positions2[0].append(next_state[3])
        positions2[1].append(next_state[4])

        # Update agent movement plot dynamically
        ax_agents.clear()
        ax_agents.set_xlim(0.5, 1)
        ax_agents.set_ylim(0.5, 1)
        ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
        ax_agents.set_xlabel("X")
        ax_agents.set_ylabel("Y")

        # Plot goals
        ax_agents.scatter(goal_states[0], goal_states[1], color='green', label="Goal 1", marker="X", s=100)
        ax_agents.scatter(goal_states[2], goal_states[3], color='purple', label="Goal 2", marker="X", s=100)

        # Plot paths and current positions
        ax_agents.plot(positions1[0], positions1[1], color='red', label="Vehicle 1 Path")
        ax_agents.plot(positions2[0], positions2[1], color='blue', label="Vehicle 2 Path")
        ax_agents.scatter(next_state[0], next_state[1], color='red', label="Vehicle 1", s=50)
        ax_agents.scatter(next_state[3], next_state[4], color='blue', label="Vehicle 2", s=50)

        ax_agents.legend()
        try:
            plt.pause(0.01)  # Pause to refresh the plot dynamically
        except KeyboardInterrupt:
            signal_handler(None, None)
        # Terminate if done
        if done:
            break

    # Decay noise after the episode
    agent.noise.decay_sigma()

    # Append reward and update reward plot
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
 # Pause for a short duration to refresh reward plot

    print(f"Episode {episode + 1}: Reward = {episode_reward}")

# Turn off interactive mode and show final plots
plt.ioff()
while True:
    user_input = input("Press 1 to close the figures: ")
    if user_input == "1":
        plt.close('all')
        break
