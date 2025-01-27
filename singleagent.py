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
import math



# Custom environment for the problem
class TwoVehiclesEnv(gym.Env):
    def __init__(self, scenario=1):
        super(TwoVehiclesEnv, self).__init__()

        # State space: [x1, y1, θ1, x2, y2, θ2]
        self.observation_space = Box(low=np.array([0.5, 0.5, 0]),
                                     high=np.array([1, 1, 2 * np.pi]),
                                     dtype=np.float32)

        # Action space: [angular_velocity1, angular_velocity2]
        self.action_space = Box(low=np.array([-0.78]),
                                high=np.array([0.78]),
                                dtype=np.float32)

        # Constant speed and minimum distance for collision
        self.v = 0.04  # Constant speed
        self.dt = 0.1  # Time step
        self.alpha = 10  # Weight for orientation reward
        self.beta = 1  # Weight for distance reward
        self.gamma = 1  # Penalty for each step
        self.epsilon = 0.1 

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
                start_states = np.array([
                        np.random.uniform(0.5, 1.0),  # Random x1
                        np.random.uniform(0.5, 1.0),  # Random y1
                        np.random.uniform(0, 2 * np.pi)  # Random θ1
                    ])
                goal_states = np.array([0.89, 0.89])

            elif scenario == 2:
                # Scenario 2: Opposite diagonal crossing
                start_states = np.array([
                        np.random.uniform(0.5, 1.0),  # Random x1
                        np.random.uniform(0.5, 1.0),  # Random y1
                        np.random.uniform(0, 2 * np.pi)  # Random θ1
                    ])
                goal_states = np.array([0.93, 0.90])
            else:
                raise ValueError("Invalid scenario! Choose scenario 1 or 2.")
            
            self.state = start_states.copy()
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
        x1, y1, θ1 = self.state
        a1 = float(action)

        # Update positions using old θ
        new_x1 = x1 + self.v * np.cos(θ1) * self.dt
        new_y1 = y1 + self.v * np.sin(θ1) * self.dt
       

        # Handle boundary constraints for both vehicles
        new_x1 = max(0.50, min(1, new_x1))
        new_y1 = max(0.50, min(1, new_y1))
       

      

        # Update orientations
        θ1 = (θ1 + a1 * self.dt) % (2 * np.pi)

        # print(new_x1)
        # print(θ1)
        # Update the state
        self.state = np.array([new_x1, new_y1, θ1])

        # Compute reward
        reward = self._compute_reward()

        # Check termination condition
        done = reward == -100 or reward == 100

        return self.state, reward, done, {}

    def _compute_reward(self):
        """
        Computes reward based on the current state.
        """


        # r = 100 if goal 
        # 10.cos(theta(agent) - theta(goal) + 10*(1/(0.1+d) - 1

        # max d = 0.7   1/0.7+0.1 = 1.25


        # Current state (x1, y1, θ1)
        x1, y1, θ1 = self.state

        # Goal state (gx1, gy1)
        gx1, gy1 = self.goal_states

        # Check if agent is at the goal
        distance_to_goal = math.sqrt((gx1 - x1)**2 + (gy1 - y1)**2)
        if distance_to_goal < self.epsilon:  # Goal reached if close enough
            return 100  # Goal reward

        # Orientation reward: Compute angle to goal
        θ_goal = math.atan2(gy1 - y1, gx1 - x1)
        orientation_reward = math.cos(θ1 - θ_goal)

        # Distance reward: Inverse of the distance
        distance_reward = 1 / (distance_to_goal + self.epsilon)

        # Combine rewards
        reward = (
            self.alpha * orientation_reward +  # Scaled orientation reward
            self.beta * distance_reward -      # Scaled distance reward
            self.gamma                         # Step penalty
        )

        return reward


class GaussianNoise:
    def __init__(self, action_dim, action_low, action_high, initial_fraction, decay_rate):
        """
        Gaussian noise for exploration with decaying standard deviation.
        Scales noise based on action range.
        """
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        # Compute action range and initial std based on a fraction of the range
        self.action_range = action_high - action_low
        self.std = initial_fraction * self.action_range  # Initial std as a fraction of the action range
        self.decay_rate = decay_rate  # Decay factor for std

    def sample(self):
        """
        Generate Gaussian noise scaled by the current standard deviation.
        Ensure the noise is rounded to 2 decimal places.
        """
        noise = np.random.normal(0, self.std, size=self.action_dim)  # Mean = 0, Std = self.std
        noise = np.clip(noise, -self.action_range / 2, self.action_range / 2)  # Clip noise to the action range
        return np.round(noise, 2)  # Round to 2 decimal places

    def decay_std(self):
        """
        Decay the standard deviation for reduced exploration over time.
        """
        self.std *= self.decay_rate



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

        self.noise = GaussianNoise(
            action_dim,
            action_low=-max_action,
            action_high=max_action,
            initial_fraction=0.2,  # 20% of the action range
            decay_rate=0.95      # Decay rate for noise
        )

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
for episode in range(50):
    env = TwoVehiclesEnv(scenario=scenario)
    state = env.reset()
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

    # Track positions of both vehicles
    positions1 = [[state[0]], [state[1]]]  # Vehicle 1

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
       
        # Update agent movement plot dynamically
        ax_agents.clear()
        ax_agents.set_xlim(0.5, 1)
        ax_agents.set_ylim(0.5, 1)
        ax_agents.set_title(f"Episode {episode + 1}: Agent Simulation")
        ax_agents.set_xlabel("X")
        ax_agents.set_ylabel("Y")

        # Plot goals
        ax_agents.scatter(goal_states[0], goal_states[1], color='green', label="Goal 1", marker="X", s=100)

        # Plot paths and current positions
        ax_agents.plot(positions1[0], positions1[1], color='red', label="Vehicle 1 Path")
        ax_agents.scatter(next_state[0], next_state[1], color='red', label="Vehicle 1", s=50)

        ax_agents.legend()
        try:
            plt.pause(0.01)  # Pause to refresh the plot dynamically
        except KeyboardInterrupt:
            signal_handler(None, None)
        # Terminate if done
        if done:
            break

    # Decay noise after the episode
    agent.noise.decay_std()

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
