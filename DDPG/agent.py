import copy
import random
from collections import namedtuple, deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


GAMMA = 0.99            # Discount factor.
TAU = 1e-3              # Soft update of target parameters.
LR_ACTOR = 1e-3         # Actor's learning rate.
LR_CRITIC = 1e-3        # Critic's learning rate.
WEIGHT_DECAY = 0.0000   # L2 weight decay.
BATCH_SIZE = 1024       # Batch size.
BUFFER_SIZE = int(1e6)  # Replay buffer size.


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


_eps = 3e-3


_pos_neg = lambda x: (-x, x)
_initialization = lambda x: _pos_neg(1. / np.sqrt(x.weight.data.size()[0]))


class Actor(nn.Module):
    """Actor Network."""

    def __init__(self, state_size, action_size, n_h1, n_h2, seed):
        """Compose the network.

        Params
        ------
        state_size: int
            # of input values.
        action_size: int
            # of output values.
        n_h1: int
            # of neurons in first hidden layer.
        n_h2: int
            # of neurons in second hidden layer.
        seed: int
            Seed for replication purposes.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, n_h1)
        self.bn1 = nn.BatchNorm1d(n_h1)
        self.fc2 = nn.Linear(n_h1, n_h2)
        self.fc3 = nn.Linear(n_h2, action_size)
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(*_initialization(self.fc1))
        self.fc2.weight.data.uniform_(*_initialization(self.fc2))
        self.fc3.weight.data.uniform_(-_eps, _eps)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic Network."""

    def __init__(self, state_size, action_size, n_h1, n_h2, seed):
        """Compose the network.

        Params
        ------
        state_size: int
            # of input values.
        action_size: int
            # of output values.
        n_h1: int
            # of neurons in first hidden layer.
        n_h2: int
            # of neurons in second hidden layer.
        seed: int
            Seed for replication purposes.
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, n_h1)
        self.bn1 = nn.BatchNorm1d(n_h1)
        self.fc2 = nn.Linear(n_h1 + action_size, n_h2)
        self.fc3 = nn.Linear(n_h2, 1)
        self.init_parameters()

    def init_parameters(self):
        self.fcs1.weight.data.uniform_(*_initialization(self.fcs1))
        self.fc2.weight.data.uniform_(*_initialization(self.fc2))
        self.fc3.weight.data.uniform_(-_eps, _eps)

    def forward(self, state, action):
        x = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent(object):
    """Agent to learn the policy."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ------
        state_size: int
            # of input values.
        action_size: int
            # of output values.
        seed: int
            Seed for replication purposes.
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Current and target actor networks.
        self.actor_local = Actor(state_size, action_size, 256, 128, seed).to(device)
        self.actor_target = Actor(state_size, action_size, 256, 128, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Current and target critic networks.
        self.critic_local = Critic(state_size, action_size, 256, 128, seed).to(device)
        self.critic_target = Critic(state_size, action_size, 256, 128, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Replay buffer.
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)

        # Ornstein-Uhlenbeck process to generate a noise signal.
        self.noise = OUNoise(action_size)
    
    def step(self, state, action, reward, next_state, done):
        """Add experience to the replay buffer (with random sampling)."""
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state, add_noise=True):
        """Returns action for given state with noise."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def start_learn(self):
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def learn(self, experiences, gamma):
        """Update policy using the given batch of experiences.

        Where the input / output of the netowrks are:
        actor(state): action
        critic(state, action): Q-value

        The critic uses the actor's Q-value of the next state for the update:
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

        Params
        ------
        experiences: Tuple[Tensor]
            Tuples like: (s, a, r, s', done).
        gamma: float
            Discount factor.
        """
        states, actions, rewards, next_states, dones = experiences

        # Critic network
        # Next-state actions (from actor) and Q-values (critic) from targets.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Q-values targets for current states (y_i).
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Critic loss.
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Optimization.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Actor network loss.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Optimization.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks.
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update target model from local mode and tau parameter.

        target = tau * local + (1 - tau) * target

        Params
        ------
        local_model: network
            To copy from.
        target_model: network
            To copy to.
        tau: float
            Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Set internal state to mean."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer(object):
    """Buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Create a ReplayBuffer object.

        Params
        ------
        buffer_size: int
            Size of the buffer.
        batch_size: int
            Size of training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
                'Experience',
                field_names=['state', 'action', 'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the buffer size."""
        return len(self.memory)
