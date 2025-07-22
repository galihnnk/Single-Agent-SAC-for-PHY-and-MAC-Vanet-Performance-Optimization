import socket
import threading
import numpy as np
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, deque
import random
import sys
import os
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal
import math
import time
from datetime import datetime
import torch.serialization
import traceback
from collections import OrderedDict
import logging
import shutil
from typing import Dict, List, Tuple, Optional, Any, Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# ULTRA-AGGRESSIVE Constants and Hyperparameters
# ==============================

# Environment parameters
N_STEP = 3
CBR_TARGET = 0.65
SINR_TARGET = 18  # dB (initial target, will be adjusted based on neighbors)
POWER_MIN, POWER_MAX = 0, 30  # dBm
BEACON_RATE_MIN, BEACON_RATE_MAX = 1, 20  # Hz
MCS_MIN, MCS_MAX = 0, 10  # IEEE 802.11bd MCS indices
MAX_NEIGHBORS = 20  # Maximum expected neighbors for normalization

# ULTRA-AGGRESSIVE EXPLORATION PARAMETERS FOR FULL PARAMETER SPACE COVERAGE
# Required deltas for full exploration: Power(±29), Beacon(±19), MCS(±10)
# With 5x exploration factor: need base bounds of Power(±6), Beacon(±4), MCS(±2)
POWER_ACTION_BOUND = 8.0      # ±8 (was ±3) - MASSIVE increase
BEACON_ACTION_BOUND = 6.0     # ±6 (was ±1) - MASSIVE increase  
MCS_ACTION_BOUND = 3.0        # ±3 (was ±1) - MASSIVE increase

# Reward weights for single agent
W1, W2 = 3.0, 1.0  # CBR and power weights
W3, W4 = 2.5, 0.8  # SINR and MCS weights
W5 = 1.5  # Neighbor impact weight
BETA = 15  # CBR reward sharpness

# ULTRA-AGGRESSIVE SAC parameters for better exploration
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2  # Temperature parameter
LR = 3e-3 
HIDDEN_UNITS = 256

# ULTRA-EXPLORATIVE settings
INITIAL_EXPLORATION_FACTOR = 5.0   # MUCH higher than original (was 2.0)
EXPLORATION_DECAY = 0.9999         # Much slower decay (was 0.9995)
MIN_EXPLORATION = 0.8              # Much higher minimum (was 1.0)
EXPLORATION_NOISE_SCALE = 1.2      # Strong additional noise
ACTION_NOISE_SCALE = 0.8           # Action-dependent noise
RANDOM_ACTION_PROB = 0.15          # 15% random actions initially

# Enhanced entropy for exploration
INITIAL_LOG_STD = 1.0              # Much higher initial variance
LOG_STD_MIN = -8                   # Less restrictive
LOG_STD_MAX = 4                    # Higher maximum variance

# Network setup
HOST = '127.0.0.1'
PORT = 5000
LOG_DIR = 'ultra_aggressive_single_agent_logs'
MODEL_SAVE_DIR = "saved_models"
MODEL_SAVE_INTERVAL = 100  # Save every 100 episodes
SHARED_MODEL_PREFIX = "shared"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MAX_LOG_LENGTH = 200  # Max characters to log for data payloads

# Clear existing model directory
import shutil
if os.path.exists(MODEL_SAVE_DIR):
    shutil.rmtree(MODEL_SAVE_DIR)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ==============================
# Enhanced Logging Setup
# ==============================
def setup_logging():
    """Setup enhanced logging with multiple levels"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('ultra_aggressive_single_agent.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_message(message: str, level: str = "INFO"):
    """Backward compatibility logging function"""
    getattr(logger, level.lower())(message)

# ==============================
# Utility Functions
# ==============================
def safe_tensor_conversion(data: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Safely convert data to tensor with proper error handling"""
    try:
        if isinstance(data, torch.Tensor):
            return data.to(dtype)
        return torch.tensor(data, dtype=dtype)
    except Exception as e:
        logger.error(f"Tensor conversion failed: {e}")
        return torch.zeros(1, dtype=dtype)

def safe_normalize_state(state: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Safely normalize state with dimension checks"""
    try:
        if state.shape[-1] != mean.shape[-1]:
            logger.warning(f"Dimension mismatch in normalization: state {state.shape} vs mean {mean.shape}")
            return state
        
        return torch.clamp((state - mean) / (std + 1e-8), -5.0, 5.0)
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return state

# ==============================
# Prioritized Replay Buffer Implementation
# ==============================
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total
        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2
            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]
        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]

class PrioritizedReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, eps=1e-2, alpha=0.6, beta=0.4):
        self.tree = SumTree(size=buffer_size)
        self.eps = eps  # minimal priority
        self.alpha = alpha  # prioritization level  
        self.beta = beta  # importance sampling correction
        self.max_priority = eps  # priority for new samples
        
        # transition: state, action, reward, next_state, done
        self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, transition):
        state, action, reward, next_state, done = transition
        self.tree.add(self.max_priority, self.count)
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)
        segment = self.tree.total / batch_size
        
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.state[sample_idxs].to(device),
            self.action[sample_idxs].to(device),
            self.reward[sample_idxs].to(device),
            self.next_state[sample_idxs].to(device),
            self.done[sample_idxs].to(device)
        )
        return batch, weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.real_size

# ==============================
# Shared Model Manager
# ==============================
class SharedModelManager:
    @staticmethod
    def save_models(feature_extractor, agent, episode=None):
        """Save shared models with security considerations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_tag = f"_ep{episode}" if episode is not None else ""
        model_dir = f"{MODEL_SAVE_DIR}/{SHARED_MODEL_PREFIX}{episode_tag}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save with secure serialization
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'target_critic1_state_dict': agent.target_critic1.state_dict(),
                'target_critic2_state_dict': agent.target_critic2.state_dict(),
                'episode': episode,
                'timestamp': timestamp,
            }, f"{model_dir}/agent.pth", _use_new_zipfile_serialization=True)
            
            torch.save({
                'state_dict': feature_extractor.state_dict(),
                'episode': episode,
                'timestamp': timestamp,
            }, f"{model_dir}/feature_extractor.pth", _use_new_zipfile_serialization=True)
            
            log_message(f"Saved shared models at episode {episode} in {model_dir}")
            return True
            
        except Exception as e:
            log_message(f"Failed to save shared models: {str(e)}", "ERROR")
            return False

    @staticmethod
    def load_latest_models(feature_extractor, agent):
        """Securely load the latest shared models"""
        try:
            # Find latest model directory
            model_dirs = [d for d in os.listdir(MODEL_SAVE_DIR) 
                         if d.startswith(SHARED_MODEL_PREFIX)]
            if not model_dirs:
                log_message("No shared models found to load", "WARNING")
                return False
                
            latest_dir = max(model_dirs, key=lambda d: os.path.getmtime(f"{MODEL_SAVE_DIR}/{d}"))
            model_dir = f"{MODEL_SAVE_DIR}/{latest_dir}"
            
            # Secure loading with weights_only=True
            device_name = next(feature_extractor.parameters()).device
            
            # Load agent
            agent_checkpoint = torch.load(f"{model_dir}/agent.pth", 
                                      map_location=device_name,
                                      weights_only=True)
            agent.actor.load_state_dict(agent_checkpoint['actor_state_dict'])
            agent.critic1.load_state_dict(agent_checkpoint['critic1_state_dict'])
            agent.critic2.load_state_dict(agent_checkpoint['critic2_state_dict'])
            agent.target_critic1.load_state_dict(agent_checkpoint['target_critic1_state_dict'])
            agent.target_critic2.load_state_dict(agent_checkpoint['target_critic2_state_dict'])
            
            # Load feature extractor
            fe_checkpoint = torch.load(f"{model_dir}/feature_extractor.pth",
                                     map_location=device_name,
                                     weights_only=True)
            feature_extractor.load_state_dict(fe_checkpoint['state_dict'])
            
            log_message(f"Loaded shared models from {model_dir}")
            return True
            
        except Exception as e:
            log_message(f"Failed to load shared models: {str(e)}", "ERROR")
            return False

# ==============================
# ULTRA-EXPLORATIVE Neural Network Architectures
# ==============================
class ExploratorySharedFeatureExtractor(nn.Module):
    """Feature extractor optimized for exploration"""
    def __init__(self, state_dim):
        super().__init__()
        # State normalization
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        
        # Simplified network for better exploration
        self.fc1 = nn.Linear(state_dim, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        
        # Exploration-friendly initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier with higher gain
                nn.init.xavier_uniform_(m.weight.data, gain=2.0)
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.1, 0.1)
        
    def forward(self, x):
        # Normalize input with safety
        x = safe_normalize_state(x, self.state_mean, self.state_std)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

class UltraExplorativeActor(nn.Module):
    """ULTRA-EXPLORATIVE single agent policy network for ALL parameters"""
    def __init__(self, feature_dim):
        super().__init__()
        # Shared feature processing - simpler for more stochasticity
        self.fc1 = nn.Linear(feature_dim, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS // 2)
        
        # MASSIVELY INCREASED ACTION BOUNDS for full parameter space exploration
        # Power mean and log std - MUCH LARGER BOUNDS
        self.power_mean = nn.Linear(HIDDEN_UNITS // 2, 1)
        self.power_logstd = nn.Parameter(torch.full((1,), INITIAL_LOG_STD))
        
        # Beacon rate mean and log std - MUCH LARGER BOUNDS
        self.beacon_mean = nn.Linear(HIDDEN_UNITS // 2, 1)
        self.beacon_logstd = nn.Parameter(torch.full((1,), INITIAL_LOG_STD))
        
        # MCS mean and log std - MUCH LARGER BOUNDS
        self.mcs_mean = nn.Linear(HIDDEN_UNITS // 2, 1)
        self.mcs_logstd = nn.Parameter(torch.full((1,), INITIAL_LOG_STD))
        
        # Exploration-friendly initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for maximum exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization with higher gain
                nn.init.kaiming_uniform_(m.weight.data, a=0.2, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias.data, -0.2, 0.2)
        
        # Initialize mean layers with smaller weights for more exploration
        nn.init.uniform_(self.power_mean.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.beacon_mean.weight.data, -0.1, 0.1)
        nn.init.uniform_(self.mcs_mean.weight.data, -0.1, 0.1)
        nn.init.zeros_(self.power_mean.bias.data)
        nn.init.zeros_(self.beacon_mean.bias.data)
        nn.init.zeros_(self.mcs_mean.bias.data)
        
    def forward(self, x):
        """Forward pass for action means"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # MASSIVELY INCREASED ACTION BOUNDS
        # Power action (tanh bounds to [-1,1], then scaled to LARGE bounds)
        power_mean = torch.tanh(self.power_mean(x)) * POWER_ACTION_BOUND
        
        # Beacon action (tanh bounds to [-1,1], then scaled to LARGE bounds)
        beacon_mean = torch.tanh(self.beacon_mean(x)) * BEACON_ACTION_BOUND
        
        # MCS action (tanh bounds to [-1,1], then scaled to LARGE bounds)
        mcs_mean = torch.tanh(self.mcs_mean(x)) * MCS_ACTION_BOUND
        
        return torch.cat([power_mean, beacon_mean, mcs_mean], dim=-1)
    
    def sample(self, x):
        """Sample actions with MAXIMUM exploration"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # ENHANCED LOG STD CLAMPING for more exploration
        power_log_std = torch.clamp(self.power_logstd, LOG_STD_MIN, LOG_STD_MAX)
        beacon_log_std = torch.clamp(self.beacon_logstd, LOG_STD_MIN, LOG_STD_MAX)
        mcs_log_std = torch.clamp(self.mcs_logstd, LOG_STD_MIN, LOG_STD_MAX)
        
        # Power action with LARGE bounds
        power_mean = torch.tanh(self.power_mean(x)) * POWER_ACTION_BOUND
        power_std = torch.exp(power_log_std)
        power_noise = torch.randn_like(power_mean)
        power_action = power_mean + power_noise * power_std
        
        # Beacon action with LARGE bounds
        beacon_mean = torch.tanh(self.beacon_mean(x)) * BEACON_ACTION_BOUND
        beacon_std = torch.exp(beacon_log_std)
        beacon_noise = torch.randn_like(beacon_mean)
        beacon_action = beacon_mean + beacon_noise * beacon_std
        
        # MCS action with LARGE bounds
        mcs_mean = torch.tanh(self.mcs_mean(x)) * MCS_ACTION_BOUND
        mcs_std = torch.exp(mcs_log_std)
        mcs_noise = torch.randn_like(mcs_mean)
        mcs_action = mcs_mean + mcs_noise * mcs_std
        
        # Calculate log probabilities
        power_dist = Normal(power_mean, power_std)
        beacon_dist = Normal(beacon_mean, beacon_std)
        mcs_dist = Normal(mcs_mean, mcs_std)
        
        log_prob = power_dist.log_prob(power_action) + beacon_dist.log_prob(beacon_action) + mcs_dist.log_prob(mcs_action)
        
        # Stack actions
        actions = torch.stack([power_action.squeeze(-1), beacon_action.squeeze(-1), mcs_action.squeeze(-1)], dim=-1)
        
        return actions, log_prob

class QNetwork(nn.Module):
    """Enhanced critic network"""
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim + action_dim, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.q_value = nn.Linear(HIDDEN_UNITS, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)

# ==============================
# ULTRA-EXPLORATIVE Agent Class
# ==============================
class UltraExplorativeSingleAgent:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.actor = UltraExplorativeActor(HIDDEN_UNITS)
        self.critic1 = QNetwork(HIDDEN_UNITS, 3)  # 3 actions (power, beacon, MCS)
        self.critic2 = QNetwork(HIDDEN_UNITS, 3)
        self.target_critic1 = QNetwork(HIDDEN_UNITS, 3)
        self.target_critic2 = QNetwork(HIDDEN_UNITS, 3)
        
        # Initialize targets
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
            
        # Enhanced optimizers with higher learning rate
        enhanced_lr = LR * 1.5
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=enhanced_lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=enhanced_lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=enhanced_lr)
        
        # Enhanced replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            state_size=3,
            action_size=3,
            buffer_size=BUFFER_SIZE,
            alpha=0.6,
            beta=0.4
        )

        # Enhanced automatic entropy adjustment
        self.target_entropy = -torch.prod(torch.Tensor([3])).item() * 0.8  # Higher entropy target
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=enhanced_lr)
        
        # ULTRA-EXPLORATIVE parameters
        self.exploration_factor = INITIAL_EXPLORATION_FACTOR
        self.exploration_decay = EXPLORATION_DECAY
        self.min_exploration = MIN_EXPLORATION
        self.action_noise_std = ACTION_NOISE_SCALE
        self.random_action_prob = RANDOM_ACTION_PROB
        
        # Track action diversity for enhanced exploration
        self.recent_actions = deque(maxlen=100)
        self.step_count = 0
        
    def select_action(self, state_features):
        """ULTRA-EXPLORATIVE action selection for ALL parameters"""
        with torch.no_grad():
            if state_features.dim() == 1:
                state_features = state_features.unsqueeze(0)
                
            # Sample actions and get log probabilities
            actions, log_prob = self.actor.sample(state_features)
            
            # MULTIPLE ULTRA-AGGRESSIVE EXPLORATION MECHANISMS
            
            # 1. MULTIPLICATIVE EXPLORATION (like old script but more aggressive)
            if self.exploration_factor > 1.0:
                actions = actions * self.exploration_factor
                # Check for overflow or NaN
                actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
                actions = torch.clamp(actions, -100.0, 100.0)  # Allow very large exploration
            
            # 2. STRONG ADDITIVE NOISE
            noise_scale = self.exploration_factor * EXPLORATION_NOISE_SCALE
            exploration_noise = torch.randn_like(actions) * noise_scale
            actions = actions + exploration_noise
            
            # Check for NaN after noise addition
            actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
            
            # 3. DIVERSITY-BASED EXPLORATION
            if len(self.recent_actions) > 10:  # Reduced threshold for faster activation
                try:
                    recent_actions_list = list(self.recent_actions)[-20:]
                    if len(recent_actions_list) >= 2:
                        valid_actions = []
                        for action_tensor in recent_actions_list:
                            if action_tensor.shape == recent_actions_list[0].shape:
                                valid_actions.append(action_tensor)
                        
                        if len(valid_actions) >= 2:
                            recent_tensor = torch.stack(valid_actions)
                            # Use corrected=False to avoid degrees of freedom warning
                            action_std = torch.std(recent_tensor, dim=0, correction=0)
                            
                            # Ensure std is not zero or NaN
                            action_std = torch.clamp(action_std, min=0.1)
                            action_std = torch.where(torch.isnan(action_std), torch.ones_like(action_std), action_std)
                            
                            diversity_noise = torch.randn_like(actions) * self.action_noise_std * (3.0 / action_std)  # Increased multiplier
                            
                            # Check for NaN in diversity noise
                            diversity_noise = torch.where(torch.isnan(diversity_noise), torch.zeros_like(diversity_noise), diversity_noise)
                            actions = actions + diversity_noise
                except Exception as e:
                    logger.warning(f"Diversity exploration failed: {e}, skipping diversity noise")
            
            # 4. RANDOM ACTION EXPLORATION
            current_random_prob = self.random_action_prob * self.exploration_factor
            if random.random() < current_random_prob:
                # Generate completely random actions within enhanced bounds
                actions[0, 0] = torch.FloatTensor(1).uniform_(-POWER_ACTION_BOUND, POWER_ACTION_BOUND)
                actions[0, 1] = torch.FloatTensor(1).uniform_(-BEACON_ACTION_BOUND, BEACON_ACTION_BOUND)
                actions[0, 2] = torch.FloatTensor(1).uniform_(-MCS_ACTION_BOUND, MCS_ACTION_BOUND)
                
                logger.info(f"🎲 RANDOM EXPLORATION ACTIVATED: {actions.flatten().tolist()}")
            
            # Update exploration parameters
            self.exploration_factor *= self.exploration_decay
            self.exploration_factor = max(self.exploration_factor, self.min_exploration)
            
            # FINAL NaN PROTECTION: Ensure actions are not NaN
            actions = torch.where(torch.isnan(actions), torch.zeros_like(actions), actions)
            
            # Track actions for diversity
            self.recent_actions.append(actions.clone())
            self.step_count += 1
            
            return actions, log_prob
   
    def calculate_reward(self, cbr, sinr, current_power, current_mcs, neighbor_count):
        """Enhanced reward function with exploration bonus"""
        # Standard reward components
        cbr_error = abs(cbr - CBR_TARGET)
        cbr_term = W1 * (1 - math.tanh(BETA * cbr_error))
        
        power_norm = (current_power - POWER_MIN) / (POWER_MAX - POWER_MIN)
        power_term = W2 * power_norm ** 1.5
        
        adaptive_sinr_target = max(10, SINR_TARGET - (neighbor_count / 5))
        sinr_diff = sinr - adaptive_sinr_target
        sinr_term = W3 * math.tanh(sinr_diff/5.0)
        
        mcs_norm = (current_mcs - MCS_MIN) / (MCS_MAX - MCS_MIN)
        mcs_term = W4 * (1 - mcs_norm)
        
        neighbor_impact = W5 * (neighbor_count / MAX_NEIGHBORS)
        
        base_reward = cbr_term - power_term + sinr_term + mcs_term - neighbor_impact
        
        # EXPLORATION BONUS during early training
        if self.exploration_factor > 1.0:
            exploration_bonus = 0.2 * self.exploration_factor  # Increased bonus
            base_reward += exploration_bonus
        
        return base_reward

# ==============================
# Enhanced Vehicle Node Implementation
# ==============================
class UltraExplorativeVehicleNode:
    def __init__(self, node_id, feature_extractor, agent, training_mode=True):
        self.node_id = node_id
        self.training_mode = training_mode
        self.feature_extractor = feature_extractor
        self.agent = agent
        
        # Initialize parameters from incoming data (not hardcoded)
        self.current_power = None
        self.current_beacon_rate = None
        self.current_mcs = None
        
        # Enhanced performance tracking
        os.makedirs(LOG_DIR, exist_ok=True)
        self.writer = SummaryWriter(f"{LOG_DIR}/vehicle_{node_id}")
        self.episode_count = 0
        self.last_save_episode = 0
        self.debug_mode = True  # Enable for exploration monitoring
        self.train_counter = 0
        self.train_interval = 3  # More frequent training
        
        # Action diversity tracking
        self.action_history = {
            'power_deltas': deque(maxlen=200),
            'beacon_deltas': deque(maxlen=200),
            'mcs_deltas': deque(maxlen=200)
        }

    def get_actions(self, state, current_params):
        """Get ULTRA-EXPLORATIVE actions for ALL parameters with ULTRA-VERBOSE logging"""
        try:
            logger.info(f"🎯 STARTING ULTRA-EXPLORATIVE ACTION SELECTION FOR {self.node_id}")
            logger.info(f"📊 Input State: CBR={state[0]:.6f}, SNR={state[1]:.3f}, Neighbors={state[2]}")
            logger.info(f"🔧 Current Parameters: Power={current_params.get('transmissionPower', 20):.1f}dBm, Beacon={current_params.get('beaconRate', 10):.2f}Hz, MCS={current_params.get('MCS', 0)}")
            
            # Retrieve current values with safe defaults
            self.current_power = float(current_params.get('transmissionPower', 20))
            self.current_beacon_rate = float(current_params.get('beaconRate', 10))
            self.current_mcs = int(current_params.get('MCS', 0))
    
            logger.info(f"🔄 Updated Internal State: Power={self.current_power:.1f}, Beacon={self.current_beacon_rate:.2f}, MCS={self.current_mcs}")
    
            # Convert state to tensor with proper shape
            state_tensor = safe_tensor_conversion(np.array(state, dtype=np.float32))
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            logger.info(f"🧠 State tensor prepared: shape={state_tensor.shape}, values={state_tensor.flatten().tolist()}")
                
            if self.debug_mode:
                logger.info(f"🎲 AGENT EXPLORATION STATUS:")
                logger.info(f"   Current Exploration Factor: {self.agent.exploration_factor:.6f}")
                logger.info(f"   Action History Length: {len(self.agent.recent_actions)}")
                logger.info(f"   Recent Action Diversity: {self._calculate_action_diversity()}")
    
            # Inference from ULTRA-EXPLORATIVE agent
            with torch.no_grad():
                features = self.feature_extractor(state_tensor)
                
                logger.info(f"🧠 Features extracted: shape={features.shape}")
    
                # Get all actions from ULTRA-EXPLORATIVE single agent
                actions, log_prob = self.agent.select_action(features)
                
                logger.info(f"🎲 RAW ACTIONS GENERATED:")
                logger.info(f"   Raw Actions: {actions.flatten().tolist()}")
                logger.info(f"   Log Probability: {log_prob.item():.6f}")
                
                # Check for NaN immediately
                if torch.isnan(actions).any():
                    logger.warning(f"⚠️  NaN detected in actions: {actions}")
                    actions = torch.zeros_like(actions)
    
                # Get action values
                power_delta = actions[0, 0].item()
                beacon_delta = actions[0, 1].item()
                mcs_delta = actions[0, 2].item()
                
                # CRITICAL: Check for NaN values and replace with zeros
                if math.isnan(power_delta):
                    logger.warning(f"⚠️  NaN detected in power_delta, replacing with 0")
                    power_delta = 0.0
                if math.isnan(beacon_delta):
                    logger.warning(f"⚠️  NaN detected in beacon_delta, replacing with 0")
                    beacon_delta = 0.0
                if math.isnan(mcs_delta):
                    logger.warning(f"⚠️  NaN detected in mcs_delta, replacing with 0")
                    mcs_delta = 0.0
                
                # Clamp deltas to reasonable bounds
                power_delta = max(-20.0, min(20.0, power_delta))
                beacon_delta = max(-15.0, min(15.0, beacon_delta))
                mcs_delta = max(-8.0, min(8.0, mcs_delta))
                
                logger.info(f"🎲 FINAL ACTION DELTAS:")
                logger.info(f"   Power Delta: {power_delta:.6f}")
                logger.info(f"   Beacon Delta: {beacon_delta:.6f}")
                logger.info(f"   MCS Delta: {mcs_delta:.6f}")
                
                # Track action diversity
                self.action_history['power_deltas'].append(power_delta)
                self.action_history['beacon_deltas'].append(beacon_delta)
                self.action_history['mcs_deltas'].append(mcs_delta)
                
                # Calculate new parameters with clamping
                new_power = max(POWER_MIN, min(POWER_MAX, self.current_power + power_delta))
                new_beacon = max(BEACON_RATE_MIN, min(BEACON_RATE_MAX, 
                                    self.current_beacon_rate + beacon_delta))
                
                # Ensure mcs_delta is finite before rounding and converting
                mcs_delta_rounded = round(mcs_delta) if math.isfinite(mcs_delta) else 0
                new_mcs = max(MCS_MIN, min(MCS_MAX, self.current_mcs + mcs_delta_rounded))
                
                logger.info(f"⚙️  PARAMETER ADJUSTMENT CALCULATION:")
                logger.info(f"   Old Power: {self.current_power:.1f} dBm → New Power: {new_power:.1f} dBm (Δ={new_power-self.current_power:.1f})")
                logger.info(f"   Old Beacon: {self.current_beacon_rate:.2f} Hz → New Beacon: {new_beacon:.2f} Hz (Δ={new_beacon-self.current_beacon_rate:.2f})")
                logger.info(f"   Old MCS: {self.current_mcs} → New MCS: {new_mcs} (Δ={new_mcs-self.current_mcs})")
                
                if self.debug_mode:
                    # Log diversity metrics
                    if len(self.action_history['power_deltas']) > 10:
                        power_std = np.std(list(self.action_history['power_deltas'])[-50:])
                        beacon_std = np.std(list(self.action_history['beacon_deltas'])[-50:])
                        mcs_std = np.std(list(self.action_history['mcs_deltas'])[-50:])
                        logger.info(f"📊 ACTION DIVERSITY METRICS:")
                        logger.info(f"   Power std: {power_std:.3f}, Beacon std: {beacon_std:.3f}, MCS std: {mcs_std:.3f}")
    
            result = {
                'power_delta': float(power_delta),
                'beacon_delta': float(beacon_delta),
                'mcs_delta': int(mcs_delta_rounded),
                'log_prob': log_prob,
                'new_power': float(new_power),
                'new_beacon_rate': float(new_beacon),
                'new_mcs': int(new_mcs),
                'exploration_factor': self.agent.exploration_factor
            }
            
            logger.info(f"✅ ULTRA-EXPLORATIVE ACTIONS COMPLETE FOR {self.node_id}:")
            logger.info(f"   Power Delta: {result['power_delta']:.6f} → New Power: {result['new_power']:.1f}dBm")
            logger.info(f"   Beacon Delta: {result['beacon_delta']:.6f} → New Beacon: {result['new_beacon_rate']:.2f}Hz") 
            logger.info(f"   MCS Delta: {result['mcs_delta']} → New MCS: {result['new_mcs']}")
            logger.info(f"   Exploration Factor: {result['exploration_factor']:.6f}")
            
            return result
    
        except Exception as e:
            logger.error(f"❌ ULTRA-EXPLORATIVE Action selection failed for {self.node_id}: {str(e)}")
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Action selection failed: {str(e)}")

    def _calculate_action_diversity(self):
        """Calculate recent action diversity for logging"""
        try:
            if len(self.action_history['power_deltas']) < 10:
                return "insufficient_data"
            
            power_std = np.std(list(self.action_history['power_deltas'])[-20:])
            beacon_std = np.std(list(self.action_history['beacon_deltas'])[-20:])
            mcs_std = np.std(list(self.action_history['mcs_deltas'])[-20:])
            
            return f"Power:{power_std:.2f}, Beacon:{beacon_std:.2f}, MCS:{mcs_std:.2f}"
        except:
            return "calculation_error"

    def apply_actions(self, actions):
        """Apply the actions to adjust parameters for next timestamp"""
        try:
            # Calculate new parameters by applying deltas to current values
            new_power = self.current_power + actions['power_delta']
            new_beacon_rate = self.current_beacon_rate + actions['beacon_delta']
            new_mcs = self.current_mcs + actions['mcs_delta']
            
            # Clamp to valid ranges
            new_power = np.clip(new_power, POWER_MIN, POWER_MAX)
            new_beacon_rate = np.clip(new_beacon_rate, BEACON_RATE_MIN, BEACON_RATE_MAX)
            new_mcs = int(np.clip(round(new_mcs), MCS_MIN, MCS_MAX))
            
            logger.info(f"📤 APPLYING ACTIONS FOR {self.node_id}:")
            logger.info(f"   Final Power: {new_power:.1f} dBm")
            logger.info(f"   Final Beacon: {new_beacon_rate:.2f} Hz")
            logger.info(f"   Final MCS: {new_mcs}")
            
            return {
                'power': float(new_power),
                'beacon_rate': float(new_beacon_rate),
                'mcs': int(new_mcs)
            }
            
        except Exception as e:
            logger.error(f"❌ Action application failed: {str(e)}")
            raise RuntimeError(f"Action application failed: {str(e)}")

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in the replay buffer"""
        try:
            # Convert to tensors with proper dimensions
            state_tensor = safe_tensor_conversion(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                
            next_state_tensor = safe_tensor_conversion(next_state)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
                
            # Ensure action has proper dimensions
            if not isinstance(action, torch.Tensor):
                action = safe_tensor_conversion(action)
            if action.dim() == 1:
                action = action.unsqueeze(0)  # Add batch dimension
                
            reward_tensor = torch.FloatTensor([reward])
            done_tensor = torch.FloatTensor([float(done)])
            
            # Create the experience tuple
            experience = (state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            self.agent.replay_buffer.add(experience)
            
            logger.info(f"💾 EXPERIENCE STORED FOR {self.node_id}:")
            logger.info(f"   Replay buffer size: {len(self.agent.replay_buffer)}")
            logger.info(f"   Reward: {reward:.6f}")
            
            # Enhanced training frequency for faster exploration learning
            if (len(self.agent.replay_buffer) >= BATCH_SIZE and 
                self.train_counter % self.train_interval == 0):
                logger.info(f"🎓 TRIGGERING AGENT TRAINING FOR {self.node_id}")
                self.update_agent()
                
        except Exception as e:
            log_message(f"Error storing experience: {str(e)}", "ERROR")

    def update_agent(self):
        """Enhanced agent update with prioritized experience replay"""
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            return
            
        try:
            logger.info(f"🎓 STARTING AGENT UPDATE FOR {self.node_id}")
            
            # Sample batch with priorities
            batch, weights, indices = self.agent.replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch
            
            # Normalize rewards
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            with torch.no_grad():
                features = self.feature_extractor(states)
                next_features = self.feature_extractor(next_states)
                
                # Sample next actions
                next_actions, next_log_probs = self.agent.actor.sample(next_features)
                
                # Target Q values
                target_q1 = self.agent.target_critic1(next_features, next_actions)
                target_q2 = self.agent.target_critic2(next_features, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.agent.log_alpha.exp() * next_log_probs.unsqueeze(-1)
                target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * GAMMA * target_q

            # Update critics
            current_q1 = self.agent.critic1(features, actions)
            current_q2 = self.agent.critic2(features, actions)
            
            # Weighted losses for prioritized replay
            td_errors1 = current_q1 - target_q
            td_errors2 = current_q2 - target_q
            critic1_loss = (td_errors1.pow(2) * weights.unsqueeze(-1)).mean()
            critic2_loss = (td_errors2.pow(2) * weights.unsqueeze(-1)).mean()
            
            # Update priorities
            td_errors = torch.min(td_errors1.abs(), td_errors2.abs()).detach()
            self.agent.replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
            
            self.agent.critic1_optim.zero_grad()
            critic1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.critic1.parameters(), 1.0)
            self.agent.critic1_optim.step()
            
            self.agent.critic2_optim.zero_grad()
            critic2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.critic2.parameters(), 1.0)
            self.agent.critic2_optim.step()
            
            # Update actor
            new_actions, log_probs = self.agent.actor.sample(features)
            q1 = self.agent.critic1(features, new_actions)
            q2 = self.agent.critic2(features, new_actions)
            q = torch.min(q1, q2)
            
            actor_loss = (self.agent.log_alpha.exp().detach() * log_probs - q).mean()
            
            # Enhanced entropy adjustment
            alpha_loss = -(self.agent.log_alpha * (log_probs.detach() + self.agent.target_entropy)).mean()
            
            self.agent.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1.0)
            self.agent.actor_optim.step()
            
            self.agent.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.agent.alpha_optim.step()
            
            # Update target networks
            for target_param, param in zip(self.agent.target_critic1.parameters(), 
                                         self.agent.critic1.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                
            for target_param, param in zip(self.agent.target_critic2.parameters(), 
                                         self.agent.critic2.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            
            self.train_counter += 1
            
            logger.info(f"✅ AGENT UPDATE COMPLETE FOR {self.node_id}")
            logger.info(f"   Actor Loss: {actor_loss.item():.6f}")
            logger.info(f"   Critic1 Loss: {critic1_loss.item():.6f}")
            logger.info(f"   Critic2 Loss: {critic2_loss.item():.6f}")
            logger.info(f"   Alpha: {self.agent.log_alpha.exp().item():.6f}")
            
            # Enhanced logging every 25 updates
            if self.train_counter % 25 == 0:
                self._log_exploration_metrics()
                
        except Exception as e:
            log_message(f"Error updating agent: {str(e)}", "ERROR")

    def _log_exploration_metrics(self):
        """Log enhanced exploration metrics"""
        if self.debug_mode and len(self.action_history['power_deltas']) > 10:
            power_std = np.std(list(self.action_history['power_deltas'])[-50:])
            beacon_std = np.std(list(self.action_history['beacon_deltas'])[-50:])
            mcs_std = np.std(list(self.action_history['mcs_deltas'])[-50:])
            
            power_range = np.ptp(list(self.action_history['power_deltas'])[-50:])
            beacon_range = np.ptp(list(self.action_history['beacon_deltas'])[-50:])
            mcs_range = np.ptp(list(self.action_history['mcs_deltas'])[-50:])
            
            log_message(
                f"🚀 EXPLORATION METRICS - Vehicle {self.node_id}: "
                f"Exploration Factor: {self.agent.exploration_factor:.4f}, "
                f"Power std/range: {power_std:.3f}/{power_range:.1f}, "
                f"Beacon std/range: {beacon_std:.3f}/{beacon_range:.1f}, "
                f"MCS std/range: {mcs_std:.3f}/{mcs_range:.1f}"
            )
            
            # TensorBoard logging
            self.writer.add_scalar('Exploration/Factor', self.agent.exploration_factor, self.agent.step_count)
            self.writer.add_scalar('Exploration/Power_Std', power_std, self.agent.step_count)
            self.writer.add_scalar('Exploration/Beacon_Std', beacon_std, self.agent.step_count)
            self.writer.add_scalar('Exploration/MCS_Std', mcs_std, self.agent.step_count)

    def save_models(self, episode=None):
        """Save all models with proper naming and directory structure"""
        if not self.training_mode:
            return
        
        episode = episode or self.episode_count
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"{MODEL_SAVE_DIR}/vehicle_{self.node_id}_ep{episode}_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Save agent
            torch.save({
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic1_state_dict': self.agent.critic1.state_dict(),
                'critic2_state_dict': self.agent.critic2.state_dict(),
                'target_critic1_state_dict': self.agent.target_critic1.state_dict(),
                'target_critic2_state_dict': self.agent.target_critic2.state_dict(),
                'actor_optim_state_dict': self.agent.actor_optim.state_dict(),
                'critic1_optim_state_dict': self.agent.critic1_optim.state_dict(),
                'critic2_optim_state_dict': self.agent.critic2_optim.state_dict(),
                'episode': episode,
                'timestamp': timestamp,
            }, f"{model_dir}/agent.pth")
            
            # Save feature extractor
            torch.save({
                'state_dict': self.feature_extractor.state_dict(),
                'episode': episode,
                'timestamp': timestamp,
            }, f"{model_dir}/feature_extractor.pth")
            
            # Save current parameters
            with open(f"{model_dir}/params.json", 'w') as f:
                json.dump({
                    'power': self.current_power,
                    'beacon_rate': self.current_beacon_rate,
                    'mcs': self.current_mcs,
                    'episode': episode,
                    'timestamp': timestamp,
                }, f)
            
            log_message(f"Saved ULTRA-EXPLORATIVE models for vehicle {self.node_id} at episode {episode}")
            self.last_save_episode = episode
            return True
            
        except Exception as e:
            log_message(f"Failed to save models for vehicle {self.node_id}: {str(e)}", "ERROR")
            return False

    def load_models(self, model_dir):
        """Load models from specified directory"""
        try:
            # Load agent
            agent_checkpoint = torch.load(f"{model_dir}/agent.pth")
            self.agent.actor.load_state_dict(agent_checkpoint['actor_state_dict'])
            self.agent.critic1.load_state_dict(agent_checkpoint['critic1_state_dict'])
            self.agent.critic2.load_state_dict(agent_checkpoint['critic2_state_dict'])
            self.agent.target_critic1.load_state_dict(agent_checkpoint['target_critic1_state_dict'])
            self.agent.target_critic2.load_state_dict(agent_checkpoint['target_critic2_state_dict'])
            self.agent.actor_optim.load_state_dict(agent_checkpoint['actor_optim_state_dict'])
            self.agent.critic1_optim.load_state_dict(agent_checkpoint['critic1_optim_state_dict'])
            self.agent.critic2_optim.load_state_dict(agent_checkpoint['critic2_optim_state_dict'])
            
            # Load feature extractor
            fe_checkpoint = torch.load(f"{model_dir}/feature_extractor.pth")
            self.feature_extractor.load_state_dict(fe_checkpoint['state_dict'])
            
            # Load parameters
            with open(f"{model_dir}/params.json", 'r') as f:
                params = json.load(f)
                self.current_power = params['power']
                self.current_beacon_rate = params['beacon_rate']
                self.current_mcs = params['mcs']
                self.episode_count = params.get('episode', 0)
            
            log_message(f"Loaded ULTRA-EXPLORATIVE models for vehicle {self.node_id}")
            return True
            
        except Exception as e:
            log_message(f"Failed to load models for vehicle {self.node_id}: {str(e)}", "ERROR")
            return False

    def maybe_save_models(self):
        """Conditional model saving based on interval"""
        if (self.training_mode and 
            self.episode_count - self.last_save_episode >= MODEL_SAVE_INTERVAL):
            self.save_models()

# Legacy wrapper for compatibility
class VehicleNode(UltraExplorativeVehicleNode):
    """Backward compatibility wrapper"""
    pass

# ==============================
# Enhanced Decentralized RL Server
# ==============================

class UltraExplorativeDecentralizedRLServer:
    def __init__(self, host, port, training_mode=True):
        self.host = host
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(5)
        self.training_mode = training_mode
        self.vehicle_nodes = {}
        self.running = True
        
        self.global_episode_count = 0
        
        # Initialize ULTRA-EXPLORATIVE shared models
        self.shared_feature_extractor = ExploratorySharedFeatureExtractor(state_dim=3)
        self.shared_agent = UltraExplorativeSingleAgent(self.shared_feature_extractor)
        
        # Load models if available
        if training_mode:
            SharedModelManager.load_latest_models(
                self.shared_feature_extractor,
                self.shared_agent
            )
        
        log_message(f"🚀 ULTRA-EXPLORATIVE SINGLE AGENT Server listening on {self.host}:{self.port}")
        log_message(f"Starting in {'TRAINING' if training_mode else 'PRODUCTION'} mode")
        log_message(f"🎯 Exploration settings:")
        log_message(f"  - Initial exploration factor: {INITIAL_EXPLORATION_FACTOR}")
        log_message(f"  - Power action bounds: ±{POWER_ACTION_BOUND}")
        log_message(f"  - Beacon action bounds: ±{BEACON_ACTION_BOUND}")
        log_message(f"  - MCS action bounds: ±{MCS_ACTION_BOUND}")
        log_message(f"  - CALCULATED MAX EXPLORATION RANGES:")
        log_message(f"    * Power: ±{POWER_ACTION_BOUND * INITIAL_EXPLORATION_FACTOR:.1f} dBm (system needs ±29)")
        log_message(f"    * Beacon: ±{BEACON_ACTION_BOUND * INITIAL_EXPLORATION_FACTOR:.1f} Hz (system needs ±19)")
        log_message(f"    * MCS: ±{MCS_ACTION_BOUND * INITIAL_EXPLORATION_FACTOR:.1f} levels (system needs ±10)")

    def start(self):
        try:
            while self.running:
                conn, addr = self.server.accept()
                log_message(f"Connection established with {addr}")
                threading.Thread(
                    target=self.handle_client,
                    args=(conn, addr),
                    daemon=True
                ).start()
        except Exception as e:
            if self.running:
                log_message(f"Server error: {str(e)}", "ERROR")
            self.stop()
    
    def handle_client(self, conn, addr):
        """Enhanced client handler with complete error protection"""
        client_start_time = time.time()
        buffer = b""
        conn.settimeout(10.0)  # Reasonable timeout
        
        try:
            while self.running:
                try:
                    # Receive data with timeout
                    try:
                        data = conn.recv(65536)  # 64KB buffer
                        if not data:
                            log_message(f"Client {addr} closed connection", "INFO")
                            break
                        buffer += data
                    except socket.timeout:
                        continue
                    
                    # Process messages while buffer contains complete messages
                    while len(buffer) >= 4:  # Need at least 4 bytes for header
                        try:
                            # Extract message length (first 4 bytes, little-endian)
                            msg_length = int.from_bytes(buffer[:4], byteorder='little', signed=False)
                            
                            # Check if we have complete message (header + payload)
                            if len(buffer) < 4 + msg_length:
                                break  # Wait for more data
                            
                            # Extract message payload
                            message_bytes = buffer[4:4+msg_length]
                            
                            try:
                                # Decode JSON message
                                message = json.loads(message_bytes.decode('utf-8', errors='strict'))
                                log_message(f"🔄 Processing ULTRA-EXPLORATIVE message from {addr} with {len(message)} vehicles", "DEBUG")
                                
                                # Process the message
                                response = self._process_batch(message, addr)
                                
                                # Prepare response
                                response_json = json.dumps(response, ensure_ascii=False).encode('utf-8')
                                response_header = len(response_json).to_bytes(4, byteorder='little')
                                
                                # Send response
                                conn.sendall(response_header + response_json)
                                
                            except json.JSONDecodeError as e:
                                error_msg = f"JSON decode error: {str(e)}"
                                log_message(error_msg, "ERROR")
                                self._send_error_response(conn, error_msg)
                                
                            except Exception as e:
                                error_msg = f"Processing error: {str(e)}"
                                log_message(error_msg, "ERROR")
                                self._send_error_response(conn, error_msg)
                                
                            finally:
                                # Always advance buffer
                                buffer = buffer[4+msg_length:]
                                
                        except Exception as e:
                            log_message(f"Header processing error: {str(e)}", "ERROR")
                            buffer = b""  # Reset buffer on header errors
                            break
                            
                except (ConnectionResetError, BrokenPipeError) as e:
                    log_message(f"Connection error: {str(e)}", "WARNING")
                    break
                except Exception as e:
                    log_message(f"Unexpected error: {str(e)}", "ERROR")
                    break
                    
        finally:
            conn.close()
            duration = time.time() - client_start_time
            log_message(f"Connection closed. Duration: {duration:.2f}s", "INFO")
        
    def _process_batch(self, batch_data, addr):
        """Process a batch of vehicle data with ULTRA-VERBOSE logging"""
        batch_response = {"vehicles": {}, "timestamp": time.time()}
        
        # VERBOSE: Log the incoming batch metadata
        logger.info(f"🔄 PROCESSING ULTRA-EXPLORATIVE SINGLE AGENT BATCH from {addr}")
        logger.info(f"📊 Total vehicles in batch: {len(batch_data)}")
        logger.info(f"🚗 Vehicle IDs: {list(batch_data.keys())}")
        
        # Process each vehicle in the batch with detailed logging
        for vehicle_id, vehicle_data in batch_data.items():
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🚗 PROCESSING VEHICLE {vehicle_id}")
                logger.info(f"{'='*60}")
                
                # VERBOSE: Log incoming vehicle data from MATLAB
                logger.info(f"📥 RECEIVED FROM MATLAB:")
                logger.info(f"   CBR: {vehicle_data.get('CBR', 0):.6f}")
                logger.info(f"   SNR: {vehicle_data.get('SNR', 0):.3f} dB")
                logger.info(f"   Neighbors: {vehicle_data.get('neighbors', 0)}")
                logger.info(f"   Current Power: {vehicle_data.get('transmissionPower', 20):.1f} dBm")
                logger.info(f"   Current Beacon: {vehicle_data.get('beaconRate', 10):.2f} Hz")
                logger.info(f"   Current MCS: {vehicle_data.get('MCS', 0)}")
                logger.info(f"   Timestamp: {vehicle_data.get('timestamp', time.time())}")

                # Get current state
                state = [
                    float(vehicle_data.get('CBR', 0)),
                    float(vehicle_data.get('SINR', 0)),
                    float(vehicle_data.get('neighbors', 0))
                ]
                
                # Get current parameters
                current_params = {
                    'transmissionPower': float(vehicle_data.get('transmissionPower', 20)),
                    'beaconRate': float(vehicle_data.get('beaconRate', 10)),
                    'MCS': int(vehicle_data.get('MCS', 0))
                }
                
                logger.info(f" RL STATE PROCESSING:")
                logger.info(f"  State: CBR={state[0]:.3f}, SINR={state[1]:.1f}dB, Neighbors={int(state[2])}")
                
                # Initialize vehicle if new with ULTRA-EXPLORATIVE agent
                if vehicle_id not in self.vehicle_nodes:
                    self.vehicle_nodes[vehicle_id] = UltraExplorativeVehicleNode(
                        vehicle_id, self.shared_feature_extractor,
                        self.shared_agent, self.training_mode
                    )
                    logger.info(f"✨ CREATED NEW ULTRA-EXPLORATIVE SINGLE AGENT vehicle node {vehicle_id}")
                    logger.info(f"   Initial Exploration Factor: {self.vehicle_nodes[vehicle_id].agent.exploration_factor:.4f}")
                
                # Get ULTRA-EXPLORATIVE actions from RL
                vehicle = self.vehicle_nodes[vehicle_id]
                
                logger.info(f"🎯 RL AGENT PROCESSING:")
                logger.info(f"   Current Exploration Factor: {vehicle.agent.exploration_factor:.6f}")
                
                actions = vehicle.get_actions(state, current_params)
                new_params = vehicle.apply_actions(actions)
                
                # VERBOSE: Log the RL processing results
                logger.info(f"🎲 RL SINGLE AGENT ACTIONS GENERATED:")
                logger.info(f"   Power Delta: {actions['power_delta']:.6f}")
                logger.info(f"   Beacon Delta: {actions['beacon_delta']:.6f}")
                logger.info(f"   MCS Delta: {actions['mcs_delta']}")
                
                logger.info(f"⚙️  PARAMETER ADJUSTMENT CALCULATION:")
                logger.info(f"   Old Power: {current_params['transmissionPower']:.1f} dBm → New Power: {new_params['power']:.1f} dBm (Δ={new_params['power']-current_params['transmissionPower']:.1f})")
                logger.info(f"   Old Beacon: {current_params['beaconRate']:.2f} Hz → New Beacon: {new_params['beacon_rate']:.2f} Hz (Δ={new_params['beacon_rate']-current_params['beaconRate']:.2f})")
                logger.info(f"   Old MCS: {current_params['MCS']} → New MCS: {new_params['mcs']} (Δ={new_params['mcs']-current_params['MCS']})")
                
                # Prepare response
                response_data = {
                    'transmissionPower': new_params['power'],
                    'beaconRate': new_params['beacon_rate'],
                    'MCS': new_params['mcs'],
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
                
                batch_response["vehicles"][vehicle_id] = response_data
                
                # VERBOSE: Log the final response being sent back to MATLAB
                logger.info(f"📤 SENDING TO MATLAB:")
                logger.info(f"   Transmission Power: {new_params['power']:.1f} dBm")
                logger.info(f"   Beacon Rate: {new_params['beacon_rate']:.2f} Hz")
                logger.info(f"   MCS: {new_params['mcs']}")
                logger.info(f"   Status: SUCCESS")
                
                # Add enhanced training info
                if self.training_mode:
                    next_state = self._simulate_next_state(state)
                    reward = vehicle.agent.calculate_reward(
                        state[0], state[1], new_params['power'], 
                        new_params['mcs'], state[2])
                    
                    batch_response["vehicles"][vehicle_id]['training'] = {
                        'reward': float(reward),
                        'exploration_factor': actions.get('exploration_factor', 1.0),
                        'power_delta': actions['power_delta'],
                        'beacon_delta': actions['beacon_delta'],
                        'mcs_delta': actions['mcs_delta']
                    }
                    
                    # Store experience
                    actions_tensor = torch.FloatTensor([
                        actions['power_delta'],
                        actions['beacon_delta'],
                        actions['mcs_delta']
                    ]).unsqueeze(0)
                    
                    vehicle.store_experience(state, actions_tensor, reward, next_state, False)
                    
                    # VERBOSE: Log enhanced training information
                    logger.info(f"🎓 TRAINING DATA:")
                    logger.info(f"   Reward: {reward:.6f}")
                    logger.info(f"   Next State: [CBR={next_state[0]:.6f}, SNR={next_state[1]:.3f}, Neighbors={next_state[2]}]")
                    logger.info(f"   Experience Stored: Buffer size={len(vehicle.agent.replay_buffer)}")
                    
                logger.info(f"✅ VEHICLE {vehicle_id} PROCESSING COMPLETE")
                    
            except Exception as e:
                error_msg = f"Error processing vehicle {vehicle_id}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
                batch_response["vehicles"][vehicle_id] = {
                    'status': 'error',
                    'error': error_msg,
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
        
        # VERBOSE: Log batch completion summary
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 BATCH PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"✅ Successfully processed: {sum(1 for v in batch_response['vehicles'].values() if v.get('status') != 'error')} vehicles")
        logger.info(f"❌ Failed to process: {sum(1 for v in batch_response['vehicles'].values() if v.get('status') == 'error')} vehicles")
        logger.info(f"🕐 Total processing time: {time.time() - batch_response['timestamp']:.4f} seconds")
        logger.info(f"📤 Sending response back to MATLAB...")
        
        return batch_response

    def _send_error_response(self, conn, error_msg):
        """Send error response to client"""
        try:
            error_response = json.dumps({
                'status': 'error',
                'error': error_msg,
                'timestamp': time.time()
            }).encode('utf-8')
            
            header = len(error_response).to_bytes(4, byteorder='little')
            conn.sendall(header + error_response)
        except:
            pass

    def _simulate_next_state(self, current_state):
        """Simple environment transition model for training"""
        try:
            cbr, snr, neighbors = current_state
            
            # Simulate CBR change (with noise and boundary checks)
            new_cbr = cbr + random.uniform(-0.05, 0.05)
            new_cbr = max(0.0, min(1.0, new_cbr))
            
            # Simulate SNR change (with noise and minimum threshold)
            new_snr = snr + random.uniform(-2.0, 2.0)
            new_snr = max(0.0, new_snr)
            
            # Simulate neighbor count change (discrete changes)
            neighbor_change = random.choice([-1, 0, 1])
            new_neighbors = max(0, neighbors + neighbor_change)
            
            return [new_cbr, new_snr, new_neighbors]
        except Exception as e:
            logger.error(f"Error in state simulation: {e}")
            return current_state
    
    def stop(self):
        """Stop the server and save shared models"""
        self.running = False
        self.server.close()
        
        if self.training_mode:
            SharedModelManager.save_models(
                self.shared_feature_extractor,
                self.shared_agent,
                episode=self.global_episode_count
            )
        log_message("🚀 ULTRA-EXPLORATIVE SINGLE AGENT Server stopped")

# Legacy wrapper for compatibility
class DecentralizedRLServer(UltraExplorativeDecentralizedRLServer):
    """Backward compatibility wrapper"""
    pass

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # Set training_mode=False for production mode
    rl_server = UltraExplorativeDecentralizedRLServer(HOST, PORT, training_mode=True)
    try:
        log_message("🚀 Starting ULTRA-EXPLORATIVE SINGLE AGENT RL server...")
        log_message(f"Enhanced exploration settings:")
        log_message(f"  - Initial exploration factor: {INITIAL_EXPLORATION_FACTOR}")
        log_message(f"  - Power action bounds: ±{POWER_ACTION_BOUND}")
        log_message(f"  - Beacon action bounds: ±{BEACON_ACTION_BOUND}")
        log_message(f"  - MCS action bounds: ±{MCS_ACTION_BOUND}")
        log_message(f"  - Multiple exploration mechanisms enabled")
        log_message(f"  - ULTRA-VERBOSE logging enabled")
        rl_server.start()
    except KeyboardInterrupt:
        log_message("🚀 ULTRA-EXPLORATIVE Server interrupted by user. Shutting down...")
        rl_server.stop()
    except Exception as e:
        log_message(f"Fatal error: {str(e)}", "CRITICAL")
        rl_server.stop()
