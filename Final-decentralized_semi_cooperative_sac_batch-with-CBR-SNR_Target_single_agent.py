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

# Environment parameters - aligned with dual agent
N_STEP = 3
CBR_TARGET = 0.65
SINR_TARGET = 12.0  # CHANGED from 18 to match dual agent realistic target
POWER_MIN, POWER_MAX = 1, 30  # CHANGED to match dual agent system_config
BEACON_RATE_MIN, BEACON_RATE_MAX = 1, 20  # Match dual agent
MCS_MIN, MCS_MAX = 0, 9  # CHANGED to match dual agent system_config
MAX_NEIGHBORS = 15  # CHANGED to match dual agent training_config.max_neighbors

# ULTRA-AGGRESSIVE EXPLORATION PARAMETERS (aligned with dual agent)
POWER_ACTION_BOUND = 15.0     # CHANGED to match dual agent realistic bounds
BEACON_ACTION_BOUND = 10.0    # CHANGED to match dual agent realistic bounds
MCS_ACTION_BOUND = 7.5        # CHANGED to match dual agent realistic bounds

# Reward weights for single agent (aligned with dual agent)
W1, W2 = 5.0, 1.0  # CHANGED W1 from 3.0 to 5.0 to match dual agent
W3, W4 = 3.0, 0.8  # CHANGED W3 from 2.5 to 3.0 to match dual agent  
W5 = 0.5  # CHANGED from 1.5 to 0.5 to match dual agent (reduced neighbor penalty)
BETA = 15  # Keep same

# SAC parameters (aligned with dual agent RealisticTrainingConfig)
BUFFER_SIZE = 120000  # CHANGED from 100000 to match dual agent
BATCH_SIZE = 256      # CHANGED from 128 to match dual agent
GAMMA = 0.99          # Keep same
TAU = 0.003           # CHANGED from 0.005 to match dual agent
ALPHA = 0.2           # Keep same
LR = 2e-4             # CHANGED from 3e-3 to match dual agent
HIDDEN_UNITS = 384    # CHANGED from 256 to match dual agent

# Enhanced exploration (aligned with dual agent)
INITIAL_EXPLORATION_FACTOR = 2.0   # CHANGED from 5.0 to match dual agent realistic values
EXPLORATION_DECAY = 0.9998         # Match dual agent
MIN_EXPLORATION = 0.3              # CHANGED from 0.8 to match dual agent
EXPLORATION_NOISE_SCALE = 0.6      # CHANGED from 1.2 to match dual agent
ACTION_NOISE_SCALE = 0.4           # CHANGED from 0.8 to match dual agent
RANDOM_ACTION_PROB = 0.05          # REDUCED from 0.15 to match dual agent approach

# Enhanced entropy (aligned with dual agent)
INITIAL_LOG_STD = 0.5              # CHANGED from 1.0 to match dual agent
LOG_STD_MIN = -10                  # CHANGED from -8 to match dual agent
LOG_STD_MAX = 2                    # CHANGED from 4 to match dual agent

# Network setup
HOST = '127.0.0.1'
PORT = 5000
LOG_DIR = 'ultra_aggressive_single_agent_logs'
MODEL_SAVE_DIR = "saved_models"
MODEL_SAVE_INTERVAL = 100  # Save every 100 episodes
SHARED_MODEL_PREFIX = "shared"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MAX_LOG_LENGTH = 200  # Max characters to log for data payloads


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
# Model Directory Setup
# ==============================
# MODIFIED: Don't clear existing models, just ensure directory exists
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Log existing models using logger directly (since log_message isn't available yet)
if os.path.exists(MODEL_SAVE_DIR):
    existing_models = os.listdir(MODEL_SAVE_DIR)
    if existing_models:
        logger.info(f"Found existing models: {existing_models}")
    else:
        logger.info("No existing models found, starting fresh")

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
    
    
class PerformanceTracker:
    """Track only essential algorithm performance metrics for Excel export"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Performance summary data (what goes to Excel)
        self.performance_data = {
            'episode': [],
            'timestamp': [],
            'avg_reward': [],
            'actor_loss': [],
            'critic_loss': [],
            'exploration_factor': [],
            'success_rate': [],
            'active_vehicles': [],
            'buffer_size': []
        }
        
        # Temporary tracking for calculations
        self.episode_rewards = []
        self.recent_losses = {'actor': [], 'critic': []}
        self.batch_success_count = 0
        self.batch_total_count = 0
        
    def update_training_metrics(self, actor_loss, critic1_loss, critic2_loss, exploration_factor, buffer_size):
        """Update training metrics (called during agent updates)"""
        avg_critic_loss = (critic1_loss + critic2_loss) / 2
        
        self.recent_losses['actor'].append(actor_loss)
        self.recent_losses['critic'].append(avg_critic_loss)
        
        # Keep only last 50 losses for moving average
        if len(self.recent_losses['actor']) > 50:
            self.recent_losses['actor'].pop(0)
            self.recent_losses['critic'].pop(0)
    
    def update_episode_metrics(self, episode, reward, active_vehicles, batch_success, batch_total):
        """Update episode-level metrics (called at episode end)"""
        self.episode_rewards.append(reward)
        self.batch_success_count += batch_success
        self.batch_total_count += batch_total
        
        # Calculate performance summary every 10 episodes
        if episode % 10 == 0:
            self._save_performance_summary(episode, active_vehicles)
    
    def _save_performance_summary(self, episode, active_vehicles):
        """Save performance summary to Excel-ready format"""
        # Calculate averages
        avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
        avg_actor_loss = np.mean(self.recent_losses['actor'][-10:]) if self.recent_losses['actor'] else 0
        avg_critic_loss = np.mean(self.recent_losses['critic'][-10:]) if self.recent_losses['critic'] else 0
        success_rate = (self.batch_success_count / max(self.batch_total_count, 1)) * 100
        
        # Get current exploration factor (from any active vehicle)
        exploration_factor = 1.0  # Default, will be updated by caller
        buffer_size = 0  # Default, will be updated by caller
        
        # Add to performance data
        self.performance_data['episode'].append(episode)
        self.performance_data['timestamp'].append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.performance_data['avg_reward'].append(round(avg_reward, 4))
        self.performance_data['actor_loss'].append(round(avg_actor_loss, 6))
        self.performance_data['critic_loss'].append(round(avg_critic_loss, 6))
        self.performance_data['exploration_factor'].append(round(exploration_factor, 4))
        self.performance_data['success_rate'].append(round(success_rate, 2))
        self.performance_data['active_vehicles'].append(active_vehicles)
        self.performance_data['buffer_size'].append(buffer_size)
        
        # Reset batch counters
        self.batch_success_count = 0
        self.batch_total_count = 0
        
        # Save to CSV (Excel-compatible)
        self._export_to_csv()
    
    def _export_to_csv(self):
        """Export performance summary to CSV and Excel with Summary Sheet"""
        try:
            df = pd.DataFrame(self.performance_data)
            csv_path = f"{self.log_dir}/rl_performance_summary.csv"
            df.to_csv(csv_path, index=False)
            
            # Create Excel file with multiple sheets
            excel_path = f"{self.log_dir}/rl_performance_summary.xlsx"
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Raw data sheet
                df.to_excel(writer, sheet_name='Raw_Data', index=False)
                
                # Summary sheet with algorithm performance analysis
                summary_df = self._create_summary_sheet(df)
                summary_df.to_excel(writer, sheet_name='Algorithm_Summary', index=False)
                
                # Trends sheet with recent performance
                trends_df = self._create_trends_sheet(df)
                trends_df.to_excel(writer, sheet_name='Recent_Trends', index=False)
            
        except Exception as e:
            log_message(f"Failed to export performance data: {e}", "ERROR")
    
    def _create_summary_sheet(self, df):
        """Create algorithm performance summary sheet"""
        if df.empty:
            return pd.DataFrame()
        
        # Calculate key performance indicators
        summary_data = {
            'Metric': [
                'Training Status',
                'Total Episodes',
                'Training Duration (Episodes)',
                '',
                '=== REWARD PERFORMANCE ===',
                'Latest Average Reward',
                'Best Average Reward',
                'Worst Average Reward',
                'Reward Improvement (%)',
                'Reward Trend (Last 5 Episodes)',
                '',
                '=== LEARNING PERFORMANCE ===',
                'Latest Actor Loss',
                'Latest Critic Loss',
                'Best Actor Loss',
                'Best Critic Loss',
                'Loss Trend (Last 5 Episodes)',
                '',
                '=== EXPLORATION STATUS ===',
                'Current Exploration Factor',
                'Exploration Decay Progress (%)',
                'Exploration Remaining',
                '',
                '=== SYSTEM PERFORMANCE ===',
                'Current Success Rate (%)',
                'Average Success Rate (%)',
                'Active Vehicles',
                'Max Buffer Size Reached',
                'Training Efficiency',
                '',
                '=== CONVERGENCE ANALYSIS ===',
                'Reward Stability (StdDev)',
                'Loss Stability (StdDev)',
                'Learning Progress Score',
                'Convergence Status'
            ],
            'Value': []
        }
        
        # Calculate values
        latest_reward = df['avg_reward'].iloc[-1] if not df.empty else 0
        best_reward = df['avg_reward'].max() if not df.empty else 0
        worst_reward = df['avg_reward'].min() if not df.empty else 0
        
        latest_actor_loss = df['actor_loss'].iloc[-1] if not df.empty else 0
        latest_critic_loss = df['critic_loss'].iloc[-1] if not df.empty else 0
        best_actor_loss = df['actor_loss'].min() if not df.empty else 0
        best_critic_loss = df['critic_loss'].min() if not df.empty else 0
        
        latest_exploration = df['exploration_factor'].iloc[-1] if not df.empty else 1.0
        exploration_progress = ((1.0 - latest_exploration) / (1.0 - 0.3)) * 100  # Using 0.3 as MIN_EXPLORATION
        
        success_rate = df['success_rate'].iloc[-1] if not df.empty else 0
        avg_success_rate = df['success_rate'].mean() if not df.empty else 0
        
        active_vehicles = df['active_vehicles'].iloc[-1] if not df.empty else 0
        max_buffer = df['buffer_size'].max() if not df.empty else 0
        
        # Calculate trends and improvements
        if len(df) >= 2:
            reward_improvement = ((latest_reward - df['avg_reward'].iloc[0]) / abs(df['avg_reward'].iloc[0]) * 100) if df['avg_reward'].iloc[0] != 0 else 0
            
            # Recent trends (last 5 episodes)
            recent_rewards = df['avg_reward'].tail(5)
            recent_losses = df['actor_loss'].tail(5)
            
            reward_trend = "↗ Improving" if recent_rewards.is_monotonic_increasing else "↘ Declining" if recent_rewards.is_monotonic_decreasing else "→ Stable"
            loss_trend = "↗ Improving" if recent_losses.is_monotonic_decreasing else "↘ Worsening" if recent_losses.is_monotonic_increasing else "→ Stable"
            
            # Stability analysis
            reward_stability = df['avg_reward'].std()
            loss_stability = df['actor_loss'].std()
            
            # Learning progress score (0-100)
            reward_score = min(100, max(0, (latest_reward / max(best_reward, 0.1)) * 50))
            loss_score = min(100, max(0, (1 - (latest_actor_loss / max(df['actor_loss'].max(), 0.1))) * 30))
            exploration_score = min(100, max(0, exploration_progress * 0.2))
            learning_score = reward_score + loss_score + exploration_score
            
            # Convergence status
            if learning_score > 80:
                convergence_status = " Well Converged"
            elif learning_score > 60:
                convergence_status = " Converging"
            elif learning_score > 40:
                convergence_status = " Early Training"
            else:
                convergence_status = " Training Started"
                
        else:
            reward_improvement = 0
            reward_trend = "N/A"
            loss_trend = "N/A"
            reward_stability = 0
            loss_stability = 0
            learning_score = 0
            convergence_status = " Insufficient Data"
        
        # Training efficiency
        training_efficiency = f"{latest_reward:.3f} reward/episode" if latest_reward > 0 else "Starting"
        
        # Fill values
        values = [
            " Active" if len(df) > 0 else " Inactive",
            len(df) * 10,  # Since we save every 10 episodes
            f"{df['episode'].iloc[-1] - df['episode'].iloc[0] if len(df) > 1 else df['episode'].iloc[-1] if len(df) > 0 else 0} episodes",
            '',
            '',
            f"{latest_reward:.4f}",
            f"{best_reward:.4f}",
            f"{worst_reward:.4f}",
            f"{reward_improvement:.2f}%",
            reward_trend,
            '',
            '',
            f"{latest_actor_loss:.6f}",
            f"{latest_critic_loss:.6f}",
            f"{best_actor_loss:.6f}",
            f"{best_critic_loss:.6f}",
            loss_trend,
            '',
            '',
            f"{latest_exploration:.4f}",
            f"{exploration_progress:.1f}%",
            f"{1.0 - latest_exploration:.4f}",
            '',
            '',
            f"{success_rate:.1f}%",
            f"{avg_success_rate:.1f}%",
            f"{active_vehicles} vehicles",
            f"{max_buffer:,} experiences",
            training_efficiency,
            '',
            '',
            f"{reward_stability:.4f}",
            f"{loss_stability:.6f}",
            f"{learning_score:.1f}/100",
            convergence_status
        ]
        
        summary_data['Value'] = values
        return pd.DataFrame(summary_data)
    
    def _create_trends_sheet(self, df):
        """Create recent trends analysis sheet"""
        if df.empty or len(df) < 5:
            return pd.DataFrame({'Message': ['Insufficient data for trend analysis. Need at least 5 data points.']})
        
        # Get last 10 episodes for trend analysis
        recent_df = df.tail(10).copy()
        
        # Calculate episode-to-episode changes
        recent_df['reward_change'] = recent_df['avg_reward'].pct_change() * 100
        recent_df['actor_loss_change'] = recent_df['actor_loss'].pct_change() * 100
        recent_df['exploration_change'] = recent_df['exploration_factor'].diff()
        
        # Create trends summary
        trends_data = {
            'Episode': recent_df['episode'].tolist(),
            'Avg_Reward': recent_df['avg_reward'].round(4).tolist(),
            'Reward_Change_%': recent_df['reward_change'].round(2).tolist(),
            'Actor_Loss': recent_df['actor_loss'].round(6).tolist(),
            'Loss_Change_%': recent_df['actor_loss_change'].round(2).tolist(),
            'Exploration': recent_df['exploration_factor'].round(4).tolist(),
            'Exploration_Change': recent_df['exploration_change'].round(4).tolist(),
            'Success_Rate_%': recent_df['success_rate'].round(1).tolist(),
            'Buffer_Size': recent_df['buffer_size'].tolist()
        }
        
        trends_df = pd.DataFrame(trends_data)
        
        # Add trend indicators
        trends_df['Reward_Trend'] = trends_df['Reward_Change_%'].apply(
            lambda x: '↗' if x > 1 else '↘' if x < -1 else '→' if pd.notna(x) else ''
        )
        trends_df['Loss_Trend'] = trends_df['Loss_Change_%'].apply(
            lambda x: '↗' if x < -1 else '↘' if x > 1 else '→' if pd.notna(x) else ''
        )
        
        return trends_df

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
        """Update priorities with proper error handling"""
        try:
            if isinstance(priorities, torch.Tensor):
                priorities = priorities.detach().cpu().numpy()
            
            # Ensure priorities is a 1D numpy array
            if isinstance(priorities, np.ndarray):
                priorities = priorities.flatten()
            
            # Ensure data_idxs is iterable
            if not hasattr(data_idxs, '__iter__'):
                data_idxs = [data_idxs]
            
            for data_idx, priority in zip(data_idxs, priorities):
                # Ensure priority is a scalar
                if isinstance(priority, (np.ndarray, list)):
                    priority = float(priority.item() if hasattr(priority, 'item') else priority[0])
                else:
                    priority = float(priority)
                
                # Ensure priority is finite and positive
                if not np.isfinite(priority) or priority <= 0:
                    priority = self.eps
                
                priority = (priority + self.eps) ** self.alpha
                self.tree.update(data_idx, priority)
                
                # Safely update max_priority
                if np.isfinite(priority):
                    self.max_priority = max(float(self.max_priority), float(priority))
                    
        except Exception as e:
            logger.error(f"Error updating priorities: {e}")
            # Fallback to eps for all priorities
            for data_idx in data_idxs:
                try:
                    priority = self.eps ** self.alpha
                    self.tree.update(data_idx, priority)
                except:
                    pass

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
        """Sample actions with improved exploration aligned with dual agent"""
        try:
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            
            # Enhanced log std clamping (align with dual agent values)
            power_log_std = torch.clamp(self.power_logstd, LOG_STD_MIN, LOG_STD_MAX)
            beacon_log_std = torch.clamp(self.beacon_logstd, LOG_STD_MIN, LOG_STD_MAX)
            mcs_log_std = torch.clamp(self.mcs_logstd, LOG_STD_MIN, LOG_STD_MAX)
            
            # Power action with realistic bounds (align with dual agent)
            power_mean = torch.tanh(self.power_mean(x)) * POWER_ACTION_BOUND
            power_std = torch.exp(power_log_std)
            power_noise = torch.randn_like(power_mean)
            power_action = power_mean + power_noise * power_std
            
            # Beacon action with realistic bounds (align with dual agent)
            beacon_mean = torch.tanh(self.beacon_mean(x)) * BEACON_ACTION_BOUND
            beacon_std = torch.exp(beacon_log_std)
            beacon_noise = torch.randn_like(beacon_mean)
            beacon_action = beacon_mean + beacon_noise * beacon_std
            
            # MCS action with realistic bounds (align with dual agent)
            mcs_mean = torch.tanh(self.mcs_mean(x)) * MCS_ACTION_BOUND
            mcs_std = torch.exp(mcs_log_std)
            mcs_noise = torch.randn_like(mcs_mean)
            mcs_action = mcs_mean + mcs_noise * mcs_std
            
            # Calculate log probabilities with safety checks
            power_dist = Normal(power_mean, power_std + 1e-8)
            beacon_dist = Normal(beacon_mean, beacon_std + 1e-8)
            mcs_dist = Normal(mcs_mean, mcs_std + 1e-8)
            
            power_log_prob = power_dist.log_prob(power_action)
            beacon_log_prob = beacon_dist.log_prob(beacon_action)
            mcs_log_prob = mcs_dist.log_prob(mcs_action)
            
            # Ensure log_probs are finite
            power_log_prob = torch.where(torch.isfinite(power_log_prob), power_log_prob, torch.tensor(-10.0))
            beacon_log_prob = torch.where(torch.isfinite(beacon_log_prob), beacon_log_prob, torch.tensor(-10.0))
            mcs_log_prob = torch.where(torch.isfinite(mcs_log_prob), mcs_log_prob, torch.tensor(-10.0))
            
            log_prob = power_log_prob + beacon_log_prob + mcs_log_prob
            
            # Stack actions
            actions = torch.stack([power_action.squeeze(-1), beacon_action.squeeze(-1), mcs_action.squeeze(-1)], dim=-1)
            
            # Ensure actions are finite
            actions = torch.where(torch.isfinite(actions), actions, torch.zeros_like(actions))
            
            return actions, log_prob
            
        except Exception as e:
            logger.error(f"Error in actor sample: {e}")
            # Return safe fallback
            batch_size = x.shape[0] if x.dim() > 1 else 1
            safe_actions = torch.zeros(batch_size, 3)
            safe_log_prob = torch.tensor(-10.0)
            return safe_actions, safe_log_prob

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
                
                logger.info(f"  RANDOM EXPLORATION ACTIVATED: {actions.flatten().tolist()}")
            
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
        """Enhanced reward function aligned with dual agent improvements"""
        try:
            # Adaptive SINR target (align with dual agent logic)
            adaptive_sinr_target = max(8.0, SINR_TARGET - (neighbor_count / 5))  # More realistic adaptation
            
            # CBR term (align with dual agent approach)
            cbr_error = abs(cbr - CBR_TARGET)
            if cbr_error < 0.05:  # Very close to target
                cbr_term = W1 * 2.0  # Bonus for being close
            else:
                cbr_term = W1 * max(0, 1 - cbr_error * 2.0)  # Linear penalty like dual agent
            
            # Power term (align with dual agent approach)
            power_norm = (current_power - POWER_MIN) / (POWER_MAX - POWER_MIN)
            power_term = W2 * (power_norm ** 2)  # Quadratic penalty
            
            # SINR term (use dual agent tanh approach for consistency)
            sinr_diff = sinr - adaptive_sinr_target
            sinr_term = W3 * math.tanh(sinr_diff / 5.0)
            
            # MCS term (align with dual agent density-aware approach)
            density_factor = min(1.0, neighbor_count / MAX_NEIGHBORS)
            mcs_norm = (current_mcs - MCS_MIN) / (MCS_MAX - MCS_MIN)
            
            if density_factor > 0.7:  # High density - prefer robust (low) MCS
                mcs_term = W4 * 0.5 * (1 - mcs_norm)
            else:  # Low density - prefer efficient (high) MCS
                mcs_term = W4 * 0.5 * mcs_norm
            
            # Neighbor impact (align with dual agent reduced penalty)
            neighbor_impact = W5 * (neighbor_count / MAX_NEIGHBORS)
            
            # Base reward calculation
            base_reward = cbr_term - power_term + sinr_term + mcs_term - neighbor_impact
            
            # Exploration bonus during early training (align with dual agent)
            if self.exploration_factor > 1.0:
                exploration_bonus = 0.2 * self.exploration_factor  # Match dual agent bonus
                base_reward += exploration_bonus
            
            # Amplify differences for better learning (align with dual agent)
            base_reward = base_reward * 1.5
            
            # Add performance bonuses/penalties (align with dual agent)
            if cbr_error < 0.05:  # Excellent CBR
                base_reward += 3.0
            elif cbr_error > 0.3:  # Poor CBR
                base_reward -= 2.0
            
            sinr_error = abs(sinr - adaptive_sinr_target)
            if sinr_error < 2.0:  # Excellent SINR
                base_reward += 2.0
            elif sinr_error > 8.0:  # Poor SINR
                base_reward -= 2.0
            
            return float(base_reward)
            
        except Exception as e:
            logger.error(f"Reward calculation failed: {e}")
            return 0.0

# ==============================
# Enhanced Vehicle Node Implementation
# ==============================
class UltraExplorativeVehicleNode:
    def __init__(self, node_id, feature_extractor, agent, training_mode=True, performance_tracker=None):
        self.node_id = node_id
        self.training_mode = training_mode
        self.feature_extractor = feature_extractor
        self.agent = agent
        self.performance_tracker = performance_tracker
        
        # Parameters
        self.current_power = None
        self.current_beacon_rate = None
        self.current_mcs = None
        
        # Minimal tracking for performance calculation
        self.episode_count = 0
        self.train_counter = 0
        self.train_interval = 3
        self.episode_reward = 0.0
        
        # TensorBoard (optional, lightweight)
        if training_mode:
            self.writer = SummaryWriter(f"{LOG_DIR}/vehicle_{node_id}")
        else:
            self.writer = None

    def get_actions(self, state, current_params):
        """Get actions - no detailed logging"""
        try:
            self.current_power = float(current_params.get('transmissionPower', 20))
            self.current_beacon_rate = float(current_params.get('beaconRate', 10))
            self.current_mcs = int(current_params.get('MCS', 0))
    
            state_tensor = safe_tensor_conversion(np.array(state, dtype=np.float32))
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            with torch.no_grad():
                features = self.feature_extractor(state_tensor)
                actions, log_prob = self.agent.select_action(features)
                
                if torch.isnan(actions).any():
                    actions = torch.zeros_like(actions)
    
                power_delta = max(-15.0, min(15.0, actions[0, 0].item()))
                beacon_delta = max(-10.0, min(10.0, actions[0, 1].item()))
                mcs_delta = max(-7.5, min(7.5, actions[0, 2].item()))
                
                # NaN protection
                if math.isnan(power_delta): power_delta = 0.0
                if math.isnan(beacon_delta): beacon_delta = 0.0
                if math.isnan(mcs_delta): mcs_delta = 0.0
                
                new_power = max(POWER_MIN, min(POWER_MAX, self.current_power + power_delta))
                new_beacon = max(BEACON_RATE_MIN, min(BEACON_RATE_MAX, 
                                    self.current_beacon_rate + beacon_delta))
                mcs_delta_rounded = round(mcs_delta) if math.isfinite(mcs_delta) else 0
                new_mcs = max(MCS_MIN, min(MCS_MAX, self.current_mcs + mcs_delta_rounded))
    
            return {
                'power_delta': float(power_delta),
                'beacon_delta': float(beacon_delta),
                'mcs_delta': int(mcs_delta_rounded),
                'log_prob': log_prob,
                'new_power': float(new_power),
                'new_beacon_rate': float(new_beacon),
                'new_mcs': int(new_mcs),
                'exploration_factor': self.agent.exploration_factor
            }
    
        except Exception as e:
            log_message(f"Action selection failed for {self.node_id}: {str(e)}", "ERROR")
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
            
            logger.info(f"  APPLYING ACTIONS FOR {self.node_id}:")
            logger.info(f"   Final Power: {new_power:.1f} dBm")
            logger.info(f"   Final Beacon: {new_beacon_rate:.2f} Hz")
            logger.info(f"   Final MCS: {new_mcs}")
            
            return {
                'power': float(new_power),
                'beacon_rate': float(new_beacon_rate),
                'mcs': int(new_mcs)
            }
            
        except Exception as e:
            logger.error(f"  Action application failed: {str(e)}")
            raise RuntimeError(f"Action application failed: {str(e)}")

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience with minimal tracking"""
        try:
            # Add reward to episode total
            self.episode_reward += reward
            
            # Convert to tensors
            state_tensor = safe_tensor_conversion(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            next_state_tensor = safe_tensor_conversion(next_state)
            if next_state_tensor.dim() == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)
                
            if not isinstance(action, torch.Tensor):
                action = safe_tensor_conversion(action)
            if action.dim() == 1:
                action = action.unsqueeze(0)
                
            reward_tensor = torch.FloatTensor([reward])
            done_tensor = torch.FloatTensor([float(done)])
            
            experience = (state_tensor, action, reward_tensor, next_state_tensor, done_tensor)
            self.agent.replay_buffer.add(experience)
            
            # Training
            if (len(self.agent.replay_buffer) >= BATCH_SIZE and 
                self.train_counter % self.train_interval == 0):
                self.update_agent()
                
            # Episode end tracking
            if done and self.performance_tracker:
                self.performance_tracker.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0.0
                
        except Exception as e:
            log_message(f"Error storing experience: {str(e)}", "ERROR")


    def update_agent(self):
        """Agent update with minimal performance tracking"""
        if len(self.agent.replay_buffer) < BATCH_SIZE:
            return
            
        try:
            batch, weights, indices = self.agent.replay_buffer.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = batch
            
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            with torch.no_grad():
                features = self.feature_extractor(states)
                next_features = self.feature_extractor(next_states)
                next_actions, next_log_probs = self.agent.actor.sample(next_features)
                
                target_q1 = self.agent.target_critic1(next_features, next_actions)
                target_q2 = self.agent.target_critic2(next_features, next_actions)
                target_q = torch.min(target_q1, target_q2) - self.agent.log_alpha.exp() * next_log_probs.unsqueeze(-1)
                target_q = rewards.unsqueeze(-1) + (1 - dones.unsqueeze(-1)) * GAMMA * target_q
    
            # Update critics
            current_q1 = self.agent.critic1(features, actions)
            current_q2 = self.agent.critic2(features, actions)
            
            td_errors1 = current_q1 - target_q
            td_errors2 = current_q2 - target_q
            critic1_loss = (td_errors1.pow(2) * weights.unsqueeze(-1)).mean()
            critic2_loss = (td_errors2.pow(2) * weights.unsqueeze(-1)).mean()
            
            # Update priorities
            td_errors = torch.min(td_errors1.abs(), td_errors2.abs()).squeeze(-1).detach()
            if td_errors.dim() > 1:
                td_errors = td_errors.flatten()
            priorities_np = np.clip(td_errors.cpu().numpy(), 1e-6, 1e6)
            self.agent.replay_buffer.update_priorities(indices, priorities_np)
            
            # Critic updates
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
            
            # Update performance tracker (for Excel export)
            if self.performance_tracker:
                self.performance_tracker.update_training_metrics(
                    actor_loss.item(),
                    critic1_loss.item(),
                    critic2_loss.item(),
                    self.agent.exploration_factor,
                    len(self.agent.replay_buffer)
                )
                
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
                f"  EXPLORATION METRICS - Vehicle {self.node_id}: "
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
        
        # Performance tracker for Excel export
        self.performance_tracker = PerformanceTracker(LOG_DIR) if training_mode else None
        
        # Initialize shared models
        self.shared_feature_extractor = ExploratorySharedFeatureExtractor(state_dim=3)
        self.shared_agent = UltraExplorativeSingleAgent(self.shared_feature_extractor)
        
        # Load models if available
        if not training_mode:
            success = SharedModelManager.load_latest_models(
                self.shared_feature_extractor, self.shared_agent
            )
            log_message(f"Production mode: {'Loaded models' if success else 'Random init'}")
        
        log_message(f"Server listening on {self.host}:{self.port}")

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
                                log_message(f"  Processing ULTRA-EXPLORATIVE message from {addr} with {len(message)} vehicles", "DEBUG")
                                
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
        """Process batch with Excel performance tracking"""
        batch_response = {"vehicles": {}, "timestamp": time.time()}
        
        self.global_episode_count += 1
        successful_vehicles = 0
        failed_vehicles = 0
        
        for vehicle_id, vehicle_data in batch_data.items():
            try:
                # Process vehicle (minimal logging)
                state = [
                    float(vehicle_data.get('CBR', 0)),
                    float(vehicle_data.get('SINR', 0)),
                    float(vehicle_data.get('neighbors', 0))
                ]
                
                current_params = {
                    'transmissionPower': float(vehicle_data.get('transmissionPower', 20)),
                    'beaconRate': float(vehicle_data.get('beaconRate', 10)),
                    'MCS': int(vehicle_data.get('MCS', 0))
                }
                
                # Initialize vehicle if new
                if vehicle_id not in self.vehicle_nodes:
                    self.vehicle_nodes[vehicle_id] = UltraExplorativeVehicleNode(
                        vehicle_id, self.shared_feature_extractor,
                        self.shared_agent, self.training_mode,
                        self.performance_tracker
                    )
    
                vehicle = self.vehicle_nodes[vehicle_id]
                vehicle.episode_count = self.global_episode_count
                
                actions = vehicle.get_actions(state, current_params)
                new_params = vehicle.apply_actions(actions)
                
                # Response
                batch_response["vehicles"][vehicle_id] = {
                    'transmissionPower': new_params['power'],
                    'beaconRate': new_params['beacon_rate'],
                    'MCS': new_params['mcs'],
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
                
                # Training with performance tracking
                if self.training_mode:
                    next_state = self._simulate_next_state(state)
                    reward = vehicle.agent.calculate_reward(
                        state[0], state[1], new_params['power'], 
                        new_params['mcs'], state[2])
                    
                    actions_tensor = torch.FloatTensor([
                        actions['power_delta'],
                        actions['beacon_delta'],
                        actions['mcs_delta']
                    ]).unsqueeze(0)
                    
                    vehicle.store_experience(state, actions_tensor, reward, next_state, False)
                
                successful_vehicles += 1
                    
            except Exception as e:
                failed_vehicles += 1
                batch_response["vehicles"][vehicle_id] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': vehicle_data.get('timestamp', time.time())
                }
        
        # Update performance tracker for Excel export
        if self.training_mode and self.performance_tracker:
            self.performance_tracker.update_episode_metrics(
                self.global_episode_count,
                0,  # Will be calculated from individual vehicle rewards
                len(self.vehicle_nodes),
                successful_vehicles,
                successful_vehicles + failed_vehicles
            )
        
        # Save shared models periodically
        if self.training_mode and self.global_episode_count % MODEL_SAVE_INTERVAL == 0:
            SharedModelManager.save_models(
                self.shared_feature_extractor,
                self.shared_agent,
                episode=self.global_episode_count
            )
        
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
        """Simple environment transition model aligned with dual agent"""
        try:
            cbr, sinr, neighbors = current_state  # CHANGED: sinr instead of snr
            
            # Simulate CBR change (align with dual agent approach)
            new_cbr = cbr + random.uniform(-0.05, 0.05)
            new_cbr = max(0.0, min(1.0, new_cbr))
            
            # Simulate SINR change (align with dual agent approach)
            new_sinr = sinr + random.uniform(-2.0, 2.0)
            new_sinr = max(0.0, new_sinr)
            
            # Simulate neighbor count change (align with dual agent approach)
            neighbor_change = random.choice([-1, 0, 1])
            new_neighbors = max(0, neighbors + neighbor_change)
            
            return [new_cbr, new_sinr, new_neighbors]
        except Exception as e:
            logger.error(f"Error in state simulation: {e}")
            return current_state
    
    def stop(self):
        """Stop the server and save shared models"""
        logger.info(" STOPPING SERVER - SAVING FINAL MODELS")
        self.running = False
        
        try:
            self.server.close()
        except:
            pass
        
        if self.training_mode:
            logger.info(f" SAVING SHARED MODELS at final episode {self.global_episode_count}")
            success = SharedModelManager.save_models(
                self.shared_feature_extractor,
                self.shared_agent,
                episode=self.global_episode_count
            )
            if success:
                logger.info(" SHARED MODELS SAVED SUCCESSFULLY")
            else:
                logger.error(" FAILED TO SAVE SHARED MODELS")
                
            # Also save individual vehicle models
            for vehicle_id, vehicle in self.vehicle_nodes.items():
                try:
                    vehicle.save_models(episode=self.global_episode_count)
                    logger.info(f" SAVED MODELS FOR VEHICLE {vehicle_id}")
                except Exception as e:
                    logger.error(f" FAILED TO SAVE MODELS FOR VEHICLE {vehicle_id}: {e}")
                    
        log_message(" ULTRA-EXPLORATIVE SINGLE AGENT Server stopped")

# Legacy wrapper for compatibility
class DecentralizedRLServer(UltraExplorativeDecentralizedRLServer):
    """Backward compatibility wrapper"""
    pass

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # Set training_mode=False for production mode
    rl_server = UltraExplorativeDecentralizedRLServer(HOST, PORT, training_mode=False)
    try:
        log_message(" Starting ULTRA-EXPLORATIVE SINGLE AGENT RL server...")
        log_message(f"Enhanced exploration settings:")
        log_message(f"  - Initial exploration factor: {INITIAL_EXPLORATION_FACTOR}")
        log_message(f"  - Power action bounds: ±{POWER_ACTION_BOUND}")
        log_message(f"  - Beacon action bounds: ±{BEACON_ACTION_BOUND}")
        log_message(f"  - MCS action bounds: ±{MCS_ACTION_BOUND}")
        log_message(f"  - Multiple exploration mechanisms enabled")
        log_message(f"  - ULTRA-VERBOSE logging enabled")
        rl_server.start()
    except KeyboardInterrupt:
        log_message(" ULTRA-EXPLORATIVE Server interrupted by user. Shutting down...")
        rl_server.stop()
    except Exception as e:
        log_message(f"Fatal error: {str(e)}", "CRITICAL")
        rl_server.stop()
