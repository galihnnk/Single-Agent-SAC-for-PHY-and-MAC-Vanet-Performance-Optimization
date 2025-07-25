# Single Agent SAC for VANET Optimization

An advanced single-agent reinforcement learning system using Soft Actor-Critic (SAC) with ultra-aggressive exploration for comprehensive VANET (Vehicular Ad-hoc Network) parameter optimization. The system uses a unified agent to simultaneously control transmission power, beacon rates, and modulation coding schemes with maximum exploration coverage of the parameter space.

## Overview

This ultra-explorative single agent RL system addresses VANET parameter optimization through a unified approach where one sophisticated SAC agent controls all communication parameters simultaneously. By implementing multiple aggressive exploration mechanisms, the system achieves comprehensive coverage of the parameter space, ensuring optimal performance discovery across diverse vehicular scenarios.

## Key Features

### Ultra-Aggressive Exploration
- **Massive Action Bounds**: Power (±8.0 dBm), Beacon (±6.0 Hz), MCS (±3.0 levels)
- **5x Exploration Factor**: Initial multiplicative exploration with gradual decay
- **Multiple Exploration Mechanisms**: Multiplicative, additive noise, diversity-based, and random actions
- **Enhanced Entropy**: High initial variance (log_std=1.0) for maximum stochasticity

### Unified Agent Architecture
- **Single SAC Agent**: Controls all three parameters (power, beacon rate, MCS) simultaneously
- **Shared Feature Extraction**: Common state processing for all parameter decisions
- **Coordinated Actions**: Joint optimization of all communication parameters
- **Prioritized Experience Replay**: Advanced experience sampling for efficient learning

### Advanced Exploration Mechanisms
- **Multiplicative Exploration**: Actions scaled by exploration factor (5.0 → 0.8)
- **Strong Additive Noise**: Gaussian noise scaled by exploration factor
- **Diversity-Based Exploration**: Actions adjusted based on recent action variance
- **Random Action Bursts**: 15% probability of completely random actions

### Production-Ready Features
- **Shared Model Management**: Centralized model saving and loading system
- **Ultra-Verbose Logging**: Detailed exploration and training metrics
- **TensorBoard Integration**: Real-time exploration visualization
- **Robust Error Handling**: NaN protection and graceful degradation

## Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Core dependencies
pip install torch numpy pandas
pip install tensorboard

# Optional: Additional analysis tools
pip install matplotlib seaborn
```

### Quick Start

1. **Configure the system** (edit script parameters):
```python
# Ultra-aggressive exploration parameters
INITIAL_EXPLORATION_FACTOR = 5.0   # Maximum exploration
EXPLORATION_DECAY = 0.9999         # Slow decay
MIN_EXPLORATION = 0.8              # High minimum
RANDOM_ACTION_PROB = 0.15          # 15% random actions

# Massive action bounds for full parameter space coverage
POWER_ACTION_BOUND = 8.0      # ±8 dBm (was ±3)
BEACON_ACTION_BOUND = 6.0     # ±6 Hz (was ±1)  
MCS_ACTION_BOUND = 3.0        # ±3 levels (was ±1)
```

2. **Training Mode**:
```bash
# Run with training enabled
python ultra_explorative_single_agent.py
```

3. **Production Mode**:
```python
# Edit the script
training_mode = False

# Run with trained models
python ultra_explorative_single_agent.py
```

## Architecture Details

### Single Agent Design

```
┌─────────────────────────────────────┐
│        Ultra-Explorative            │
│        Single SAC Agent             │
│                                     │
│  ┌─────────────────────────────┐   │
│  │   Shared Feature Extractor  │   │
│  │   (State Processing)        │   │
│  └─────────────────────────────┘   │
│                │                    │
│  ┌─────────────────────────────┐   │
│  │      Unified Actor          │   │
│  │                             │   │
│  │  • Power Control (±8.0)     │   │
│  │  • Beacon Rate (±6.0)       │   │
│  │  • MCS Selection (±3.0)     │   │
│  └─────────────────────────────┘   │
│                │                    │
│  ┌─────────────────────────────┐   │
│  │    Dual Critics (Q1, Q2)    │   │
│  │  (Action-Value Estimation)  │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Multi-Layer Exploration System

| Exploration Type | Mechanism | Impact |
|------------------|-----------|---------|
| **Multiplicative** | Actions × exploration_factor | 5x initial scaling |
| **Additive Noise** | Gaussian noise injection | Strong randomization |
| **Diversity-Based** | Variance-based adjustments | Prevents action clustering |
| **Random Actions** | Complete parameter randomization | 15% exploration bursts |

## Configuration Guide

### Exploration Parameters

```python
# Ultra-aggressive settings for maximum exploration
INITIAL_EXPLORATION_FACTOR = 5.0   # Much higher than standard (2.0)
EXPLORATION_DECAY = 0.9999         # Much slower than standard (0.9995)
MIN_EXPLORATION = 0.8              # Much higher than standard (1.0)
EXPLORATION_NOISE_SCALE = 1.2      # Strong additional noise
ACTION_NOISE_SCALE = 0.8           # Action-dependent noise
RANDOM_ACTION_PROB = 0.15          # 15% random actions initially
```

### Action Boundaries

```python
# Massive bounds for full parameter space coverage
POWER_ACTION_BOUND = 8.0      # ±8 dBm (system range: 0-30 dBm)
BEACON_ACTION_BOUND = 6.0     # ±6 Hz (system range: 1-20 Hz)
MCS_ACTION_BOUND = 3.0        # ±3 levels (system range: 0-10)

# With 5x exploration factor, maximum exploration ranges:
# Power: ±40 dBm (exceeds system bounds for full coverage)
# Beacon: ±30 Hz (exceeds system bounds for full coverage)
# MCS: ±15 levels (exceeds system bounds for full coverage)
```

### Enhanced Entropy Settings

```python
# High variance for maximum exploration
INITIAL_LOG_STD = 1.0              # Much higher initial variance
LOG_STD_MIN = -8                   # Less restrictive minimum
LOG_STD_MAX = 4                    # Higher maximum variance
```

## Performance Metrics

### Exploration Coverage
- **Parameter Space Coverage**: Full coverage of all parameter combinations
- **Action Diversity**: Multi-mechanism diversity maintenance
- **Exploration Efficiency**: Rapid discovery of optimal regions
- **Convergence Balance**: Exploration vs exploitation trade-off

### Training Metrics
- **Experience Diversity**: Variance in stored experiences
- **Action Distribution**: Spread across parameter space
- **Reward Discovery**: Identification of high-reward regions
- **Training Stability**: Convergence despite aggressive exploration

## Usage Examples

### Maximum Exploration Training
```python
# Configure for maximum parameter space exploration
INITIAL_EXPLORATION_FACTOR = 5.0
EXPLORATION_DECAY = 0.9999
MIN_EXPLORATION = 0.8
RANDOM_ACTION_PROB = 0.2  # Even more random actions

# Ultra-large action bounds
POWER_ACTION_BOUND = 10.0
BEACON_ACTION_BOUND = 8.0
MCS_ACTION_BOUND = 4.0
```

### Balanced Exploration Training
```python
# More conservative exploration for stable training
INITIAL_EXPLORATION_FACTOR = 3.0
EXPLORATION_DECAY = 0.9995
MIN_EXPLORATION = 0.5
RANDOM_ACTION_PROB = 0.1

# Standard action bounds
POWER_ACTION_BOUND = 6.0
BEACON_ACTION_BOUND = 4.0
MCS_ACTION_BOUND = 2.0
```

### Production Deployment
```python
# Disable exploration for production use
training_mode = False
# System will use deterministic actions from trained policy
```

## Output Files

### Model Management
- **Shared Models**: `saved_models/shared_EPISODE_TIMESTAMP/`
  - `agent.pth` - Complete SAC agent (actor + critics)
  - `feature_extractor.pth` - Shared feature extraction network
- **Vehicle Models**: `saved_models/vehicle_ID_EPISODE_TIMESTAMP/`
  - Individual vehicle states and parameters

### Training Logs
- **Main Log**: `ultra_aggressive_single_agent.log`
- **TensorBoard**: `ultra_aggressive_single_agent_logs/vehicle_*/`
- **Exploration Metrics**: Real-time diversity and coverage tracking

## Advanced Features

### Prioritized Experience Replay
```python
# Advanced experience sampling based on TD-error
replay_buffer = PrioritizedReplayBuffer(
    state_size=3,
    action_size=3,
    buffer_size=100000,
    alpha=0.6,
    beta=0.4
)
```

### Multi-Mechanism Exploration
```python
# 1. Multiplicative scaling
actions = actions * exploration_factor

# 2. Additive noise injection
noise = torch.randn_like(actions) * noise_scale
actions = actions + noise

# 3. Diversity-based adjustments
diversity_noise = torch.randn_like(actions) * (1.0 / action_std)
actions = actions + diversity_noise

# 4. Random action bursts
if random.random() < random_action_prob:
    actions = torch.uniform(-bounds, bounds)
```

### Ultra-Verbose Logging
- **Action Selection**: Detailed logging of all exploration mechanisms
- **Parameter Changes**: Before/after comparisons for all adjustments
- **Training Progress**: Comprehensive metrics for convergence analysis
- **Diversity Metrics**: Real-time action variance and coverage tracking

## Troubleshooting

### Common Issues

**Excessive Exploration**
```
[WARNING] Actions exceeding reasonable bounds due to aggressive exploration
```
*Solution*: Reduce `INITIAL_EXPLORATION_FACTOR` or increase `EXPLORATION_DECAY`

**NaN in Actions**
```
[WARNING] NaN detected in actions, replacing with zeros
```
*Solution*: System automatically handles NaN values. Check entropy settings if persistent.

**Training Instability**
```
[ERROR] High variance in training losses
```
*Solution*: Reduce exploration parameters or increase training frequency

### Performance Optimization

**Slow Exploration Convergence**
- Increase `RANDOM_ACTION_PROB` for more diverse exploration
- Adjust action bounds to focus on relevant parameter ranges
- Monitor diversity metrics to ensure adequate coverage

**Poor Parameter Coverage**
- Increase `INITIAL_EXPLORATION_FACTOR`
- Extend `MIN_EXPLORATION` duration
- Check action diversity metrics in TensorBoard

## Research Applications

- **Parameter Space Analysis**: Comprehensive exploration of VANET parameter interactions
- **Optimal Region Discovery**: Identification of high-performance parameter combinations
- **Exploration Algorithm Research**: Multi-mechanism exploration strategy evaluation
- **VANET Performance Limits**: Discovery of theoretical performance boundaries
- **Unified Control Systems**: Single-agent approaches to multi-parameter optimization

## Technical Specifications

### System Requirements
- **Memory**: 8GB RAM minimum for large replay buffers
- **Compute**: GPU recommended for faster training
- **Storage**: 2GB for models and extensive logging
- **Network**: TCP socket communication for MATLAB integration

### Exploration Guarantees
- **Full Parameter Coverage**: Mathematical guarantee of parameter space exploration
- **Diversity Maintenance**: Multiple mechanisms prevent action clustering
- **Adaptive Exploration**: Dynamic adjustment based on training progress
- **Convergence Safety**: Bounded exploration with gradual decay

## License

This project is licensed under the MIT License.

## Acknowledgments

- **SAC Algorithm**: Soft Actor-Critic research for entropy-regularized RL
- **Exploration Research**: Multi-mechanism exploration strategy development
- **PyTorch Team**: Excellent deep learning framework for RL implementation
- **VANET Community**: Domain expertise for realistic parameter optimization

---

**⭐ Star this repository if it helps your VANET exploration research!**

*Built with 🔬 for comprehensive parameter space exploration*
