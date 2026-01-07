from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
import os
import csv
from dotenv import load_dotenv
from datetime import datetime, timezone
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
import uuid

# Load environment variables
load_dotenv()

# -----------------------
# App and Database Setup
# -----------------------
app = Flask(__name__)
CORS(app)

mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/therapyconnect")
client = MongoClient(mongo_uri)
db = client.therapyconnect

# Collections
users_collection = db.users
therapists_collection = db.therapists
instructors_collection = db.instructors
sessions_collection = db.sessions
games_collection = db.games
patient_games_collection = db.patient_games
reports_collection = db.reports

# -----------------------
# Helper Functions and Classes
# -----------------------
class GameConfig:
    """Game configuration class for bubble_game"""
    def __init__(self, game_name: str, difficulty: str = "medium", enabled: bool = True, target_score: int = 20, spawnAreaSize: float = 5.0, bubbleSpeedAction: float = 5.0, bubbleLifetime: float = 3.0, spawnHeight: float = 3.0, numBubbles: int = 10,bubbleSize: float = 1.0):
        self.game_name = game_name
        self.difficulty = difficulty
        self.enabled = enabled
        self.target_score = target_score
        self.spawnAreaSize = spawnAreaSize
        self.bubbleSpeedAction = bubbleSpeedAction
        self.bubbleLifetime = bubbleLifetime
        self.spawnHeight = spawnHeight
        self.numBubbles = numBubbles
        self.bubbleSize = bubbleSize

def game_config_to_dict(config: GameConfig) -> dict:
    """Convert GameConfig to dictionary"""
    return {
        'game_name': config.game_name,
        'difficulty': config.difficulty,
        'target_score': config.target_score,
        'max_bubbles': config.max_bubbles,
        'spawn_area': config.spawn_area,
        'enabled': config.enabled
    }

def dict_to_game_config(data: dict) -> GameConfig:
    """Convert dictionary to GameConfig"""
    return GameConfig(
        game_name=data.get('game_name', ''),
        difficulty=data.get('difficulty', 'medium'),
        target_score=data.get('target_score', 20),
        max_bubbles=data.get('max_bubbles', 10),
        spawn_area=data.get('spawn_area', {'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5}),
        enabled=data.get('enabled', True)
    )

def generate_user_code(name: str) -> str:
    """Generate a unique user code from name initials and random digits"""
    initials = ''.join(word[0].upper() for word in name.split() if word)
    if not initials:
        initials = "UC"
    random_digits = ''.join(random.choices(string.digits, k=4))
    return f"{initials}{random_digits}"

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())[:8].upper()

def to_json(document: dict) -> dict:
    """Convert MongoDB ObjectId to string in a document"""
    if not document:
        return {}
    
    doc = document.copy()
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
        elif isinstance(value, list):
            doc[key] = [str(item) if isinstance(item, ObjectId) else item for item in value]
        elif isinstance(value, dict):
            doc[key] = to_json(value)
    
    return doc

def is_valid_email(email: str) -> bool:
    """Basic email validation"""
    return '@' in email and '.' in email

# -----------------------
# PPO Implementation
# -----------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.action_head(x)
        return logits

    def get_dist(self, state):
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class CustomPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=10000)

    def update_alpha(self, new_lr):
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_alpha(self):
        return self.policy_optimizer.param_groups[0]['lr']

    def update_gamma(self, new_gamma):
        self.gamma = new_gamma

    def get_gamma(self):
        return self.gamma

    def process_actions(self, action_index):
        delta_size = 0.2
        delta_prob = 0.05
        delta_spawn_rate = 1
        adjustments = {
            "bubble_size": 0.0,
            "negative_prob": 0.0,
            "positive_prob": 0.0,
            "spawn_rate": 0.0,
        }
        if action_index == 0:
            adjustments["bubble_size"] += delta_size
        elif action_index == 1:
            adjustments["bubble_size"] -= delta_size
        elif action_index == 2:
            adjustments["negative_prob"] += delta_prob
        elif action_index == 3:
            adjustments["negative_prob"] -= delta_prob
        elif action_index == 4:
            adjustments["positive_prob"] += delta_prob
        elif action_index == 5:
            adjustments["positive_prob"] -= delta_prob
        elif action_index == 6:
            adjustments["spawn_rate"] += delta_spawn_rate
        elif action_index == 7:
            adjustments["spawn_rate"] -= delta_spawn_rate
        return adjustments

    def get_action(self, state):
        device = next(self.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist = self.policy_net.get_dist(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self, batch_size=64, epochs=10):
        if len(self.memory) < batch_size:
            return
        device = next(self.policy_net.parameters()).device
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(epochs):
            dist = self.policy_net.get_dist(states)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), targets)
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        self.memory.clear()

state_dim = 7
action_dim = 8
ppo_model = CustomPPO(state_dim, action_dim)

MODEL_PATH = "ppo_model.pth"

def save_model():
    checkpoint = {
        "policy_state_dict": ppo_model.policy_net.state_dict(),
        "value_state_dict": ppo_model.value_net.state_dict(),
        "policy_optimizer": ppo_model.policy_optimizer.state_dict(),
        "value_optimizer": ppo_model.value_optimizer.state_dict(),
        "gamma": ppo_model.gamma,
        "clip_epsilon": ppo_model.clip_epsilon,
    }
    torch.save(checkpoint, MODEL_PATH)

def load_model():
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        ppo_model.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        ppo_model.value_net.load_state_dict(checkpoint["value_state_dict"])
        ppo_model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        ppo_model.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        ppo_model.gamma = checkpoint["gamma"]
        ppo_model.clip_epsilon = checkpoint["clip_epsilon"]
        print("✅ PPO Model loaded")
    else:
        print("⚠️ No saved model found")

load_model()

# -----------------------
# DQN Implementation
# -----------------------

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

class CustomDQN:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Main and target networks
        self.q_network = DQNetwork(state_dim, action_dim)
        self.target_network = DQNetwork(state_dim, action_dim)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def update_lr(self, new_lr):
        """Update learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

    def update_gamma(self, new_gamma):
        """Update discount factor"""
        self.gamma = new_gamma

    def get_gamma(self):
        """Get current discount factor"""
        return self.gamma

    def process_actions(self, action_index):
        """
        Convert action index to environment adjustments
        Actions:
        0-3: Set light 0-3 to Red
        4-7: Set light 0-3 to Orange
        8-11: Set light 0-3 to Green
        12-13: Increase/Decrease light change speed
        14-15: Increase/Decrease spawn rate
        """
        delta_speed = 0.2
        delta_spawn = 0.5
        
        adjustments = {
            "light_states": [-1, -1, -1, -1],  # -1 means no change, 0=Red, 1=Orange, 2=Green
            "light_speed_change": 0.0,
            "spawn_rate_change": 0.0,
        }
        
        # Light state changes (0-11)
        if action_index < 12:
            light_index = action_index % 4
            light_state = action_index // 4
            adjustments["light_states"][light_index] = light_state
        # Light speed changes (12-13)
        elif action_index == 12:
            adjustments["light_speed_change"] = -delta_speed  # Decrease (faster changes)
        elif action_index == 13:
            adjustments["light_speed_change"] = delta_speed   # Increase (slower changes)
        # Spawn rate changes (14-15)
        elif action_index == 14:
            adjustments["spawn_rate_change"] = -delta_spawn   # Spawn faster
        elif action_index == 15:
            adjustments["spawn_rate_change"] = delta_spawn    # Spawn slower
        
        return adjustments

    def get_action(self, state, training=True):
        """
        Epsilon-greedy action selection
        Returns: action_index, q_value
        """
        device = next(self.q_network.parameters()).device
        
        # Exploration
        if training and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                q_value = q_values[0, action].item()
            return action, q_value
        
        # Exploitation
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()
            q_value = q_values[0, action].item()
        
        return action, q_value

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        """Train the DQN using experience replay"""
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0, "q_value": 0.0}
        
        device = next(self.q_network.parameters()).device
        
        # Sample random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "loss": loss.item(),
            "q_value": current_q_values.mean().item(),
            "epsilon": self.epsilon
        }

# State: [leftHand_active, leftLeg_active, rightLeg_active, rightHand_active,
#         light0_state(0-2), light1_state, light2_state, light3_state,
#         active_lanterns(0-5), score_normalized(-1 to 1)]
state_dim = 10

# Actions: 12 light changes + 2 speed changes + 2 spawn rate changes = 16 actions
action_dim = 16

dqn_model = CustomDQN(state_dim, action_dim)

MODEL_PATH = "dqn_rogl_model.pth"

def save_model_dqn():
    """Save DQN model checkpoint"""
    checkpoint = {
        "q_network_state_dict": dqn_model.q_network.state_dict(),
        "target_network_state_dict": dqn_model.target_network.state_dict(),
        "optimizer_state_dict": dqn_model.optimizer.state_dict(),
        "epsilon": dqn_model.epsilon,
        "gamma": dqn_model.gamma,
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"✅ DQN Model saved to {MODEL_PATH}")

def load_model_dqn():
    """Load DQN model checkpoint"""
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        dqn_model.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        dqn_model.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        dqn_model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        dqn_model.epsilon = checkpoint["epsilon"]
        dqn_model.gamma = checkpoint["gamma"]
        print("✅ DQN Model loaded successfully")
    else:
        print("⚠️ No saved model found, starting fresh")

# Load model at startup
load_model_dqn()

# -----------------------
# MC PPO Implementation
# -----------------------

class MCPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        # Output continuous actions in range [-1, 1]
        return self.tanh(self.action_head(x))

    def get_dist(self, state):
        mean = self.forward(state)
        # Small fixed std for exploration
        std = torch.ones_like(mean) * 0.1
        return torch.distributions.Normal(mean, std)


class MCValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.value_head(x)


class CustomMCPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = MCPolicyNetwork(state_dim, action_dim)
        self.value_net = MCValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=10000)

    def update_alpha(self, new_lr):
        """Update learning rate"""
        for param_group in self.policy_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_alpha(self):
        """Get current learning rate"""
        return self.policy_optimizer.param_groups[0]['lr']

    def update_gamma(self, new_gamma):
        """Update discount factor"""
        self.gamma = new_gamma

    def get_gamma(self):
        """Get current discount factor"""
        return self.gamma

    def process_actions(self, action_array):
        """
        Convert network output to environment adjustments
        action_array: [upper_threshold_delta, lower_threshold_delta, gap_delta, difficulty_delta]
        All values in range [-1, 1]
        """
        delta_threshold = 5.0  # Max threshold change per step
        delta_gap = 1.0        # Max gap change per step
        
        adjustments = {
            "upper_threshold_change": float(action_array[0]) * delta_threshold,  # ±5%
            "lower_threshold_change": float(action_array[1]) * delta_threshold,  # ±5%
            "gap_change": float(action_array[2]) * delta_gap,                     # ±1 second
            "difficulty_change": float(action_array[3])                            # -1 to 1 (for selection)
        }
        
        return adjustments

    def get_action(self, state):
        """
        Sample action from policy
        Returns: action_array, log_prob
        """
        device = next(self.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        dist = self.policy_net.get_dist(state_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action.cpu().detach().numpy()[0], log_prob.item()

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done, log_prob))

    def update(self, batch_size=64, epochs=10):
        """Train PPO using collected experiences"""
        if len(self.memory) < batch_size:
            return {"loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
        
        device = next(self.policy_net.parameters()).device
        
        # Get all experiences
        states, actions, rewards, next_states, dones, old_log_probs = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            targets = rewards + self.gamma * next_values * (1 - dones)
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO training loop
        total_policy_loss = 0
        total_value_loss = 0
        
        for epoch in range(epochs):
            # Get current policy distribution
            dist = self.policy_net.get_dist(states)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            
            # PPO clipped objective
            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            
            # Value loss
            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), targets)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            # Update value
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        self.memory.clear()
        
        return {
            "loss": (total_policy_loss + total_value_loss) / epochs,
            "policy_loss": total_policy_loss / epochs,
            "value_loss": total_value_loss / epochs
        }


# State: [similarity_history (last 5), current_score, difficulty_level]
# Total: 5 + 1 + 1 = 7 dimensions
state_dim = 7

# Actions: [upper_threshold_change, lower_threshold_change, gap_change, difficulty_change]
# Total: 4 continuous actions
action_dim = 4

mc_ppo_model = CustomMCPPO(state_dim, action_dim)

MC_MODEL_PATH = "mc_ppo_model.pth"

def save_mc_model():
    """Save MC PPO model checkpoint"""
    checkpoint = {
        "policy_state_dict": mc_ppo_model.policy_net.state_dict(),
        "value_state_dict": mc_ppo_model.value_net.state_dict(),
        "policy_optimizer": mc_ppo_model.policy_optimizer.state_dict(),
        "value_optimizer": mc_ppo_model.value_optimizer.state_dict(),
        "gamma": mc_ppo_model.gamma,
        "clip_epsilon": mc_ppo_model.clip_epsilon,
    }
    torch.save(checkpoint, MC_MODEL_PATH)
    print(f"✅ MC PPO Model saved to {MC_MODEL_PATH}")

def load_mc_model():
    """Load MC PPO model checkpoint"""
    if os.path.exists(MC_MODEL_PATH):
        try:
            checkpoint = torch.load(MC_MODEL_PATH, map_location="cpu")
            mc_ppo_model.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            mc_ppo_model.value_net.load_state_dict(checkpoint["value_state_dict"])
            mc_ppo_model.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
            mc_ppo_model.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
            mc_ppo_model.gamma = checkpoint["gamma"]
            mc_ppo_model.clip_epsilon = checkpoint["clip_epsilon"]
            print("✅ MC PPO Model loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return False
    else:
        print("⚠️ No saved MC model found, starting fresh")
        return False

# Load model at startup
load_mc_model()

# -----------------------
# Routes
# -----------------------
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        user_type = data.get('userType')
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if not all([user_type, name, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if user_type not in ['therapist', 'instructor']:
            return jsonify({'error': 'Invalid user type'}), 400
        
        if not is_valid_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'User already exists with this email'}), 400

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_code = generate_user_code(name)
        
        while users_collection.find_one({'userCode': user_code}):
            user_code = generate_user_code(name)

        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'userType': user_type,
            'userCode': user_code,
            'createdAt': datetime.now(timezone.utc)
        }
        
        result = users_collection.insert_one(user_data)
        user_id = result.inserted_id

        if user_type == 'therapist':
            therapists_collection.insert_one({
                'userId': user_id,
                'name': name,
                'email': email,
                'userCode': user_code,
                'patients': []
            })
        else:
            instructors_collection.insert_one({
                'userId': user_id,
                'name': name,
                'email': email,
                'userCode': user_code,
                'sessions': []
            })

        return jsonify({
            'message': 'User created successfully',
            'userCode': user_code,
            'userId': str(user_id),
            'userType': user_type
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        user = users_collection.find_one({'email': email})
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Invalid credentials'}), 401

        return jsonify({
            'message': 'Login successful',
            'user': {
                'id': str(user['_id']),
                'name': user['name'],
                'email': user['email'],
                'userType': user['userType'],
                'userCode': user.get('userCode')
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/patients', methods=['GET'])
def get_therapist_patients():
    try:
        therapist_id = request.args.get('therapistId')
        if not therapist_id:
            return jsonify({'error': 'therapistId is required'}), 400

        therapist = therapists_collection.find_one({'userId': ObjectId(therapist_id)})
        if not therapist:
            return jsonify({'error': 'Therapist not found'}), 404

        patient_ids = therapist.get('patients', [])
        patients = []
        
        for patient_id in patient_ids:
            patient = users_collection.find_one({'_id': ObjectId(patient_id)})
            if patient:
                patients.append(to_json(patient))

        return jsonify({'patients': patients}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/add-patient', methods=['POST'])
def add_patient():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        therapist_id = data.get('therapistId')
        patient_data = data.get('patient')

        if not therapist_id or not patient_data:
            return jsonify({'error': 'Missing required fields'}), 400

        therapist = therapists_collection.find_one({'userId': ObjectId(therapist_id)})
        if not therapist:
            return jsonify({'error': 'Therapist not found'}), 404

        patient_user_data = {
            'name': patient_data.get('name'),
            'email': patient_data.get('email'),
            'age': patient_data.get('age'),
            'condition': patient_data.get('condition'),
            'userType': 'patient',
            'therapistId': ObjectId(therapist_id),
            'createdAt': datetime.now(timezone.utc)
        }
        
        user_code = generate_user_code(patient_user_data['name'])
        while users_collection.find_one({'userCode': user_code}):
            user_code = generate_user_code(patient_user_data['name'])
        patient_user_data['userCode'] = user_code

        patient_result = users_collection.insert_one(patient_user_data)
        patient_id = patient_result.inserted_id

        therapists_collection.update_one(
            {'userId': ObjectId(therapist_id)},
            {'$push': {'patients': patient_id}}
        )

        return jsonify({
            'message': 'Patient added successfully',
            'patientId': str(patient_id),
            'userCode': user_code
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/assign-instructor', methods=['POST'])
def assign_instructor():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        therapist_id = data.get('therapistId')
        patient_id = data.get('patientId')
        instructor_id = data.get('instructorId')

        if not all([therapist_id, patient_id, instructor_id]):
            return jsonify({'error': 'Missing required fields'}), 400

        therapist = therapists_collection.find_one({'userId': ObjectId(therapist_id)})
        if not therapist:
            return jsonify({'error': 'Therapist not found'}), 404

        if ObjectId(patient_id) not in therapist.get('patients', []):
            return jsonify({'error': 'Patient not found for this therapist'}), 404

        instructor = instructors_collection.find_one({'userId': ObjectId(instructor_id)})
        if not instructor:
            return jsonify({'error': 'Instructor not found'}), 404

        users_collection.update_one(
            {'_id': ObjectId(patient_id)},
            {'$set': {'instructorId': ObjectId(instructor_id)}}
        )

        return jsonify({'message': 'Instructor assigned successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Update the session creation endpoint to handle reviews for existing sessions
@app.route('/api/session', methods=['POST'])
def create_session():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        therapist_id = data.get('therapistId')
        patient_id = data.get('patientId')
        instructor_id = data.get('instructorId')
        game_data = data.get('gameData', {})
        review = data.get('review', '')
        rating = data.get('rating', 0)
        session_id = data.get('sessionId')  # Allow passing existing session ID
        
        # Check if this is a review for an existing session
        existing_session = None
        if session_id:
            existing_session = sessions_collection.find_one({'sessionId': session_id})
            if not existing_session:
                return jsonify({'error': 'Session not found'}), 404
        
        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        if rating and (rating < 1 or rating > 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400

        # If updating an existing session with a review
        if existing_session:
            update_data = {
                'review': review,
                'rating': rating,
                'status': 'completed',
                'endTime': datetime.now(timezone.utc)
            }
            
            # Add game data if provided
            if game_data:
                update_data['gameData'] = {**existing_session.get('gameData', {}), **game_data}
            
            sessions_collection.update_one(
                {'sessionId': session_id},
                {'$set': update_data}
            )
            
            return jsonify({
                'message': 'Session review added successfully',
                'sessionId': session_id
            }), 200
        
        # Otherwise create a new session
        if not session_id:
            session_id = generate_session_id()

        session_data = {
            'sessionId': session_id,
            'therapistId': ObjectId(therapist_id),
            'patientId': ObjectId(patient_id),
            'gameData': game_data,
            'review': review,
            'rating': rating,
            'date': datetime.now(timezone.utc),
            'status': 'active'
        }
        
        if instructor_id:
            session_data['instructorId'] = ObjectId(instructor_id)

        result = sessions_collection.insert_one(session_data)
        session_id_db = result.inserted_id

        if instructor_id:
            instructors_collection.update_one(
                {'userId': ObjectId(instructor_id)},
                {'$push': {'sessions': session_id_db}}
            )

        return jsonify({
            'message': 'Session created successfully',
            'sessionId': session_id,
            'sessionIdDb': str(session_id_db)
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['PUT'])
def update_session(session_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        game_data = data.get('gameData', {})
        status = data.get('status')

        update_data = {}
        if game_data:
            update_data['gameData'] = game_data
        if status:
            update_data['status'] = status

        if not update_data:
            return jsonify({'error': 'No data to update'}), 400

        # Use the session_id from URL, not the JSON
        sessions_collection.update_one(
            {'sessionId': session_id},
            {'$set': update_data}
        )

        return jsonify({'message': 'Session updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/session/<session_id>/close', methods=['POST'])
def close_session(session_id):
    try:
        if not session_id:
            return jsonify({'error': 'sessionId is required'}), 400

        sessions_collection.update_one(
            {'sessionId': session_id},
            {'$set': {'status': 'completed', 'endTime': datetime.now(timezone.utc)}}
        )

        return jsonify({'message': 'Session closed successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    try:
        user_id = request.args.get('userId')
        user_code = request.args.get('userCode')
        user_type = request.args.get('userType')
        patient_id = request.args.get('patientId')

        if not user_type:
            return jsonify({'error': 'userType is required'}), 400

        query = {}
        
        if user_type == 'therapist':
            if user_id:
                query['therapistId'] = ObjectId(user_id)
            elif user_code:
                therapist = therapists_collection.find_one({'userCode': user_code})
                if therapist:
                    query['therapistId'] = therapist['userId']
                else:
                    return jsonify({'error': 'Therapist not found'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required'}), 400
                
        elif user_type == 'instructor':
            if user_id:
                query['instructorId'] = ObjectId(user_id)
            elif user_code:
                instructor = instructors_collection.find_one({'userCode': user_code})
                if instructor:
                    query['instructorId'] = instructor['userId']
                else:
                    return jsonify({'error': 'Instructor not found'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required'}), 400
                
        elif user_type == 'patient':
            if user_id:
                query['patientId'] = ObjectId(user_id)
            elif user_code:
                patient = users_collection.find_one({'userCode': user_code, 'userType': 'patient'})
                if patient:
                    query['patientId'] = patient['_id']
                else:
                    return jsonify({'error': 'Patient not found'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required'}), 400
        else:
            return jsonify({'error': 'Invalid userType'}), 400

        if patient_id:
            query['patientId'] = ObjectId(patient_id)

        sessions = list(sessions_collection.find(query).sort('date', -1))
        return jsonify({'sessions': [to_json(s) for s in sessions]}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    try:
        session = sessions_collection.find_one({'sessionId': session_id})
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        return jsonify({'session': to_json(session)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
@app.route('/api/session/id/<id>', methods=['GET'])
def get_session_id(id):
    try:
        session = sessions_collection.find_one({'_id': ObjectId(id)})
        
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        return jsonify({'session': to_json(session)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/user/<user_id>', methods=['GET'])
def get_user_by_id(user_id):
    """Get user details by user ID"""
    try:
        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Validate if user_id is a valid ObjectId
        if not ObjectId.is_valid(user_id):
            return jsonify({'error': 'Invalid user ID format'}), 400

        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Return user data without sensitive information
        user_data = {
            'id': str(user['_id']),
            'name': user['name'],
            'email': user.get('email', ''),
            'userType': user.get('userType', ''),
            'userCode': user.get('userCode', ''),
            'createdAt': user.get('createdAt', '')
        }

        return jsonify({'user': user_data}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/user-by-code', methods=['GET'])
def get_user_by_code():
    try:
        user_code = request.args.get('userCode')
        if not user_code:
            return jsonify({'error': 'userCode is required'}), 400

        user = users_collection.find_one({'userCode': user_code})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({'user': to_json(user)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/patient-games', methods=['GET'])
def get_patient_games():
    try:
        patient_id = request.args.get('patientId')
        therapist_id = request.args.get('therapistId')
        
        if not patient_id or not therapist_id:
            return jsonify({'error': 'patientId and therapistId are required'}), 400

        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })
        
        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        patient_games = patient_games_collection.find_one({
            'patientId': ObjectId(patient_id),
            'therapistId': ObjectId(therapist_id)
        })

        if not patient_games:
            default_configs = {
                'bubble_game': GameConfig(
                    game_name='bubble_game',
                    difficulty='medium',
                    target_score=20,
                    max_bubbles=10,
                    spawn_area={'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5},
                    enabled=True
                ).__dict__
            }
            return jsonify({
                'games': default_configs
            }), 200

        return jsonify({
            'games': patient_games.get('game_configs', {})
        }), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/update-patient-games', methods=['POST'])
def update_patient_games():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        therapist_id = data.get('therapistId')
        patient_id = data.get('patientId')
        game_configs = data.get('gameConfigs', {})

        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })
        
        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        validated_configs = {}
        for game_name, config in game_configs.items():
            if config.get('enabled', True) and game_name == 'bubble_game':
                if not all(key in config for key in ["difficulty", "spawnAreaSize", "bubbleSpeedAction", "bubbleLifetime", "spawnHeight", "numBubbles", "bubbleSize"]):
                    return jsonify({'error': f'Missing required fields for {game_name}'}), 400

            validated_configs[game_name] = config

        patient_games_collection.update_one(
            {
                'patientId': ObjectId(patient_id),
                'therapistId': ObjectId(therapist_id)
            },
            {
                '$set': {
                    'game_configs': validated_configs,
                    'updatedAt': datetime.now(timezone.utc)
                }
            },
            upsert=True
        )

        return jsonify({'message': 'Game configurations updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/patient/game-config', methods=['GET'])
def get_patient_game_config():
    try:
        patient_id = request.args.get('patientId')
        game_name = request.args.get('gameName', 'bubble_game')
        
        if not patient_id:
            return jsonify({'error': 'patientId is required'}), 400

        patient_games = patient_games_collection.find_one({
            'patientId': ObjectId(patient_id)
        })

        if not patient_games:
            default_config = GameConfig(
                game_name=game_name,
                difficulty='medium',
                target_score=20,
                max_bubbles=10,
                spawn_area={'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5},
                enabled=True
            )
            return jsonify({'config': default_config.__dict__}), 200

        config_data = patient_games.get('game_configs', {}).get(game_name)
        
        if not config_data:
            return jsonify({'error': f'No configuration found for {game_name}'}), 404

        if not config_data.get('enabled', True):
            return jsonify({'error': f'Game {game_name} is disabled for this patient'}), 403

        return jsonify({'config': config_data}), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/available-games', methods=['GET'])
def get_available_games():
    try:
        games = list(games_collection.find({'available': True}))
        
        if not games:
            default_games = [
                {
                    'name': 'bubble_game',
                    'display_name': 'Bubble Pop Game',
                    'description': 'Pop bubbles to improve coordination and reaction time',
                    'category': 'motor_skills',
                    'configurable_fields': [
                        {'name': 'difficulty', 'type': 'select', 'options': ['easy', 'medium', 'hard'], 'default': 'medium'},
                        {'name': 'target_score', 'type': 'number', 'min': 5, 'max': 100, 'default': 20},
                        {'name': 'max_bubbles', 'type': 'number', 'min': 5, 'max': 50, 'default': 10},
                        {'name': 'spawn_area', 'type': 'object', 'fields': [
                            {'name': 'x_min', 'type': 'number', 'default': -5},
                            {'name': 'x_max', 'type': 'number', 'default': 5},
                            {'name': 'y_min', 'type': 'number', 'default': 1},
                            {'name': 'y_max', 'type': 'number', 'default': 5},
                            {'name': 'z_min', 'type': 'number', 'default': -5},
                            {'name': 'z_max', 'type': 'number', 'default': 5}
                        ]}
                    ]
                }
            ]
            games_collection.insert_many(default_games)
            games = list(games_collection.find({'available': True}))
        
        return jsonify({'games': [to_json(game) for game in games]}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/instructors', methods=['GET'])
def get_instructors():
    try:
        instructors = list(instructors_collection.find({}))
        return jsonify({'instructors': [to_json(instructor) for instructor in instructors]}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/create-report', methods=['POST'])
def create_report():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        therapist_id = data.get('therapistId')
        patient_id = data.get('patientId')
        therapist_name = data.get('therapistName')
        report_data = data.get('reportData', {})
        session_ids = data.get('sessionIds', [])

        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        # Validate therapist-patient access
        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })

        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        # Fetch sessions
        sessions = list(sessions_collection.find({
            "_id": {"$in": [ObjectId(sid) for sid in session_ids]}
        }))

        # Convert sessions into structured format
        formatted_sessions = []
        for s in sessions:
            formatted_sessions.append({
                "_id": str(s["_id"]),
                "sessionId": s.get("sessionId", ""),
                "date": s.get("date"),
                "gameData": s.get("gameData", {}),
                "analytics": s.get("analytics", {}),
                "rating": s.get("rating"),
                "review": s.get("review"),
                "duration": s.get("duration"),
                "status": s.get("status", "completed"),
                "instructorId": s.get("instructorId"),
                "endTime": s.get("endTime"),
                "therapistId": s.get("therapistId"),
                "patientId": s.get("patientId")
            })

        # Extract unique instructor IDs from sessions
        instructor_ids = []
        for s in sessions:
            if s.get('instructorId') and s['instructorId'] not in instructor_ids:
                instructor_ids.append(s['instructorId'])

        # Fetch instructor details
        instructors_list = []
        for instructor_id in instructor_ids:
            try:
                instructor = users_collection.find_one({"_id": ObjectId(instructor_id)})
                if instructor:
                    instructors_list.append({
                        "_id": str(instructor["_id"]),
                        "name": instructor.get("name", "Unknown Instructor"),
                        "email": instructor.get("email", "No email"),
                        "role": instructor.get("role", "Instructor")
                    })
            except Exception as e:
                print(f"Error fetching instructor {instructor_id}: {str(e)}")

        # Calculate instructor performance metrics
        instructor_performance = []
        for instructor in instructors_list:
            instructor_sessions = [s for s in formatted_sessions if s.get('instructorId') == instructor['_id']]
            
            if instructor_sessions:
                # Calculate averages
                avg_score = sum(s.get('gameData', {}).get('score', 0) for s in instructor_sessions) / len(instructor_sessions)
                avg_rating = sum(s.get('rating', 0) for s in instructor_sessions) / len(instructor_sessions)
                
                # Calculate movement averages
                left_hand_sum = sum(s.get('gameData', {}).get('leftHandMaxFromHip', 0) for s in instructor_sessions)
                right_hand_sum = sum(s.get('gameData', {}).get('rightHandMaxFromHip', 0) for s in instructor_sessions)
                
                instructor_performance.append({
                    "instructorId": instructor['_id'],
                    "instructorName": instructor['name'],
                    "sessionsCount": len(instructor_sessions),
                    "avgScore": round(avg_score, 1),
                    "avgRating": round(avg_rating, 1),
                    "avgLeftHand": round(left_hand_sum / len(instructor_sessions), 2) if len(instructor_sessions) > 0 else 0,
                    "avgRightHand": round(right_hand_sum / len(instructor_sessions), 2) if len(instructor_sessions) > 0 else 0
                })

        # Prepare graph data
        graph_data = {
            "scoreTrend": [
                {
                    "date": s.get('date'),
                    "score": s.get('gameData', {}).get('score', 0),
                    "sessionNumber": idx + 1,
                    "instructorName": next((i['name'] for i in instructors_list if i['_id'] == s.get('instructorId')), 'Unknown')
                }
                for idx, s in enumerate(formatted_sessions)
            ],
            "movementMetrics": [
                {
                    "date": s.get('date'),
                    "leftHand": s.get('gameData', {}).get('leftHandMaxFromHip', 0),
                    "rightHand": s.get('gameData', {}).get('rightHandMaxFromHip', 0),
                    "leftLeg": s.get('gameData', {}).get('leftLegMax', 0),
                    "rightLeg": s.get('gameData', {}).get('rightLegMax', 0),
                    "instructorId": s.get('instructorId')
                }
                for s in formatted_sessions
            ],
            "gameConfig": [
                {
                    "date": s.get('date'),
                    "bubbleSpeed": s.get('analytics', {}).get('bubbleSpeedAction', 0),
                    "bubbleLifetime": s.get('analytics', {}).get('bubbleLifetime', 0),
                    "bubbleSize": s.get('analytics', {}).get('bubbleSize', 0),
                    "numBubbles": s.get('analytics', {}).get('numBubbles', 0),
                    "spawnAreaSize": s.get('analytics', {}).get('spawnAreaSize', 0),
                    "spawnHeight": s.get('analytics', {}).get('spawnHeight', 0),
                    "instructorId": s.get('instructorId')
                }
                for s in formatted_sessions
            ],
            "instructorPerformance": instructor_performance
        }

        # Prepare final report document
        report = {
            "therapistId": therapist_id,
            "patientId": patient_id,
            "therapistName": therapist_name,
            "reportData": {
                "title": report_data.get("title", ""),
                "summary": report_data.get("summary", ""),
                "progress": report_data.get("progress", ""),
                "recommendations": report_data.get("recommendations", ""),
                "aiGenerated": report_data.get("aiGenerated", {}),
                "instructors": instructors_list,
                "graphData": graph_data,
                "instructorPerformance": instructor_performance
            },
            "sessions": formatted_sessions,
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
            "version": "2.0"  # Added version to track new format
        }

        result = reports_collection.insert_one(report)
        
        return jsonify({
            "success": True,
            "message": "Report created successfully with analytics",
            "reportId": str(result.inserted_id)
        }), 200

    except Exception as e:
        print("Server error:", str(e))
        return jsonify({'error': 'Server error: ' + str(e)}), 500

@app.route('/api/therapist/reports', methods=['GET'])
def get_reports():
    try:
        therapist_id = request.args.get('therapistId')

        if not therapist_id:
            return jsonify({'error': 'therapistId is required'}), 400

        # Fetch reports for therapist
        reports_cursor = reports_collection.find(
            {"therapistId": therapist_id}
        ).sort("createdAt", -1)

        reports = []
        for report in reports_cursor:
            # Convert ObjectId to string
            report["_id"] = str(report["_id"])
            
            # Convert all ObjectIds in the document to strings
            report = convert_objectid_to_str(report)
            
            # Ensure reportData has all expected fields with defaults
            report_data = report.get("reportData", {})
            
            if not isinstance(report_data, dict):
                report_data = {}
            
            # Ensure all required fields exist
            report["reportData"] = {
                "title": report_data.get("title", ""),
                "summary": report_data.get("summary", ""),
                "progress": report_data.get("progress", ""),
                "recommendations": report_data.get("recommendations", ""),
                "aiGenerated": report_data.get("aiGenerated", {}),
                "instructors": report_data.get("instructors", []),
                "graphData": report_data.get("graphData", {}),
                "instructorPerformance": report_data.get("instructorPerformance", [])
            }
            
            # Convert ObjectIds in instructors list
            if isinstance(report["reportData"]["instructors"], list):
                for instructor in report["reportData"]["instructors"]:
                    if isinstance(instructor, dict) and "_id" in instructor and isinstance(instructor["_id"], ObjectId):
                        instructor["_id"] = str(instructor["_id"])
            
            # Handle sessions array - convert all ObjectIds
            if "sessions" in report and isinstance(report["sessions"], list):
                for session in report["sessions"]:
                    if isinstance(session, dict):
                        # Convert _id if it exists
                        if "_id" in session and isinstance(session["_id"], ObjectId):
                            session["_id"] = str(session["_id"])
                        # Convert other ObjectId fields
                        if "instructorId" in session and isinstance(session["instructorId"], ObjectId):
                            session["instructorId"] = str(session["instructorId"])
                        if "therapistId" in session and isinstance(session["therapistId"], ObjectId):
                            session["therapistId"] = str(session["therapistId"])
                        if "patientId" in session and isinstance(session["patientId"], ObjectId):
                            session["patientId"] = str(session["patientId"])
            
            reports.append(report)

        return jsonify({"success": True, "reports": reports}), 200

    except Exception as e:
        print("Error fetching reports:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error: ' + str(e)}), 500


@app.route('/api/therapist/report/<report_id>', methods=['GET'])
def get_report(report_id):
    try:
        therapist_id = request.args.get('therapistId')

        if not therapist_id:
            return jsonify({'error': 'therapistId is required'}), 400

        # Validate ObjectId
        if not ObjectId.is_valid(report_id):
            return jsonify({"error": "Invalid report ID"}), 400

        report = reports_collection.find_one({
            "_id": ObjectId(report_id),
            "therapistId": therapist_id
        })

        if not report:
            return jsonify({"error": "Report not found"}), 404

        # Convert ObjectId to string
        report["_id"] = str(report["_id"])
        
        # Convert all ObjectIds in the document to strings
        report = convert_objectid_to_str(report)
        
        # Ensure reportData has all expected fields
        report_data = report.get("reportData", {})
        
        if not isinstance(report_data, dict):
            report_data = {}
        
        # Create complete reportData structure
        report["reportData"] = {
            "title": report_data.get("title", ""),
            "summary": report_data.get("summary", ""),
            "progress": report_data.get("progress", ""),
            "recommendations": report_data.get("recommendations", ""),
            "aiGenerated": report_data.get("aiGenerated", {}),
            "instructors": report_data.get("instructors", []),
            "graphData": report_data.get("graphData", {}),
            "instructorPerformance": report_data.get("instructorPerformance", [])
        }
        
        # Convert ObjectIds in instructors list
        if isinstance(report["reportData"]["instructors"], list):
            for instructor in report["reportData"]["instructors"]:
                if isinstance(instructor, dict) and "_id" in instructor and isinstance(instructor["_id"], ObjectId):
                    instructor["_id"] = str(instructor["_id"])
        
        # Handle sessions array
        if "sessions" in report and isinstance(report["sessions"], list):
            for session in report["sessions"]:
                if isinstance(session, dict):
                    # Convert ObjectId fields
                    if "_id" in session and isinstance(session["_id"], ObjectId):
                        session["_id"] = str(session["_id"])
                    if "instructorId" in session and isinstance(session["instructorId"], ObjectId):
                        session["instructorId"] = str(session["instructorId"])
                    if "therapistId" in session and isinstance(session["therapistId"], ObjectId):
                        session["therapistId"] = str(session["therapistId"])
                    if "patientId" in session and isinstance(session["patientId"], ObjectId):
                        session["patientId"] = str(session["patientId"])
                    
                    # Ensure nested objects exist
                    if not session.get("gameData"):
                        session["gameData"] = {}
                    if not session.get("analytics"):
                        session["analytics"] = {}

        return jsonify({"success": True, "report": report}), 200

    except Exception as e:
        print("Error fetching report:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Server error: ' + str(e)}), 500


# Helper function to convert ObjectIds to strings recursively
def convert_objectid_to_str(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_dict[key] = convert_objectid_to_str(value)
        return new_dict
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    else:
        return obj

@app.route('/api/therapist/delete-report/<report_id>', methods=['DELETE'])
def delete_report(report_id):
    try:
        result = reports_collection.delete_one({"_id": ObjectId(report_id)})

        if result.deleted_count == 0:
            return jsonify({"error": "Report not found"}), 404

        return jsonify({"success": True, "message": "Report deleted successfully"}), 200

    except Exception as e:
        return jsonify({"error": "Server error: " + str(e)}), 500

@app.route("/ppo_action", methods=["POST"])
def ppo_action():
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({"error": "State is required"}), 400

        state_list = data["state"]
        state = np.array(state_list, dtype=np.float32)

        fatigue = data.get("fatigue", 0.0)
        success = data.get("success", 0.0)
        engagement = data.get("engagement", 0.0)

        gamma = ppo_model.get_gamma() * (1 - fatigue) * engagement * success
        alpha = ppo_model.get_alpha() * (1 - fatigue) * engagement

        ppo_model.update_gamma(gamma)
        ppo_model.update_alpha(alpha)

        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)), mode='constant')
        elif len(state) > state_dim:
            state = state[:state_dim]

        action_index, log_prob = ppo_model.get_action(state)
        adjustments = ppo_model.process_actions(action_index)

        return jsonify({"action_index": action_index, "adjustments": adjustments, "log_prob": log_prob})

    except Exception as e:
        print(f"Error in ppo_action: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/ppo_train", methods=["POST"])
def ppo_train():
    try:
        data = request.get_json()
        if not data or "transitions" not in data:
            return jsonify({"error": "Transitions are required"}), 400

        transitions = data["transitions"]

        for transition in transitions:
            state = np.array(transition["state"], dtype=np.float32)
            action = int(transition["action"])  # discrete
            reward = transition["reward"]
            next_state = np.array(transition["next_state"], dtype=np.float32)
            done = transition.get("done", False)
            log_prob = transition.get("log_prob", 0.0)

            ppo_model.store_transition(state, action, reward, next_state, done, log_prob)

        ppo_model.update()
        save_model()
        return jsonify({"message": "Training completed successfully"})

    except Exception as e:
        print(f"Error in ppo_train: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
@app.route("/store_session_data", methods=["POST"])
def store_session_data():
    try:
        data = request.get_json()

        # Create folder if missing
        folder = "csv_logs"
        os.makedirs(folder, exist_ok=True)

        # File name = today's date
        file_path = os.path.join(folder, "session_log.csv")

        # If file missing → write header
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "time",
                    "state",
                    "leftHand_x","leftHand_y","leftHand_z",
                    "rightHand_x","rightHand_y","rightHand_z",
                    "leftShoulder_x","leftShoulder_y","leftShoulder_z",
                    "rightShoulder_x","rightShoulder_y","rightShoulder_z",
                    "hip_x","hip_y","hip_z",
                    "head_x","head_y","head_z",
                    "fatigue",
                    "engagement",
                    "success"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                data.get("time", 0),
                data.get("state", []),

                data["leftHand"]["x"], data["leftHand"]["y"], data["leftHand"]["z"],
                data["rightHand"]["x"], data["rightHand"]["y"], data["rightHand"]["z"],
                data["leftShoulder"]["x"], data["leftShoulder"]["y"], data["leftShoulder"]["z"],
                data["rightShoulder"]["x"], data["rightShoulder"]["y"], data["rightShoulder"]["z"],
                data["hip"]["x"], data["hip"]["y"], data["hip"]["z"],
                data["head"]["x"], data["head"]["y"], data["head"]["z"],

                data.get("fatigue", 0),
                data.get("engagement", 0),
                data.get("success", 0)
            ])

        return jsonify({"status": "saved"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/dqn_action", methods=["POST"])
def dqn_action():
    """
    Receive state from Unity and return DQN action
    Expected state format: [leftHand_active, leftLeg_active, rightLeg_active, rightHand_active,
                           light0_state, light1_state, light2_state, light3_state,
                           active_lanterns, score_normalized]
    """
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({"error": "State is required"}), 400

        state_list = data["state"]
        state = np.array(state_list, dtype=np.float32)

        # Get fatigue and engagement metrics
        fatigue = data.get("fatigue", 0.0)
        engagement = data.get("engagement", 0.0)
        
        # Adjust learning rate and gamma based on player state
        base_lr = dqn_model.get_lr()
        base_gamma = dqn_model.get_gamma()
        
        # Reduce learning during high fatigue, increase during high engagement
        adjusted_lr = base_lr * (1 - fatigue * 0.5) * (0.5 + engagement * 0.5)
        adjusted_gamma = base_gamma * (1 - fatigue * 0.2) * (0.8 + engagement * 0.2)
        
        dqn_model.update_lr(adjusted_lr)
        dqn_model.update_gamma(adjusted_gamma)

        # Ensure state dimension matches
        if len(state) < dqn_model.state_dim:
            state = np.pad(state, (0, dqn_model.state_dim - len(state)), mode='constant')
        elif len(state) > dqn_model.state_dim:
            state = state[:dqn_model.state_dim]

        # Get action from DQN
        training = data.get("training", True)
        action_index, q_value = dqn_model.get_action(state, training=training)
        
        # Convert action to adjustments
        adjustments = dqn_model.process_actions(action_index)

        return jsonify({
            "action_index": int(action_index),
            "adjustments": adjustments,
            "q_value": float(q_value),
            "epsilon": float(dqn_model.epsilon)
        })

    except Exception as e:
        print(f"Error in dqn_action: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/dqn_train", methods=["POST"])
def dqn_train():
    """
    Receive transitions and train DQN
    Expected format: list of transitions with state, action, reward, next_state, done
    """
    try:
        data = request.get_json()
        if not data or "transitions" not in data:
            return jsonify({"error": "Transitions are required"}), 400

        transitions = data["transitions"]

        # Store all transitions in replay buffer
        for transition in transitions:
            state = np.array(transition["state"], dtype=np.float32)
            action = int(transition["action"])
            reward = float(transition["reward"])
            next_state = np.array(transition["next_state"], dtype=np.float32)
            done = bool(transition.get("done", False))

            dqn_model.store_transition(state, action, reward, next_state, done)

        # Perform multiple updates
        num_updates = data.get("num_updates", 5)
        training_stats = []
        
        for _ in range(num_updates):
            stats = dqn_model.update()
            training_stats.append(stats)

        # Update target network periodically
        update_target = data.get("update_target", False)
        if update_target:
            dqn_model.update_target_network()
            print("🎯 Target network updated")

        # Save model
        save_model_dqn()

        # Average stats
        avg_loss = np.mean([s["loss"] for s in training_stats])
        avg_q = np.mean([s["q_value"] for s in training_stats])

        return jsonify({
            "message": "Training completed successfully",
            "avg_loss": float(avg_loss),
            "avg_q_value": float(avg_q),
            "epsilon": float(dqn_model.epsilon),
            "memory_size": len(dqn_model.memory)
        })

    except Exception as e:
        print(f"Error in dqn_train: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/store_rogl_session", methods=["POST"])
def store_rogl_session():
    """
    Store ROGL game session data to CSV
    """
    try:
        data = request.get_json()

        # Create folder if missing
        folder = "csv_logs"
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "rogl_session_log.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "time",
                    "state",
                    "leftHand_x", "leftHand_y", "leftHand_z", "leftHand_active",
                    "rightHand_x", "rightHand_y", "rightHand_z", "rightHand_active",
                    "leftLeg_x", "leftLeg_y", "leftLeg_z", "leftLeg_active",
                    "rightLeg_x", "rightLeg_y", "rightLeg_z", "rightLeg_active",
                    "light0_state", "light1_state", "light2_state", "light3_state",
                    "active_lanterns",
                    "score",
                    "fatigue",
                    "engagement"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                data.get("time", 0),
                data.get("state", []),
                
                data["leftHand"]["x"], data["leftHand"]["y"], data["leftHand"]["z"], data["leftHand"]["active"],
                data["rightHand"]["x"], data["rightHand"]["y"], data["rightHand"]["z"], data["rightHand"]["active"],
                data["leftLeg"]["x"], data["leftLeg"]["y"], data["leftLeg"]["z"], data["leftLeg"]["active"],
                data["rightLeg"]["x"], data["rightLeg"]["y"], data["rightLeg"]["z"], data["rightLeg"]["active"],
                
                data.get("light0_state", 0),
                data.get("light1_state", 0),
                data.get("light2_state", 0),
                data.get("light3_state", 0),
                data.get("active_lanterns", 0),
                data.get("score", 0),
                data.get("fatigue", 0),
                data.get("engagement", 0)
            ])

        return jsonify({"status": "saved"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/mc_action", methods=["POST"])
def mc_action():
    """
    Receive state from Unity and return PPO action
    Expected state format: [similarity_history (5 values), current_score, difficulty_level]
    """
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({"error": "State is required"}), 400

        state_list = data["state"]
        state = np.array(state_list, dtype=np.float32)

        # Get fatigue and engagement metrics
        fatigue = data.get("fatigue", 0.0)
        engagement = data.get("engagement", 0.0)
        success_rate = data.get("success", 0.5)
        
        # Adjust learning parameters based on player state
        base_lr = mc_ppo_model.get_alpha()
        base_gamma = mc_ppo_model.get_gamma()
        
        # Reduce learning during high fatigue, increase during high engagement
        adjusted_lr = base_lr * (1 - fatigue * 0.3) * (0.7 + engagement * 0.3)
        adjusted_gamma = base_gamma * (1 - fatigue * 0.1) * (0.9 + success_rate * 0.1)
        
        mc_ppo_model.update_alpha(adjusted_lr)
        mc_ppo_model.update_gamma(adjusted_gamma)

        # Ensure state dimension matches
        if len(state) < mc_ppo_model.policy_net.fc1.in_features:
            state = np.pad(state, (0, mc_ppo_model.policy_net.fc1.in_features - len(state)), mode='constant')
        elif len(state) > mc_ppo_model.policy_net.fc1.in_features:
            state = state[:mc_ppo_model.policy_net.fc1.in_features]

        # Get action from PPO
        action_array, log_prob = mc_ppo_model.get_action(state)
        
        # Convert action to adjustments
        adjustments = mc_ppo_model.process_actions(action_array)

        return jsonify({
            "action": action_array.tolist(),
            "adjustments": adjustments,
            "log_prob": float(log_prob)
        })

    except Exception as e:
        print(f"Error in mc_action: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/mc_train", methods=["POST"])
def mc_train():
    """
    Receive transitions and train MC PPO
    Expected format: list of transitions with state, action, reward, next_state, done, log_prob
    """
    try:
        data = request.get_json()
        if not data or "transitions" not in data:
            return jsonify({"error": "Transitions are required"}), 400

        transitions = data["transitions"]

        # Store all transitions in memory
        for transition in transitions:
            state = np.array(transition["state"], dtype=np.float32)
            action = np.array(transition["action"], dtype=np.float32)
            reward = float(transition["reward"])
            next_state = np.array(transition["next_state"], dtype=np.float32)
            done = bool(transition.get("done", False))
            log_prob = float(transition.get("log_prob", 0.0))

            mc_ppo_model.store_transition(state, action, reward, next_state, done, log_prob)

        # Train the model
        batch_size = data.get("batch_size", 64)
        epochs = data.get("epochs", 10)
        
        training_stats = mc_ppo_model.update(batch_size=batch_size, epochs=epochs)
        
        # Save model
        save_mc_model()

        return jsonify({
            "message": "MC PPO training completed successfully",
            "loss": float(training_stats["loss"]),
            "policy_loss": float(training_stats["policy_loss"]),
            "value_loss": float(training_stats["value_loss"]),
            "memory_size": len(mc_ppo_model.memory)
        })

    except Exception as e:
        print(f"Error in mc_train: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route("/store_mc_session", methods=["POST"])
def store_mc_session():
    """
    Store MC game session data to CSV
    """
    try:
        data = request.get_json()

        # Create folder if missing
        folder = "csv_logs"
        os.makedirs(folder, exist_ok=True)

        file_path = os.path.join(folder, "mc_session_log.csv")
        file_exists = os.path.isfile(file_path)

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "time",
                    "similarity_current",
                    "similarity_avg_5s",
                    "score",
                    "difficulty_level",
                    "upper_threshold",
                    "lower_threshold",
                    "gap_between_actions",
                    "current_animation",
                    "fatigue",
                    "engagement",
                    "success_rate",
                    "time_above_threshold",
                    "time_below_threshold"
                ])

            writer.writerow([
                datetime.now().isoformat(),
                data.get("time", 0),
                data.get("similarity_current", 0),
                data.get("similarity_avg_5s", 0),
                data.get("score", 0),
                data.get("difficulty_level", 0),
                data.get("upper_threshold", 80),
                data.get("lower_threshold", 70),
                data.get("gap_between_actions", 5),
                data.get("current_animation", ""),
                data.get("fatigue", 0),
                data.get("engagement", 0),
                data.get("success_rate", 0),
                data.get("time_above_threshold", 0),
                data.get("time_below_threshold", 0)
            ])

        return jsonify({"status": "saved"}), 200

    except Exception as e:
        print(f"Error in store_mc_session: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/usercodeunity', methods=['POST'])
def usercode_unity():
    try:
        if request.is_json:
            data = request.get_json()
            user_code = data.get('usercode') or data.get('userCode')
            game_name = data.get('gameName')  # optional now
        else:
            user_code = request.form.get('usercode') or request.form.get('userCode')
            game_name = request.form.get('gameName')  # optional now
            
        if not user_code:
            return jsonify({'error': 'usercode is required'}), 400

        user = users_collection.find_one({'userCode': user_code})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        game_configs = {}
        if user['userType'] == 'patient':
            patient_games = patient_games_collection.find_one({
                'patientId': user['_id']
            })
            
            if patient_games and 'game_configs' in patient_games:
                # ✅ return all game configs instead of single one
                game_configs = patient_games['game_configs']

        # ✅ default configs for games without stored configs
        default_config = {
            'difficulty': 'medium'
        }

        # Fetch all available games
        games = list(games_collection.find({'available': True}))
        
        if not games:
            games = [
                {'name': 'Memory Match', 'description': 'Improve memory skills', 'difficulty': 'easy'},
                {'name': 'Pattern Recognition', 'description': 'Enhance cognitive abilities', 'difficulty': 'medium'},
                {'name': 'Reaction Time', 'description': 'Develop faster responses', 'difficulty': 'hard'}
            ]

        # ✅ merge defaults for missing configs
        merged_configs = {}
        for game in games:
            gname = game.get('name').lower().replace(" ", "_")
            merged_configs[gname] = game_configs.get(gname, default_config)

        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'name': user['name'],
                'userCode': user['userCode'],
                'userType': user['userType']
            },
            'gameConfigs': merged_configs,   # ✅ send all configs
            'games': [to_json(game) for game in games]
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
@app.route("/therapistdefinedlimit/usercode", methods=["POST"])
def therapist_defined_limit():
    try:
        data = request.get_json() or request.form
        user_code = data.get("usercode")
        game_name = data.get("gameName", "bubble_game")

        if not user_code:
            return jsonify({"error": "usercode is required"}), 400

        user = users_collection.find_one({'userCode': user_code})
        if not user:
            return jsonify({"error": "User not found"}), 404

        if user['userType'] == 'patient':
            patient_games = patient_games_collection.find_one({
                'patientId': user['_id']
            })
            
            if patient_games:
                config_data = patient_games.get('game_configs', {}).get(game_name)
                if config_data and config_data.get('enabled', True):
                    return jsonify({
                        "success": True, 
                        "limits": config_data
                    }), 200

        default_limits = {
            "targetScore": 20,
            "maxBubbles": 10,
            "difficulty": "medium",
            "spawn_area": {"x_min": -5, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": -5, "z_max": 5}
        }

        return jsonify({"success": True, "limits": default_limits}), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        client.admin.command('ismaster')
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'database': 'disconnected', 'error': str(e)}), 500

def initialize_sample_games():
    if games_collection.count_documents({}) == 0:
        sample_games = [
            {
                'name': 'bubble_game',
                'display_name': 'Bubble Pop Game',
                'description': 'Pop bubbles to improve coordination and reaction time',
                'category': 'motor_skills',
                'configurable_fields': [
                    {'name': 'difficulty', 'type': 'select', 'options': ['easy', 'medium', 'hard'], 'default': 'medium'},
                    {'name': 'target_score', 'type': 'number', 'min': 5, 'max': 100, 'default': 20},
                    {'name': 'max_bubbles', 'type': 'number', 'min': 5, 'max': 50, 'default': 10},
                    {'name': 'spawn_area', 'type': 'object', 'fields': [
                        {'name': 'x_min', 'type': 'number', 'default': -5},
                        {'name': 'x_max', 'type': 'number', 'default': 5},
                        {'name': 'y_min', 'type': 'number', 'default': 1},
                        {'name': 'y_max', 'type': 'number', 'default': 5},
                        {'name': 'z_min', 'type': 'number', 'default': -5},
                        {'name': 'z_max', 'type': 'number', 'default': 5}
                    ]}
                ],
                'available': True,
                'createdAt': datetime.now(timezone.utc)
            },
            {
                'name': 'ghost_catcher',
                'display_name': 'Ghost Catcher',
                'description': 'Catch ghosts to enhance focus and agility',
                'category': 'motor_skills',
                'configurable_fields': [
                    {'name': 'difficulty', 'type': 'select', 'options': ['easy', 'medium', 'hard'], 'default': 'easy'},
                    {'name': 'target_score', 'type': 'number', 'min': 10, 'max': 30, 'default': 20},
                    {'name': 'spawn_count', 'type': 'number', 'min': 1, 'max': 15, 'default': 5},
                    {'name': 'time_delay', 'type': 'number', 'min': 3, 'max': 10, 'default': 5},
                    {'name': 'lights_green_prob', 'type': 'object', 'fields': [
                        {'name': 'lh', 'type': 'number', 'default': 0.4},
                        {'name': 'll', 'type': 'number', 'default': 0.6},
                        {'name': 'rl', 'type': 'number', 'default': 0.6},
                        {'name': 'rh', 'type': 'number', 'default': 0.4}
                    ]}
                ],
                'available': True,
                'createdAt': datetime.now(timezone.utc)
            },
            {
                'name': 'get_set_repeat',
                'display_name': 'Get Set Repeat',
                'description': 'Repeat the actions to boost coordination and attention',
                'category': 'coordination',
                'configurable_fields': [
                    {'name': 'difficulty', 'type': 'select', 'options': ['easy', 'medium', 'hard'], 'default': 'easy'},
                    {'name': 'target_score', 'type': 'number', 'min': 10, 'max': 30, 'default': 20},
                    {'name': 'action_time_delay', 'type': 'number', 'min': 3, 'max': 10, 'default': 5},
                    {'name': 'num_actions', 'type': 'number', 'min': 1, 'max': 15, 'default': 2},
                    {'name': 'similarity_min', 'type': 'number', 'min': 40, 'max': 80, 'default': 65},
                    {'name': 'similarity_max', 'type': 'number', 'min': 70, 'max': 100, 'default': 80}
                ],
                'available': True,
                'createdAt': datetime.now(timezone.utc)
            }
        ]
        games_collection.insert_many(sample_games)
        print("Sample games initialized")

# -----------------------
# Run the app
# -----------------------
if __name__ == '__main__':
    initialize_sample_games()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)