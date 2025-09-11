# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import bcrypt
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import random
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Dict, Any
import json

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
game_limits_collection = db.game_limits

# Add these helper functions and classes
class GameConfig:
    """Game configuration class"""
    def __init__(self, game_name: str, difficulty: str, target_score: int, 
                 max_bubbles: int, spawn_area: dict, enabled: bool = True):
        self.game_name = game_name
        self.difficulty = difficulty
        self.target_score = target_score
        self.max_bubbles = max_bubbles
        self.spawn_area = spawn_area  # {x_min, x_max, y_min, y_max, z_min, z_max}
        self.enabled = enabled

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

# -------------------------
# Custom PPO Implementation (Python 3.12+ Compatible)
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  # Actions between -1 and 1
        return x

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomPPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.value_net = ValueNetwork(state_dim)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.memory = deque(maxlen=10000)
        
    def get_action(self, state):
        device = next(self.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.policy_net(state_tensor)
        return action.squeeze(0).cpu().numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size=64, epochs=10):
        if len(self.memory) < batch_size:
            return
        
        # Get device
        device = next(self.policy_net.parameters()).device
        
        # Convert memory to tensors with device
        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update policy and value networks
        for _ in range(epochs):
            # Policy loss
            current_actions = self.policy_net(states)
            policy_loss = -torch.mean(advantages * current_actions)
            
            # Value loss
            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), rewards + self.gamma * next_values * (1 - dones))
            
            # Update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        # Clear memory
        self.memory.clear()

# Initialize custom PPO model
state_dim = 11
action_dim = 7
ppo_model = CustomPPO(state_dim, action_dim)

# -----------------------
# Helper Functions
# -----------------------
def generate_user_code(name: str) -> str:
    """Generate a unique user code from name initials and random digits"""
    initials = ''.join(word[0].upper() for word in name.split() if word)
    if not initials:
        initials = "UC"  # Default if name is empty
    random_digits = ''.join(random.choices(string.digits, k=4))
    return f"{initials}{random_digits}"

def to_json(document: dict) -> dict:
    """Convert MongoDB ObjectId to string in a document"""
    if not document:
        return {}
    
    doc = document.copy()
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    
    # Convert other ObjectId fields
    for key, value in doc.items():
        if isinstance(value, ObjectId):
            doc[key] = str(value)
        elif isinstance(value, list):
            doc[key] = [str(item) if isinstance(item, ObjectId) else item for item in value]
    
    return doc

def is_valid_email(email: str) -> bool:
    """Basic email validation"""
    return '@' in email and '.' in email

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

        # Validate input
        if not all([user_type, name, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if user_type not in ['therapist', 'instructor']:
            return jsonify({'error': 'Invalid user type. Must be therapist or instructor'}), 400
        
        if not is_valid_email(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        # Check if user already exists
        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'User already exists with this email'}), 400

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Generate user code for therapists and instructors
        user_code = generate_user_code(name)
        
        # Check if user code is unique
        while users_collection.find_one({'userCode': user_code}):
            user_code = generate_user_code(name)  # Regenerate if not unique

        # Create user document
        user_data = {
            'name': name,
            'email': email,
            'password': hashed_password,
            'userType': user_type,
            'userCode': user_code,
            'createdAt': datetime.now(timezone.utc)
        }
        
        # Insert user
        result = users_collection.insert_one(user_data)
        user_id = result.inserted_id

        # Add to specific collection (therapist or instructor)
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
        
        # Check password
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

        # Find therapist by userId (not _id)
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

        # Validate therapist exists
        therapist = therapists_collection.find_one({'userId': ObjectId(therapist_id)})
        if not therapist:
            return jsonify({'error': 'Therapist not found'}), 404

        # Create patient user
        patient_user_data = {
            'name': patient_data.get('name'),
            'email': patient_data.get('email'),
            'age': patient_data.get('age'),
            'condition': patient_data.get('condition'),
            'userType': 'patient',
            'therapistId': ObjectId(therapist_id),
            'createdAt': datetime.now(timezone.utc)
        }
        
        # Generate unique user code for patient
        user_code = generate_user_code(patient_user_data['name'])
        while users_collection.find_one({'userCode': user_code}):
            user_code = generate_user_code(patient_user_data['name'])
        patient_user_data['userCode'] = user_code

        # Insert patient
        patient_result = users_collection.insert_one(patient_user_data)
        patient_id = patient_result.inserted_id

        # Add to therapist's patients
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

        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        # Validate rating
        if rating and (rating < 1 or rating > 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400

        session_data = {
            'therapistId': ObjectId(therapist_id),
            'patientId': ObjectId(patient_id),
            'instructorId': ObjectId(instructor_id),
            'gameData': game_data,
            'review': review,
            'rating': rating,
            'date': datetime.now(timezone.utc)
        }
        
        # Add instructor if provided
        if instructor_id:
            session_data['instructorId'] = ObjectId(instructor_id)

        result = sessions_collection.insert_one(session_data)
        session_id = result.inserted_id

        # Add to instructor's sessions if applicable
        if instructor_id:
            instructors_collection.update_one(
                {'userId': ObjectId(instructor_id)},
                {'$push': {'sessions': session_id}}
            )

        return jsonify({
            'message': 'Session created successfully',
            'sessionId': str(session_id)
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    try:
        user_id = request.args.get('userId')
        user_code = request.args.get('userCode')
        user_type = request.args.get('userType')

        if not user_type:
            return jsonify({'error': 'userType is required'}), 400

        # Build query based on available parameters
        query = {}
        
        if user_type == 'therapist':
            if user_id:
                query['therapistId'] = ObjectId(user_id)
            elif user_code:
                # Find therapist by userCode and get their userId
                therapist = therapists_collection.find_one({'userCode': user_code})
                if therapist:
                    query['therapistId'] = therapist['userId']
                else:
                    return jsonify({'error': 'Therapist not found with this user code'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required for therapist'}), 400
                
        elif user_type == 'instructor':
            if user_id:
                query['instructorId'] = ObjectId(user_id)
            elif user_code:
                # Find instructor by userCode and get their userId
                instructor = instructors_collection.find_one({'userCode': user_code})
                if instructor:
                    query['instructorId'] = instructor['userId']
                else:
                    return jsonify({'error': 'Instructor not found with this user code'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required for instructor'}), 400
                
        elif user_type == 'patient':
            if user_id:
                query['patientId'] = ObjectId(user_id)
            elif user_code:
                # Find patient by userCode and get their userId
                patient = users_collection.find_one({'userCode': user_code, 'userType': 'patient'})
                if patient:
                    query['patientId'] = patient['_id']
                else:
                    return jsonify({'error': 'Patient not found with this user code'}), 404
            else:
                return jsonify({'error': 'userId or userCode is required for patient'}), 400
        else:
            return jsonify({'error': 'Invalid userType'}), 400

        sessions = list(sessions_collection.find(query).sort('date', -1))
        return jsonify({'sessions': [to_json(s) for s in sessions]}), 200
        
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

@app.route('/api/user/<user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        return jsonify({'user': to_json(user)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    
@app.route('/api/therapist/patient-games', methods=['GET'])
def get_patient_games():
    """Get all games configured for a specific patient"""
    try:
        patient_id = request.args.get('patientId')
        therapist_id = request.args.get('therapistId')
        
        if not patient_id or not therapist_id:
            return jsonify({'error': 'patientId and therapistId are required'}), 400

        # Verify therapist has access to this patient
        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })
        
        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        # Get patient's game configurations
        patient_games = patient_games_collection.find_one({
            'patientId': ObjectId(patient_id),
            'therapistId': ObjectId(therapist_id)
        })

        if not patient_games:
            # Return default configurations if none exist
            default_configs = {
                'bubble_game': GameConfig(
                    game_name='bubble_game',
                    difficulty='medium',
                    target_score=20,
                    max_bubbles=10,
                    spawn_area={'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5},
                    enabled=True
                )
            }
            return jsonify({
                'games': {name: game_config_to_dict(config) for name, config in default_configs.items()}
            }), 200

        return jsonify({
            'games': patient_games.get('game_configs', {})
        }), 200

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/update-patient-games', methods=['POST'])
def update_patient_games():
    """Update game configurations for a patient"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        therapist_id = data.get('therapistId')
        patient_id = data.get('patientId')
        game_configs = data.get('gameConfigs', {})

        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        # Verify therapist has access to this patient
        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })
        
        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        # Validate game configurations
        validated_configs = {}
        for game_name, config in game_configs.items():
            if config.get('enabled', True):
                # Validate required fields for enabled games
                if not all(key in config for key in ['difficulty', 'target_score', 'max_bubbles', 'spawn_area']):
                    return jsonify({'error': f'Missing required fields for {game_name}'}), 400
                
                # Validate spawn area
                spawn_area = config['spawn_area']
                required_keys = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max']
                if not all(key in spawn_area for key in required_keys):
                    return jsonify({'error': f'Invalid spawn area for {game_name}'}), 400

            validated_configs[game_name] = config

        # Update or create patient game configurations
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
    """Get game configuration for a specific patient and game"""
    try:
        patient_id = request.args.get('patientId')
        game_name = request.args.get('gameName', 'bubble_game')
        
        if not patient_id:
            return jsonify({'error': 'patientId is required'}), 400

        # Get patient's game configurations
        patient_games = patient_games_collection.find_one({
            'patientId': ObjectId(patient_id)
        })

        if not patient_games:
            # Return default configuration if none exists
            default_config = GameConfig(
                game_name=game_name,
                difficulty='medium',
                target_score=20,
                max_bubbles=10,
                spawn_area={'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5},
                enabled=True
            )
            return jsonify({'config': game_config_to_dict(default_config)}), 200

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
    """Get list of all available games for therapists to assign"""
    try:
        games = list(games_collection.find({'available': True}))
        
        # Add default games if none exist
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

# PPO Model Endpoint - Updated for Unity compatibility
@app.route("/ppo_action", methods=["POST"])
def ppo_action():
    try:
        data = request.get_json()
        if not data or "state" not in data:
            return jsonify({"error": "State is required"}), 400
            
        # Get state from Unity request
        state_list = data["state"]
        
        # Convert to numpy array and ensure it has the correct shape
        state = np.array(state_list, dtype=np.float32)
        
        # If state doesn't have exactly 11 dimensions, pad or truncate
        if len(state) < state_dim:
            # Pad with zeros if state is shorter than expected
            state = np.pad(state, (0, state_dim - len(state)), mode='constant')
        elif len(state) > state_dim:
            # Truncate if state is longer than expected
            state = state[:state_dim]
        
        # Get action from PPO model
        action = ppo_model.get_action(state)
        
        # Return action as list for Unity
        return jsonify({"action": action.tolist()})
        
    except Exception as e:
        print(f"Error in ppo_action: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# PPO Training Endpoint
@app.route("/api/ppo_train", methods=["POST"])
def ppo_train():
    try:
        data = request.get_json()
        if not data or "transitions" not in data:
            return jsonify({"error": "Transitions are required"}), 400
            
        transitions = data["transitions"]
        
        # Store transitions in memory
        for transition in transitions:
            state = np.array(transition["state"], dtype=np.float32)
            action = np.array(transition["action"], dtype=np.float32)
            reward = transition["reward"]
            next_state = np.array(transition["next_state"], dtype=np.float32)
            done = transition["done"]
            
            ppo_model.store_transition(state, action, reward, next_state, done)
        
        # Update the model
        ppo_model.update()
        
        return jsonify({"message": "Training completed successfully"})
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Endpoint for Unity integration
@app.route('/usercodeunity', methods=['POST'])
def usercode_unity():
    try:
        # Support both form data and JSON
        if request.is_json:
            data = request.get_json()
            user_code = data.get('usercode') or data.get('userCode')
            game_name = data.get('gameName', 'bubble_game')
        else:
            user_code = request.form.get('usercode') or request.form.get('userCode')
            game_name = request.form.get('gameName', 'bubble_game')
            
        if not user_code:
            return jsonify({'error': 'usercode is required'}), 400

        user = users_collection.find_one({'userCode': user_code})
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Get game configuration for this patient
        game_config = None
        if user['userType'] == 'patient':
            patient_games = patient_games_collection.find_one({
                'patientId': user['_id']
            })
            
            if patient_games:
                config_data = patient_games.get('game_configs', {}).get(game_name)
                if config_data and config_data.get('enabled', True):
                    game_config = config_data

        # If no specific configuration, use defaults
        if not game_config:
            game_config = {
                'difficulty': 'medium',
                'target_score': 20,
                'max_bubbles': 10,
                'spawn_area': {'x_min': -5, 'x_max': 5, 'y_min': 1, 'y_max': 5, 'z_min': -5, 'z_max': 5}
            }

        # Get available games
        games = list(games_collection.find({'available': True}))
        
        # If no games in database, return some default games
        if not games:
            games = [
                {'name': 'Memory Match', 'description': 'Improve memory skills', 'difficulty': 'easy'},
                {'name': 'Pattern Recognition', 'description': 'Enhance cognitive abilities', 'difficulty': 'medium'},
                {'name': 'Reaction Time', 'description': 'Develop faster responses', 'difficulty': 'hard'}
            ]

        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'name': user['name'],
                'userCode': user['userCode'],
                'userType': user['userType']
            },
            'gameConfig': game_config,
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

        # Find user
        user = users_collection.find_one({'userCode': user_code})
        if not user:
            return jsonify({"error": "User not found"}), 404

        # Get game configuration for this patient
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

        # Return default limits if no specific configuration found
        default_limits = {
            "targetScore": 20,
            "maxBubbles": 10,
            "difficulty": "medium",
            "spawn_area": {"x_min": -5, "x_max": 5, "y_min": 1, "y_max": 5, "z_min": -5, "z_max": 5}
        }

        return jsonify({"success": True, "limits": default_limits}), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Test database connection
        client.admin.command('ismaster')
        return jsonify({'status': 'healthy', 'database': 'connected'}), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'database': 'disconnected', 'error': str(e)}), 500

# Initialize some sample games if collection is empty
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
                'name': 'memory_match',
                'display_name': 'Memory Match',
                'description': 'Match pairs of cards to improve memory',
                'category': 'cognitive',
                'configurable_fields': [
                    {'name': 'difficulty', 'type': 'select', 'options': ['easy', 'medium', 'hard'], 'default': 'easy'},
                    {'name': 'grid_size', 'type': 'select', 'options': ['4x4', '6x6', '8x8'], 'default': '4x4'},
                    {'name': 'time_limit', 'type': 'number', 'min': 30, 'max': 300, 'default': 120}
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
    # Initialize sample data
    initialize_sample_games()
    
    # Run the application with reloader disabled to avoid socket issues
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)