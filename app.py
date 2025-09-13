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
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
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
    
    def process_actions(self, raw_actions):
        processed = np.zeros_like(raw_actions)

        ranges = {
            0: (0, 5),    # spawnAreaSize
            1: (0, 10),    # bubbleSpeedAction
            2: (1, 10),    # bubbleLifetime
            3: (0, 20),    # spawnHeight
            4: (1, 20),    # numBubbles (int)
            5: (0.1, 5),   # bubbleSize # guidanceOn is handled separately
        }

        for i in range(6):  # first 6 are continuous
            low, high = ranges[i]
            processed[i] = low + (raw_actions[i] + 1) * 0.5 * (high - low)

        # integer rounding
        processed[4] = int(round(processed[4]))
        # boolean flag
        processed[6] = 1 if raw_actions[6] > 0 else 0

        return processed
        
    def get_action(self, state):
        device = next(self.policy_net.parameters()).device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_action = self.policy_net(state_tensor)
        raw_action = raw_action.squeeze(0).cpu().numpy()

        # 🔑 Scale into valid Unity values before returning
        return self.process_actions(raw_action)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self, batch_size=64, epochs=10):
        if len(self.memory) < batch_size:
            return
        
        device = next(self.policy_net.parameters()).device
        
        states, actions, rewards, next_states, dones = zip(*self.memory)
        
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(epochs):
            current_actions = self.policy_net(states)
            policy_loss = -torch.mean(torch.sum(current_actions * advantages.unsqueeze(1), dim=1))
            
            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), rewards + self.gamma * next_values * (1 - dones))
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        self.memory.clear()

# Initialize PPO model
state_dim = 11
action_dim = 7
ppo_model = CustomPPO(state_dim, action_dim)

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
            if config.get('enabled', True):
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
        report_data = data.get('reportData', {})
        session_ids = data.get('sessionIds', [])

        if not therapist_id or not patient_id:
            return jsonify({'error': 'therapistId and patientId are required'}), 400

        therapist = therapists_collection.find_one({
            'userId': ObjectId(therapist_id),
            'patients': ObjectId(patient_id)
        })
        
        if not therapist:
            return jsonify({'error': 'Patient not found or access denied'}), 404

        report = {
            'therapistId': ObjectId(therapist_id),
            'patientId': ObjectId(patient_id),
            'reportData': report_data,
            'sessionIds': [ObjectId(sid) for sid in session_ids],
            'createdAt': datetime.now(timezone.utc)
        }

        result = reports_collection.insert_one(report)
        report_id = result.inserted_id

        return jsonify({
            'message': 'Report created successfully',
            'reportId': str(report_id)
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/therapist/reports', methods=['GET'])
def get_reports():
    try:
        therapist_id = request.args.get('therapistId')
        patient_id = request.args.get('patientId')
        
        if not therapist_id:
            return jsonify({'error': 'therapistId is required'}), 400

        query = {'therapistId': ObjectId(therapist_id)}
        if patient_id:
            query['patientId'] = ObjectId(patient_id)

        reports = list(reports_collection.find(query).sort('createdAt', -1))
        return jsonify({'reports': [to_json(report) for report in reports]}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/report/<report_id>', methods=['GET'])
def get_report(report_id):
    try:
        report = reports_collection.find_one({'_id': ObjectId(report_id)})
        if not report:
            return jsonify({'error': 'Report not found'}), 404

        return jsonify({'report': to_json(report)}), 200
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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
        
        action = ppo_model.get_action(state)
        
        return jsonify({"action": action.tolist()})
        
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
            action = np.array(transition["action"], dtype=np.float32)
            reward = transition["reward"]
            next_state = np.array(transition["next_state"], dtype=np.float32)
            
            # 🔹 Use .get() with default False for done
            done = transition.get("done", False)
            
            ppo_model.store_transition(state, action, reward, next_state, done)
        
        ppo_model.update()
        
        return jsonify({"message": "Training completed successfully"})
    except Exception as e:
        print(f"Error in ppo_train: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

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
    initialize_sample_games()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)