from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from datetime import datetime, timedelta, date
import pickle
import numpy as np
import os
import json
import re
import pandas as pd

# Flask application setup
app = Flask(__name__)

# --- Configuration ---
app.config['SECRET_KEY'] = 'your-super-secret-key-CHANGE-THIS-IN-PRODUCTION'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/scholarship_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-jwt-secret-string-CHANGE-THIS-IN-PRODUCTION'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)

# --- Standardization Helper Functions (IDENTICAL to generate_new_scholarships.py) ---
# THESE FUNCTIONS MUST BE DEFINED HERE, BEFORE ANY ROUTE OR OTHER CODE THAT USES THEM
def standardize_string(s):
    if pd.isna(s) or s is None:
        return None
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def process_level_of_study(s):
    if pd.isna(s) or s is None:
        return None
    s = standardize_string(s)
    if 'phd' in s: return 'phd'
    if 'master' in s: return 'masters'
    if 'postgraduate' in s: return 'masters'
    if 'undergraduate' in s or 'bachelor' in s: return 'bachelors'
    if 'short course' in s: return 'short course'
    if 'fellowship' in s: return 'fellowship'
    if 'postdoctoral' in s: return 'postdoctoral'
    return s

def process_eligible_degrees(s):
    if pd.isna(s) or s is None:
        return None
    s = standardize_string(s)
    s_list = [item.strip() for item in s.split(',')]
    s_list = [item for item in s_list if item]
    if s_list:
        if 'business administration (mba)' in s_list[0]: return 'mba'
        if 'science' in s_list[0] and 'engineering' in s_list[0]: return 'science & engineering'
        return s_list[0]
    return None

def process_scholarship_coverage(s):
    if pd.isna(s) or s is None:
        return None
    s = standardize_string(s)
    if 'fully funded' in s or 'full tuition' in s or 'full funding' in s or '100% tuition' in s: return 'fully funded'
    if 'partial' in s or 'tuition fee discount' in s or '50% tuition' in s or '25% of first semester tuition' in s: return 'partially funded'
    if 'stipend' in s or 'living expenses' in s or 'airfare' in s: return 'stipend & living expenses'
    if 'one-off payment' in s: return 'one-time grant'
    return s

def process_country_name(s):
    if pd.isna(s) or s is None:
        return None
    s = standardize_string(s)
    if s == 'united kingdom (uk)': return 'united kingdom'
    if s == 'united states of america' or s == 'united states' or s == 'usa': return 'united states'
    return s

def process_ielts_requirement_str(s):
    if pd.isna(s) or s is None:
        return None
    s = standardize_string(s)
    if 'not mandatory' in s or 'not required' in s or 'no ielts required' in s: return 'not required'
    if 'yes' in s or 'required' in s: return 'required'
    match = re.search(r'(\d+(?:\.\d+)?)', s)
    if match:
        return f"required ({match.group(1)} ielts)"
    return s

# --- Database Models ---
class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'created_at': self.created_at.isoformat()
        }

class Scholarship(db.Model):
    __tablename__ = 'scholarships'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=True)
    description = db.Column(db.Text)
    level = db.Column(db.String(50), nullable=False) # Maps to level_of_study in CSV/JSON
    country = db.Column(db.String(100), nullable=False) # Maps to country/destination_country in CSV/JSON
    min_gpa = db.Column(db.Float)
    min_ielts = db.Column(db.Float)
    amount = db.Column(db.String(100)) # Maps to scholarship coverage in JSON
    deadline = db.Column(db.Date)
    requirements = db.Column(db.Text) # Maps to eligible_degrees in JSON
    
    scholarship_institution = db.Column(db.String(200))
    scholarship_coverage = db.Column(db.String(100))
    eligible_regions_nationalities = db.Column(db.String(255))
    category = db.Column(db.String(100))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'level': self.level,
            'country': self.country,
            'min_gpa': self.min_gpa,
            'min_ielts': self.min_ielts,
            'amount': self.amount,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'requirements': self.requirements,
            'scholarship_institution': self.scholarship_institution,
            'scholarship_coverage': self.scholarship_coverage,
            'eligible_regions_nationalities': self.eligible_regions_nationalities,
            'category': self.category,
            'created_at': self.created_at.isoformat()
        }

class UserProfile(db.Model):
    __tablename__ = 'user_profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    preferred_level = db.Column(db.String(50)) # Maps to level_of_study in ML model
    preferred_country = db.Column(db.String(100)) # Maps to country in ML model
    ielts_score = db.Column(db.Float)
    gpa = db.Column(db.Float)
    field_of_study = db.Column(db.String(100)) # Maps to eligible_degrees in ML model
    
    sop_quality = db.Column(db.String(50))
    research_experience = db.Column(db.Float)
    extracurricular_score = db.Column(db.Float)
    need = db.Column(db.Float)

    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    user = db.relationship('User', backref='profile')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'preferred_level': self.preferred_level,
            'preferred_country': self.preferred_country,
            'ielts_score': self.ielts_score,
            'gpa': self.gpa,
            'field_of_study': self.field_of_study,
            'sop_quality': self.sop_quality,
            'research_experience': self.research_experience,
            'extracurricular_score': self.extracurricular_score,
            'need': self.need,
            'updated_at': self.updated_at.isoformat()
        }

# --- ML Model and Scholarship JSON Data Loading ---

ML_MODEL_PICKLE = "new_try2.pkl"
# IMPORTANT: This now points to your newly generated 500 standardized JSON file
CLEANED_DATA_JSON = "data.json" 

model_pipeline = None
try:
    if os.path.exists(ML_MODEL_PICKLE):
        with open(ML_MODEL_PICKLE, "rb") as f:
            model_pipeline = pickle.load(f)
        print(f"✅ ML model pipeline loaded from {ML_MODEL_PICKLE}")
    else:
        print(f"❌ Error: ML model '{ML_MODEL_PICKLE}' not found. Recommendation system will not function.")
except Exception as e:
    print(f"❌ Error loading ML model: {e}. Recommendation system will not function.")

def get_scholarships_from_json():
    if not hasattr(g, '_scholarships_list_cache'):
        try:
            with open(CLEANED_DATA_JSON, "r", encoding='utf-8') as f:
                g._scholarships_list_cache = json.load(f)
            print(f"✅ Loaded {len(g._scholarships_list_cache)} scholarships from {CLEANED_DATA_JSON}")
        except FileNotFoundError:
            print(f"❌ Error: {CLEANED_DATA_JSON} not found. Please ensure it exists.")
            g._scholarships_list_cache = []
        except Exception as e:
            print(f"❌ Error loading scholarships from JSON: {e}")
            g._scholarships_list_cache = []
    return g._scholarships_list_cache

# Define the features the ML model expects for prediction
# THESE ARE THE COLUMNS FROM YOUR scholarship_historical_data_standardized.csv
MODEL_FEATURES = [
    'gpa', 'ielts_score', 'sop_quality', 'research_experience', 'extracurricular_score', 'need',
    'scholarship_institution', 'scholarship_coverage', 'eligible regions / nationalities', 'category',
    'country',               # Directly matches CSV column name
    'eligible_degrees',      # Directly matches CSV column name
    'level_of_study'         # Directly matches CSV column name
]

# --- General Helper functions ---
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_recommendations_fallback(user_profile_data, scholarships):
    recommendations = []
    
    for scholarship_obj in scholarships:
        score = 0
        
        # Level match (case-insensitive) - uses level_of_study for scholarship_obj
        if scholarship_obj.level and user_profile_data.get('preferred_level'):
            if scholarship_obj.level.lower() == user_profile_data['preferred_level'].lower():
                score += 40
        
        # Country match (case-insensitive) - uses country for scholarship_obj
        if scholarship_obj.country and user_profile_data.get('preferred_country'):
            if scholarship_obj.country.lower() == user_profile_data['preferred_country'].lower():
                score += 30
        
        # GPA requirement
        if scholarship_obj.min_gpa is not None and user_profile_data.get('gpa') is not None:
            if user_profile_data['gpa'] >= scholarship_obj.min_gpa:
                score += 20
            else:
                score -= 10
        
        # IELTS requirement
        if scholarship_obj.min_ielts is not None and user_profile_data.get('ielts_score') is not None:
            if user_profile_data['ielts_score'] >= scholarship_obj.min_ielts:
                score += 10
            else:
                score -= 5
        
        if score > 0:
            recommendations.append({
                'scholarship': scholarship_obj.to_dict(),
                'score': score
            })
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:5]


# --- API Routes ---
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not all(k in data for k in ('name', 'email', 'password')):
        return jsonify({'error': 'Name, email, and password are required'}), 400
    if not validate_email(data['email']):
        return jsonify({'error': 'Invalid email format'}), 400
    if len(data['password']) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    user = User(name=data['name'], email=data['email'])
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        access_token = create_access_token(identity=str(user.id))
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'access_token': access_token
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not all(k in data for k in ('email', 'password')):
        return jsonify({'error': 'Email and password are required'}), 400
    user = User.query.filter_by(email=data['email']).first()
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid email or password'}), 401
    access_token = create_access_token(identity=str(user.id))
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(),
        'access_token': access_token
    })

# --- Profile route without JWT (for testing) ---
@app.route('/api/profile', methods=['GET', 'POST'])
# @jwt_required() # TEMPORARILY COMMENTED OUT FOR DEVELOPMENT
def profile():
    if request.method == 'GET':
        user_id_param = request.args.get('user_id', type=int)
        if not user_id_param:
            return jsonify({'error': 'user_id query parameter is required for GET /api/profile without JWT'}), 400
        
        user = User.query.get(user_id_param)
        user_profile = UserProfile.query.filter_by(user_id=user_id_param).first()
        if user:
            response = {
                "id": user.id,
                "name": user.name,
                "email": user.email
            }
            if user_profile:
                response['profile'] = user_profile.to_dict()
            return jsonify(response)
        else:
            return jsonify({'message': 'User not found'}), 404
    
    elif request.method == 'POST':
        data = request.get_json()
        
        user_id_from_request = data.get('user_id', type=int)
        if not user_id_from_request:
            return jsonify({'error': 'user_id is required in the JSON body for POST /api/profile without JWT'}), 400
        
        user = User.query.get(user_id_from_request)
        if not user:
            return jsonify({'error': 'User not found with provided user_id'}), 404

        if data.get('gpa') is not None:
            try:
                gpa_val = float(data['gpa'])
                if not (0 <= gpa_val <= 4.0):
                    return jsonify({'error': 'GPA must be between 0 and 4.0'}), 400
                data['gpa'] = gpa_val
            except ValueError:
                return jsonify({'error': 'Invalid GPA format, must be a number.'}), 400

        if data.get('ielts_score') is not None:
            try:
                ielts_val = float(data['ielts_score'])
                if not (0 <= ielts_val <= 9.0):
                    return jsonify({'error': 'IELTS score must be between 0 and 9.0'}), 400
                data['ielts_score'] = ielts_val
            except ValueError:
                return jsonify({'error': 'Invalid IELTS score format, must be a number.'}), 400
        
        for num_field in ['research_experience', 'extracurricular_score', 'need']:
            if data.get(num_field) is not None:
                try:
                    data[num_field] = float(data[num_field])
                except ValueError:
                    return jsonify({'error': f'Invalid format for {num_field}, must be a number.'}), 400
        
        profile = UserProfile.query.filter_by(user_id=user_id_from_request).first()
        
        if profile:
            for key, value in data.items():
                if hasattr(UserProfile, key) and key not in ['id', 'user_id', 'updated_at']:
                    setattr(profile, key, value)
            profile.updated_at = datetime.utcnow()
        else:
            profile_data = {k: v for k, v in data.items() if hasattr(UserProfile, k) and k not in ['id', 'user_id', 'updated_at']}
            profile = UserProfile(user_id=user_id_from_request, **profile_data)
        
        try:
            db.session.add(profile)
            db.session.commit()
            return jsonify({
                'message': 'Profile updated successfully',
                'profile': profile.to_dict()
            })
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'Profile update failed: {str(e)}'}), 500

@app.route('/api/scholarships', methods=['GET'])
def get_scholarships():
    level = request.args.get('level')
    country = request.args.get('country')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    
    query = Scholarship.query
    
    if level:
        query = query.filter(Scholarship.level.ilike(f'%{level}%'))
    if country:
        query = query.filter(Scholarship.country.ilike(f'%{country}%'))
    
    scholarships = query.paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    return jsonify({
        'scholarships': [s.to_dict() for s in scholarships.items],
        'total': scholarships.total,
        'pages': scholarships.pages,
        'current_page': page
    })

@app.route('/api/scholarships/<int:scholarship_id>', methods=['GET'])
def get_scholarship(scholarship_id):
    scholarship = Scholarship.query.get_or_404(scholarship_id)
    return jsonify(scholarship.to_dict())

@app.route('/api/debug/scholarship-fields', methods=['GET'])
def debug_scholarship_fields():
    scholarships_list_from_json = get_scholarships_from_json()

    try:
        if not scholarships_list_from_json:
            return jsonify({"error": "No scholarships loaded from JSON"}), 500
        
        sample_fields = []
        for i, scholarship in enumerate(scholarships_list_from_json[:5]):
            sample_fields.append({
                "index": i,
                "name": scholarship.get('name', f'Scholarship {i+1}'),
                "fields": list(scholarship.keys())
            })
        
        all_fields = set()
        for scholarship in scholarships_list_from_json:
            all_fields.update(scholarship.keys())
        
        return jsonify({
            "total_scholarships_in_json": len(scholarships_list_from_json),
            "sample_json_scholarships": sample_fields,
            "all_unique_json_fields": sorted(list(all_fields)),
            "expected_fields_for_ml_model_mapping_from_json": [
                'scholarship_institution', 'scholarship coverage', 'eligible regions / nationalities', 'category',
                'destination_country', 'level_of_study', 'eligible_degrees', 'name'
            ]
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Error in debug_scholarship_fields: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/scholarships', methods=['POST'])
def add_scholarship():
    data = request.get_json()

    # --- Apply Standardization here (identical to how it's applied in get_scholarships_from_json) ---
    name = data.get('name')

    description_parts = []
    if data.get('country'): description_parts.append(f"Origin Country: {data['country']}")
    if data.get('scholarship_institution'): description_parts.append(f"Institution: {data['scholarship_institution']}")
    if data.get('level_of_study'): description_parts.append(f"Level: {data['level_of_study']}")
    if data.get('eligible_degrees'): description_parts.append(f"Degrees: {data['eligible_degrees']}")
    if data.get('category'): description_parts.append(f"Category: {data['category']}")
    description = " | ".join(description_parts) if description_parts else None

    level_processed = process_level_of_study(data.get('level_of_study'))
    country_db = process_country_name(data.get('destination_country'))

    min_gpa_db = None # Not directly in JSON
    
    min_ielts_db = None
    ielts_raw_db = data.get('ielts_requirement', '')
    match_ielts = re.search(r'(\d+(?:\.\d+)?)', ielts_raw_db)
    if match_ielts:
        try: min_ielts_db = float(match_ielts.group(1))
        except ValueError: pass

    amount_db = process_scholarship_coverage(data.get('scholarship coverage'))
    requirements_db = process_eligible_degrees(data.get('eligible_degrees'))

    deadline_db = None
    deadline_raw_db = data.get('application deadline', '').strip()
    for fmt_db in ['%Y-%m-%d', '%d-%b-%y', '%d-%b-%Y', '%B %d, %Y', '%b %d, %Y', '%Y-%m-%d %H:%M %p']:
        try:
            if 'PM' in deadline_raw_db or 'AM' in deadline_raw_db or ':' in deadline_raw_db:
                date_part_to_parse = deadline_raw_db.split('(')[0].strip()
                if re.match(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', date_part_to_parse, re.IGNORECASE):
                    parsed_dt_db = datetime.strptime(date_part_to_parse, '%B %d, %Y')
                    deadline_db = parsed_dt_db.date()
                    break
                else:
                    continue 
            else:
                deadline_db = datetime.strptime(deadline_raw_db, fmt_db).date()
                break
        except ValueError: continue
    
    if 'rolling' in deadline_raw_db.lower() or 'varies' in deadline_raw_db.lower():
        deadline_db = date(2099, 12, 31)
    elif deadline_raw_db and not deadline_db:
        print(f"Warning: Could not parse deadline '{deadline_raw_db}'. Setting to None.")

    scholarship_institution_db = standardize_string(data.get('scholarship_institution'))
    eligible_regions_nationalities_db = standardize_string(data.get('eligible regions / nationalities'))
    category_db = standardize_string(data.get('category'))

    scholarship = Scholarship(
        name=name,
        description=description,
        level=level_processed,
        country=country_db,
        min_gpa=min_gpa_db,
        min_ielts=min_ielts_db,
        amount=amount_db,
        deadline=deadline_db,
        requirements=requirements_db,
        scholarship_institution=scholarship_institution_db,
        scholarship_coverage=amount_db, # Re-using processed amount for scholarship_coverage in DB
        eligible_regions_nationalities=eligible_regions_nationalities_db,
        category=category_db
    )

    try:
        db.session.add(scholarship)
        db.session.commit()
        return jsonify({
            'message': 'Scholarship added successfully',
            'scholarship': scholarship.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        import traceback
        print(f"❌ Error adding scholarship: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to add scholarship', 'details': str(e)}), 500

# --- ML Recommendation Route (without JWT) ---
@app.route('/api/recommendations', methods=['POST'])
# @jwt_required() # TEMPORARILY COMMENTED OUT FOR DEVELOPMENT
def recommend_scholarships_ml():
    user_input_from_request = request.get_json()

    if not user_input_from_request:
        return jsonify({"error": "No user input provided in JSON format."}), 400

    required_user_fields = [
        'gpa', 'ielts_score', 'preferred_level', 'preferred_country', 'field_of_study',
        'sop_quality', 'research_experience', 'extracurricular_score', 'need'
    ]
    for field in required_user_fields:
        if field not in user_input_from_request or user_input_from_request[field] is None:
            return jsonify({"error": f"Missing or null required user profile field: '{field}' for ML recommendation."}), 400

    # --- Standardize user input before passing to ML model ---
    # Apply standardization functions to ensure consistency with training data
    user_input_for_ml = {
        'gpa': float(user_input_from_request.get('gpa')), # Ensure float
        'ielts_score': float(user_input_from_request.get('ielts_score')), # Ensure float
        'sop_quality': standardize_string(user_input_from_request.get('sop_quality')), # Standardize
        'research_experience': float(user_input_from_request.get('research_experience')), # Ensure float
        'extracurricular_score': float(user_input_from_request.get('extracurricular_score')), # Ensure float
        'need': float(user_input_from_request.get('need')), # Ensure float
        
        # Maps user's profile fields to the MODEL_FEATURES names
        'country': process_country_name(user_input_from_request.get('preferred_country')), # User's 'preferred_country' maps to 'country' for ML
        'eligible_degrees': process_eligible_degrees(user_input_from_request.get('field_of_study')), # User's 'field_of_study' maps to 'eligible_degrees' for ML
        'level_of_study': process_level_of_study(user_input_from_request.get('preferred_level')), # User's 'preferred_level' maps to 'level_of_study' for ML
    }

    if model_pipeline is None:
        print("ML model not loaded. Using fallback recommendation logic.")
        all_scholarships_db = Scholarship.query.all()
        fallback_recs = get_recommendations_fallback(user_input_for_ml, all_scholarships_db)
        return jsonify({
            "message": "ML model unavailable. Fallback recommendations provided.",
            "recommendations": [
                {"name": r['scholarship']['name'], "match_probability": r['score'], "id": r['scholarship']['id']}
                for r in fallback_recs
            ]
        })

    recommendations = []
    all_scholarships_json = get_scholarships_from_json() 

    for s_idx, s_json in enumerate(all_scholarships_json):
        try:
            combined_features_dict = {}

            # Add user input (already standardized)
            for feature in ['gpa', 'ielts_score', 'sop_quality', 'research_experience', 'extracurricular_score', 'need',
                            'country', 'eligible_degrees', 'level_of_study']: # Use MODEL_FEATURES names for user data
                combined_features_dict[feature] = user_input_for_ml.get(feature)

            # Add scholarship attributes from JSON data - APPLY STANDARDIZATION HERE
            # Note: JSON keys like 'scholarship coverage' are still used for .get(), then processed
            combined_features_dict['scholarship_institution'] = standardize_string(s_json.get('scholarship_institution'))
            combined_features_dict['scholarship_coverage'] = process_scholarship_coverage(s_json.get('scholarship coverage'))
            combined_features_dict['eligible regions / nationalities'] = standardize_string(s_json.get('eligible regions / nationalities'))
            combined_features_dict['category'] = standardize_string(s_json.get('category'))

            # Handle scholarship's own country/degrees/level from JSON
            # Overwrite user input if user's value was None, otherwise use user's preference
            if combined_features_dict['country'] is None:
                combined_features_dict['country'] = process_country_name(s_json.get('destination_country'))
            if combined_features_dict['eligible_degrees'] is None:
                combined_features_dict['eligible_degrees'] = process_eligible_degrees(s_json.get('eligible_degrees'))
            if combined_features_dict['level_of_study'] is None:
                combined_features_dict['level_of_study'] = process_level_of_study(s_json.get('level_of_study'))
            
            # Create DataFrame for Prediction, ensuring MODEL_FEATURES order
            input_df = pd.DataFrame([combined_features_dict], columns=MODEL_FEATURES)

            # --- DEBUG PRINTS START (Uncomment if you need to inspect input/transformed data) ---
            # current_scholarship_name = s_json.get('name', f"Scholarship {s_idx + 1}")
            # print(f"\n--- Processing: {current_scholarship_name} (Index: {s_idx}) ---")
            # print(f"Input DataFrame to pipeline:\n{input_df}")

            # preprocessor = model_pipeline.named_steps['preprocessor']
            # transformed_data = preprocessor.transform(input_df)
            # if hasattr(transformed_data, 'toarray'): transformed_data = transformed_data.toarray()
            # print(f"Transformed data (input to classifier):\n{transformed_data}")
            # --- DEBUG PRINTS END ---

            prob = model_pipeline.predict_proba(input_df)[0][1] 
            match_score = round(prob * 100, 2)

            # --- DEBUG PRINT FOR SCORE (Uncomment to see all scores) ---
            # print(f"Calculated match_score for {current_scholarship_name}: {match_score}%")
            # --- END DEBUG PRINT FOR SCORE ---

            if match_score >= 10.0: # Filter out very low matches (adjust threshold as needed)
                recommendations.append({
                    "name": s_json.get('name', f"Scholarship {s_idx + 1}"),
                    "match_probability": match_score,
                    "id": s_idx + 1 # Using loop index for ID
                })

        except Exception as e:
            print(f"❌ Error processing scholarship {s_json.get('name', s_json.get('id', 'N/A'))}: {e}")
            import traceback
            print(f"Full traceback for error: {traceback.format_exc()}")
            continue

    recommendations.sort(key=lambda x: x['match_probability'], reverse=True)
    top_recommendations = recommendations[:10]

    return jsonify({
        "message": f"Found {len(top_recommendations)} matching scholarships",
        "recommendations": top_recommendations
    })


# # --- Database Initialization (Development Mode: Drops and Recreates Tables) ---
# def create_tables_and_populate_samples():
#     print("\n--- Running create_tables_and_populate_samples() ---")
    
#     try:
#         print("Attempting to drop all existing database tables...")
#         db.drop_all()
#         print("✅ Successfully dropped all tables.")
        
#         print("Attempting to create all database tables based on models...")
#         db.create_all()
#         print("✅ Successfully created all tables.")
        
#         print("Adding sample scholarships to database from JSON...")
#         all_json_scholarships = get_scholarships_from_json() 
        
#         if not all_json_scholarships:
#             print("No scholarships loaded from JSON file. Skipping sample data population.")
#             return

#         scholarships_to_add = []
#         for scholarship_data in all_json_scholarships:
#             # --- Apply Standardization here for DB population (identical to how it's applied for ML input) ---
#             name = scholarship_data.get('name')

#             description_parts = []
#             if scholarship_data.get('country'): description_parts.append(f"Origin Country: {scholarship_data['country']}")
#             if scholarship_data.get('scholarship_institution'): description_parts.append(f"Institution: {scholarship_data['scholarship_institution']}")
#             if scholarship_data.get('level_of_study'): description_parts.append(f"Level: {scholarship_data['level_of_study']}")
#             if scholarship_data.get('eligible_degrees'): description_parts.append(f"Degrees: {scholarship_data['eligible_degrees']}")
#             if scholarship_data.get('category'): description_parts.append(f"Category: {data['category']}")
#             description = " | ".join(description_parts) if description_parts else None

#             level_processed = process_level_of_study(scholarship_data.get('level_of_study'))
#             country_db = process_country_name(scholarship_data.get('destination_country'))

#             min_gpa_db = None 
            
#             min_ielts_db = None
#             ielts_raw_db = scholarship_data.get('ielts_requirement', '')
#             match_ielts = re.search(r'(\d+(?:\.\d+)?)', ielts_raw_db)
#             if match_ielts:
#                 try: min_ielts_db = float(match_ielts.group(1))
#                 except ValueError: pass

#             amount_db = process_scholarship_coverage(scholarship_data.get('scholarship coverage'))
#             requirements_db = process_eligible_degrees(scholarship_data.get('eligible_degrees'))

#             deadline_db = None
#             deadline_raw_db = scholarship_data.get('application deadline', '').strip()
#             for fmt_db in ['%Y-%m-%d', '%d-%b-%y', '%d-%b-%Y', '%B %d, %Y', '%b %d, %Y', '%Y-%m-%d %H:%M %p']:
#                 try:
#                     if 'PM' in deadline_raw_db or 'AM' in deadline_raw_db or ':' in deadline_raw_db:
#                         date_part_to_parse = deadline_raw_db.split('(')[0].strip()
#                         if re.match(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', date_part_to_parse, re.IGNORECASE):
#                             parsed_dt_db = datetime.strptime(date_part_to_parse, '%B %d, %Y')
#                             deadline_db = parsed_dt_db.date()
#                             break
#                         else:
#                             continue 
#                     else:
#                         deadline_db = datetime.strptime(deadline_raw_db, fmt_db).date()
#                         break
#                 except ValueError: continue
            
#             if 'rolling' in deadline_raw_db.lower() or 'varies' in deadline_raw_db.lower():
#                 deadline_db = date(2099, 12, 31)
#             elif deadline_raw_db and not deadline_db:
#                 print(f"Warning: Could not parse deadline '{deadline_raw_db}'. Setting to None.")

#             scholarship_institution_db = standardize_string(scholarship_data.get('scholarship_institution'))
#             eligible_regions_nationalities_db = standardize_string(scholarship_data.get('eligible regions / nationalities'))
#             category_db = standardize_string(scholarship_data.get('category'))

#             scholarships_to_add.append(Scholarship(
#                 name=name,
#                 description=description,
#                 level=level_processed,
#                 country=country_db,
#                 min_gpa=min_gpa_db,
#                 min_ielts=min_ielts_db,
#                 amount=amount_db,
#                 deadline=deadline_db,
#                 requirements=requirements_db,
#                 scholarship_institution=scholarship_institution_db,
#                 scholarship_coverage=amount_db, # Re-using processed amount for scholarship_coverage in DB
#                 eligible_regions_nationalities=eligible_regions_nationalities_db,
#                 category=category_db
#             ))
        
#         try:
#             db.session.bulk_save_objects(scholarships_to_add)
#             db.session.commit()
#             print(f"✅ Added {len(scholarships_to_add)} sample scholarships to database from JSON.")
#         except Exception as e:
#             db.session.rollback()
#             print(f"❌ Error adding sample scholarships from JSON: {e}")
#             import traceback
#             print(f"Full traceback: {traceback.format_exc()}")
#             raise 

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': error.description}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    import traceback
    print(f"❌ Internal Server Error: {traceback.format_exc()}")
    return jsonify({'error': 'Internal server error'}), 500


# --- App Run ---
if __name__ == '__main__':
    ML_MODEL_PICKLE = "new_try2.pkl"
    
    model_pipeline = None # Initialize as None
    try:
        if os.path.exists(ML_MODEL_PICKLE):
            with open(ML_MODEL_PICKLE, "rb") as f:
                model_pipeline = pickle.load(f)
            print(f"✅ ML model pipeline loaded from {ML_MODEL_PICKLE}") # Corrected
        else:
            print(f"❌ Error: ML model '{ML_MODEL_PICKLE}' not found. Recommendation system will not function.") # Corrected
    except Exception as e:
        print(f"❌ Error loading ML model: {e}. Recommendation system will not function.")
        
    app.run(debug=True, host='0.0.0.0', port=5000)