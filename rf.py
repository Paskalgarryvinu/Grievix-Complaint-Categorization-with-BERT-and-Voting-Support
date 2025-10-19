from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
import joblib
import datetime
import os
import uuid
from typing import Dict, List, Optional
from werkzeug.utils import secure_filename
import json

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with Flask

# Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DB_NAME = "municipal_complaints"
COLLECTION_NAME = "complaints"
MODEL_VERSION = "1.2.0"

# Configure upload folder for complaint photos
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Connect to MongoDB
try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    complaints_collection = db[COLLECTION_NAME]
    activity_collection = db["activity"]  # Add activity collection
    print("\u2705 MongoDB connected successfully!")
except Exception as e:
    print(f"\u274c MongoDB connection error: {e}")
    db = None

# Define paths for ML components
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model_retrained.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer_retrained.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_retrained.pkl")

# Load ML components
try:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH) if os.path.exists(VECTORIZER_PATH) else None
    label_encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

    if all([model, tfidf_vectorizer, label_encoder]):
        print(f"\u2705 All ML components loaded successfully! (v{MODEL_VERSION})")
    else:
        print("\u26A0 Warning: Some ML components are missing!")
except Exception as e:
    print(f"\u274c Error loading ML components: {e}")
    model, tfidf_vectorizer, label_encoder = None, None, None

# Define your categories and keywords
CATEGORIES = [
    "Water Issues",
    "Road Issues",
    "Garbage Issues",
    "Electricity",
    "Drainage Issues",
    "Other"
]

# Enhanced category mapping with more keywords
CATEGORY_KEYWORDS = {
    "Water Issues": ["water", "drinking", "supply", "leak", "pipe", "tap", "smell", "taste", "pressure"],
    "Road Issues": ["road", "pothole", "asphalt", "street", "highway", "repair", "damage", "construction"],
    "Garbage Issues": ["garbage", "trash", "waste", "collection", "dump", "bin", "clean", "disposal"],
    "Electricity": ["electricity", "power", "outage", "blackout", "wire", "transformer", "voltage", "flickering"],
    "Drainage Issues": ["drainage", "sewer", "flood", "waterlogging", "blockage", "clog", "overflow"],
    "Other": ["noise", "loudspeaker", "park", "tree", "animal", "stray", "public", "nuisance"]
}

def manual_category_detection(complaint_text: str) -> Optional[str]:
    """Check if complaint should be manually categorized based on keywords"""
    complaint_text = complaint_text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.lower() in complaint_text for keyword in keywords):
            print(f"🔧 Manual keyword match: {category}")
            return category
    return None

def validate_prediction(predicted_category: str, complaint_text: str) -> str:
    """Ensure predicted category makes sense for the complaint"""
    complaint_text = complaint_text.lower()
    
    # If prediction is not in our defined categories, default to Other
    if predicted_category not in CATEGORIES:
        return "Other"
    
    return predicted_category

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "active",
        "model_version": MODEL_VERSION,
        "endpoints": [
            "/submit_complaint", 
            "/get_categories", 
            "/get_complaints",
            "/update_status",
            "/assign_department"
        ],
        "categories": CATEGORIES
    })

@app.route("/predict_category", methods=["POST"])
def predict_category():
    """Endpoint to predict category without saving the complaint"""
    try:
        data = request.json
        complaint_text = ""
        if data is not None:
            complaint_text = data.get("complaint", "").strip().lower()
        else:
            complaint_text = ""

        if len(complaint_text) < 10:
            return jsonify({"error": "Complaint must be at least 10 characters"}), 400

        # First try manual categorization
        manual_category = manual_category_detection(complaint_text)
        
        if manual_category:
            predicted_category = manual_category
        elif model and tfidf_vectorizer and label_encoder:
            # Fall back to ML model if manual detection fails
            complaint_tfidf = tfidf_vectorizer.transform([complaint_text])
            predicted_category_num = model.predict(complaint_tfidf)[0]
            predicted_category = label_encoder.inverse_transform([predicted_category_num])[0]
            predicted_category = validate_prediction(predicted_category, complaint_text)
        else:
            # If no model available, use manual detection or default to Other
            predicted_category = manual_category_detection(complaint_text) or "Other"

        # Final validation
        if predicted_category not in CATEGORIES:
            predicted_category = "Other"

        return jsonify({
            "category": predicted_category,
            "confidence": 0.85,  # Mock confidence score
            "auto_corrected": bool(manual_category)
        })

    except Exception as e:
        print(f"❌ Error in predict_category: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/submit_complaint", methods=["POST"])
def submit_complaint():
    try:
        print("Received complaint submission request")
        
        # Handle both JSON and form data
        if request.is_json:
            data = request.json
            complaint_text = ""
            location = ""
            has_photo = False
            submitted_by = "Anonymous"
            
            if data is not None:
                complaint_text = data.get("complaint", "").strip()
                location = data.get("location", "")
                has_photo = data.get("hasPhoto", False)
                submitted_by = data.get("submitted_by", "Anonymous")
        else:
            complaint_text = request.form.get("complaint", "").strip()
            location = request.form.get("location", "Not specified")
            has_photo = 'photo' in request.files and bool(request.files['photo'].filename)
            submitted_by = request.form.get("submitted_by", "Anonymous")

        print(f"Complaint text: {complaint_text}")
        print(f"Location: {location}")
        
        if len(complaint_text) < 10:
            return jsonify({"success": False, "message": "Complaint must be at least 10 characters"}), 400

        # First try manual categorization
        manual_category = manual_category_detection(complaint_text.lower())
        
        if manual_category:
            predicted_category = manual_category
        elif model and tfidf_vectorizer and label_encoder:
            # Fall back to ML model if manual detection fails
            complaint_tfidf = tfidf_vectorizer.transform([complaint_text.lower()])
            predicted_category_num = model.predict(complaint_tfidf)[0]
            predicted_category = label_encoder.inverse_transform([predicted_category_num])[0]
            predicted_category = validate_prediction(predicted_category, complaint_text.lower())
        else:
            # If no model available, use manual detection or default to Other
            predicted_category = manual_category_detection(complaint_text.lower()) or "Other"

        print(f"Predicted category: {predicted_category}")
        
        # Final validation
        if predicted_category not in CATEGORIES:
            predicted_category = "Other"

        # Generate unique ID for the complaint
        complaint_id = str(uuid.uuid4())
        
        # Get additional fields from form data
        severity = request.form.get("severity", 5)
        try:
            severity = int(severity)
        except (ValueError, TypeError):
            severity = 5
            
        tags_json = request.form.get("tags", "[]")
        try:
            tags = json.loads(tags_json) if isinstance(tags_json, str) else []
        except (json.JSONDecodeError, TypeError):
            tags = []
            
        anonymous = request.form.get("anonymous", "false").lower() == "true"
        
        # Calculate priority score based on severity and keywords
        priority_score = severity  # Start with severity rating
        if "urgent" in complaint_text.lower() or "emergency" in complaint_text.lower():
            priority_score = min(10, priority_score + 2)  # Boost by 2, max 10
        elif "soon" in complaint_text.lower() or "important" in complaint_text.lower():
            priority_score = min(10, priority_score + 1)  # Boost by 1, max 10
            
        # Handle photo upload if present
        photo_filename = None
        if has_photo and 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename:
                try:
                    photo_filename = secure_filename(f"{complaint_id}_{photo.filename}")
                    photo_path = os.path.join(app.config['UPLOAD_FOLDER'], photo_filename)
                    photo.save(photo_path)
                    print(f"Photo saved at: {photo_path}")
                except Exception as e:
                    print(f"Error saving photo: {e}")
                    photo_filename = None
            
        # Store complaint with metadata
        complaint_entry = {
            "_id": complaint_id,
            "complaint": complaint_text,
            "category": predicted_category,
            "location": location,
            "has_photo": bool(photo_filename),
            "photo_path": photo_filename,
            "timestamp": datetime.datetime.utcnow(),
            "prediction_source": "manual" if manual_category else "model",
            "model_version": MODEL_VERSION,
            "status": "new",
            "priority_score": priority_score,
            "severity": severity,
            "tags": tags,
            "anonymous": anonymous,
            "votes": 0,
            "comments": [],
            "submitted_by": submitted_by
        }
        
        print(f"Inserting complaint with ID: {complaint_id}")
        complaints_collection.insert_one(complaint_entry)
        
        # Log activity
        log_activity("new_complaint", f"New complaint submitted in {predicted_category} category")
        
        return jsonify({
            "success": True,
            "complaint_id": complaint_id,
            "category": predicted_category,
            "message": "Complaint submitted successfully!"
        })
    except Exception as e:
        print(f"❌ Error in submit_complaint: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/update_status", methods=["POST"])
def update_status():
    """Update the status of a complaint"""
    try:
        data = request.json
        complaint_id = None
        new_status = None
        
        if data is not None:
            complaint_id = data.get("complaintId")
            new_status = data.get("status")
        
        if not complaint_id or not new_status:
            return jsonify({"error": "Missing required fields"}), 400
            
        if new_status not in ["new", "in_progress", "resolved"]:
            return jsonify({"error": "Invalid status value"}), 400
            
        result = complaints_collection.update_one(
            {"_id": complaint_id},
            {"$set": {"status": new_status}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Complaint not found"}), 404
            
        # Log activity
        status_text = "New" if new_status == "new" else "In Progress" if new_status == "in_progress" else "Resolved"
        log_activity("status_update", f"Complaint #{complaint_id[:8]} marked as {status_text}")
            
        return jsonify({"success": True})
    except Exception as e:
        print(f"❌ Error in update_status: {e}")
        return jsonify({"error": "Internal server error"}), 500
        
@app.route("/assign_department", methods=["POST"])
def assign_department():
    """Assign a complaint to a department"""
    try:
        data = request.json
        complaint_id = None
        department = None
        
        if data is not None:
            complaint_id = data.get("complaintId")
            department = data.get("department")
        
        if not complaint_id or not department:
            return jsonify({"error": "Missing required fields"}), 400
            
        result = complaints_collection.update_one(
            {"_id": complaint_id},
            {"$set": {"assigned_department": department}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Complaint not found"}), 404
            
        return jsonify({"success": True})
    except Exception as e:
        print(f"❌ Error in assign_department: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/save_admin_note", methods=["POST"])
def save_admin_note():
    """Save an admin note for a complaint"""
    try:
        data = request.json
        complaint_id = None
        note_text = None
        
        if data is not None:
            complaint_id = data.get("complaintId")
            note_text = data.get("noteText")
        
        if not complaint_id or not note_text:
            return jsonify({"error": "Missing required fields"}), 400
            
        # Create admin note object
        admin_note = {
            "text": note_text,
            "timestamp": datetime.datetime.utcnow(),
            "admin": "Administrator"  # In a real app, this would be the logged-in admin
        }
        
        # Add note to the complaint
        result = complaints_collection.update_one(
            {"_id": complaint_id},
            {"$push": {"admin_notes": admin_note}}
        )
        
        if result.matched_count == 0:
            return jsonify({"error": "Complaint not found"}), 404
            
        return jsonify({"success": True, "message": "Admin note saved successfully"})
    except Exception as e:
        print(f"❌ Error in save_admin_note: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/upload_photo/<complaint_id>", methods=["POST"])
def upload_photo(complaint_id):
    """Handle photo upload for a specific complaint"""
    try:
        if 'photo' not in request.files:
            return jsonify({"error": "No photo part in the request"}), 400
            
        photo = request.files['photo']
        if photo.filename == '':
            return jsonify({"error": "No photo selected"}), 400
            
        # Check if complaint exists
        complaint = complaints_collection.find_one({"_id": complaint_id})
        if not complaint:
            return jsonify({"error": "Complaint not found"}), 404
            
        # Save the photo
        filename = f"{complaint_id}.jpg"
        photo_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        photo.save(photo_path)
        
        # Update complaint record with photo path
        complaints_collection.update_one(
            {"_id": complaint_id},
            {"$set": {"has_photo": True, "photo_path": filename}}
        )
        
        return jsonify({
            "message": "Photo uploaded successfully",
            "photo_path": filename
        })
        
    except Exception as e:
        print(f"❌ Error in upload_photo: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/photos/<filename>", methods=["GET"])
def get_photo(filename):
    """Serve uploaded photos"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/get_categories", methods=["GET"])
def get_categories():
    try:
        return jsonify(CATEGORIES)
    except Exception as e:
        print(f"\u274c Error in get_categories: {e}")
        return jsonify({"error": str(e)}), 500
        
@app.route('/get_analytics', methods=['GET'])
def get_analytics():
    try:
        # Get total complaints count
        total_complaints = complaints_collection.count_documents({})
        
        # Get counts by category
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        category_results = list(complaints_collection.aggregate(pipeline))
        category_counts = {item['_id']: item['count'] for item in category_results}
        
        # Get counts by status
        resolved_count = complaints_collection.count_documents({"status": "resolved"})
        pending_count = complaints_collection.count_documents({"status": {"$ne": "resolved"}})
        
        return jsonify({
            'total_complaints': total_complaints,
            'category_counts': category_counts,
            'resolved_count': resolved_count,
            'pending_count': pending_count
        })
    except Exception as e:
        print(f"Error in get_analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/get_complaints", methods=["GET"])
def get_complaints():
    """Get complaints with optional filtering and sorting"""
    try:
        # Get query parameters
        category = request.args.get("category")
        status = request.args.get("status")
        sort_by = request.args.get("sort", "newest")
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 10))
        search = request.args.get("search", "")
        submitted_by = request.args.get("submitted_by")
        
        # Build query
        query = {}
        if category and category != "All":
            query["category"] = category
        if status and status != "All":
            # Handle multiple statuses (comma-separated)
            if "," in status:
                statuses = status.split(",")
                query["status"] = {"$in": statuses}
            else:
                query["status"] = status
        if search:
            query["$text"] = {"$search": search}
        if submitted_by:
            query["submitted_by"] = submitted_by
            
        # Determine sort order
        if sort_by == "newest":
            sort_order = [("timestamp", -1)]
        elif sort_by == "oldest":
            sort_order = [("timestamp", 1)]
        elif sort_by == "highest_priority":
            sort_order = [("priority_score", -1), ("votes", -1), ("timestamp", -1)]
        elif sort_by == "most_votes":
            sort_order = [("votes", -1), ("timestamp", -1)]
        else:
            sort_order = [("timestamp", -1)]
            
        # Calculate pagination
        skip = (page - 1) * per_page
        
        # Create text index if it doesn't exist
        try:
            complaints_collection.create_index([("complaint_text", "text")])
        except Exception as e:
            print(f"Warning: Could not create text index: {e}")
        
        # Execute query
        total_count = complaints_collection.count_documents(query)
        complaints = list(complaints_collection.find(
            query, 
            sort=sort_order,
            skip=skip,
            limit=per_page
        ))
        
        # Convert ObjectId to string for JSON serialization
        for complaint in complaints:
            if "_id" in complaint:
                complaint["_id"] = str(complaint["_id"])
            if "timestamp" in complaint:
                complaint["timestamp"] = complaint["timestamp"].isoformat()
                
        return jsonify({
            "complaints": complaints,
            "total": total_count,
            "page": page,
            "per_page": per_page,
            "total_pages": (total_count + per_page - 1) // per_page
        })
    except Exception as e:
        print(f"❌ Error in get_complaints: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/vote_complaint', methods=['POST'])
def vote_complaint():
    try:
        data = request.json
        complaint_id = None
        vote_type = 'upvote'
        user_email = None
        
        if data is not None:
            complaint_id = data.get('complaintId')
            vote_type = data.get('voteType', 'upvote')
            user_email = data.get('userEmail')  # Get the user's email
        
        if not complaint_id:
            return jsonify({'error': 'Complaint ID is required'}), 400
            
        # Check if the user has already voted on this complaint
        complaint = complaints_collection.find_one({'_id': complaint_id})
        if not complaint:
            return jsonify({'error': 'Complaint not found'}), 404
            
        # Check if user has already voted
        if user_email:
            voters = complaint.get('voters', [])
            if user_email in voters:
                return jsonify({'error': 'You have already voted on this complaint'}), 400
        
        # Update vote count
        vote_change = 1 if vote_type == 'upvote' else -1
        
        # Prepare update operations
        update_ops = {
            '$inc': {'votes': vote_change, 'priority_score': vote_change * 0.5}
        }
        
        # Add user to voters list if email is provided
        if user_email:
            update_ops['$addToSet'] = {'voters': user_email}
        
        # Update the complaint with the new vote count and voter tracking
        result = complaints_collection.update_one(
            {'_id': complaint_id},
            update_ops
        )
        
        if result.modified_count == 0:
            return jsonify({'error': 'Complaint not found or vote not recorded'}), 404
            
        # Get the updated complaint
        updated_complaint = complaints_collection.find_one({'_id': complaint_id})
        votes_count = 0
        priority_score = 5
        
        if updated_complaint is not None:
            votes_count = updated_complaint.get('votes', 0)
            priority_score = updated_complaint.get('priority_score', 5)
            
        return jsonify({
            'success': True,
            'message': f'Vote {"added" if vote_type == "upvote" else "removed"}',
            'votes': votes_count,
            'priority_score': priority_score
        })
    except Exception as e:
        print(f"❌ Error in vote_complaint: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/add_comment', methods=['POST'])
def add_comment():
    try:
        data = request.json
        complaint_id = None
        comment_text = None
        
        if data is not None:
            complaint_id = data.get('complaintId')
            comment_text = data.get('comment')
        
        if not complaint_id or not comment_text:
            return jsonify({'error': 'Complaint ID and comment text are required'}), 400
            
        # Create comment object
        comment = {
            'text': comment_text,
            'timestamp': datetime.datetime.utcnow(),
            'user': 'Anonymous User',  # In a real app, this would be the logged-in user
            'comment_id': str(uuid.uuid4())  # Generate a unique ID for the comment
        }
        
        # Add comment to the complaint
        result = complaints_collection.update_one(
            {'_id': complaint_id},
            {'$push': {'comments': comment}}
        )
        
        if result.modified_count == 0:
            return jsonify({'error': 'Complaint not found or comment not added'}), 404
            
        return jsonify({
            'success': True,
            'message': 'Comment added successfully',
            'comment': comment
        })
    except Exception as e:
        print(f"❌ Error in add_comment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/get_recent_activity", methods=["GET"])
def get_recent_activity():
    """Get recent activity logs"""
    try:
        # Get the 10 most recent activities
        activities = list(activity_collection.find().sort("timestamp", -1).limit(10))
        
        # Convert ObjectId to string and format timestamp
        for activity in activities:
            if "_id" in activity:
                activity["_id"] = str(activity["_id"])
            if "timestamp" in activity:
                # Calculate time ago
                now = datetime.datetime.utcnow()
                diff = now - activity["timestamp"]
                
                if diff.total_seconds() < 60:
                    activity["time_ago"] = "just now"
                elif diff.total_seconds() < 3600:
                    minutes = int(diff.total_seconds() / 60)
                    activity["time_ago"] = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                elif diff.total_seconds() < 86400:
                    hours = int(diff.total_seconds() / 3600)
                    activity["time_ago"] = f"{hours} hour{'s' if hours > 1 else ''} ago"
                else:
                    days = int(diff.total_seconds() / 86400)
                    activity["time_ago"] = f"{days} day{'s' if days > 1 else ''} ago"
                
                activity["timestamp"] = activity["timestamp"].isoformat()
                
        return jsonify(activities)
    except Exception as e:
        print(f"Error in get_recent_activity: {e}")
        return jsonify({"error": str(e)}), 500

def log_activity(activity_type, message):
    """Log an activity to the activity collection"""
    try:
        activity = {
            "type": activity_type,
            "message": message,
            "timestamp": datetime.datetime.utcnow()
        }
        activity_collection.insert_one(activity)
    except Exception as e:
        print(f"Error logging activity: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)



