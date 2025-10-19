# Project Overview - Grievix Complaint Management System

## 🎯 Project Purpose
The Grievix system is a municipal complaint management platform that allows citizens to report issues, track progress, and engage with their local government. It provides transparency and accountability in public service delivery.

## 🛠 Technologies Used

### Frontend (React Application)
- *Framework*: React.js with functional components and hooks
- *Routing*: react-router-dom v7.4.1
- *State Management*: Built-in React useState and useEffect hooks
- *HTTP Client*: Axios for API communication
- *Styling*: CSS modules and plain CSS
- *Build Tool*: Create React App (react-scripts)

### Backend (Python Flask API)
- *Framework*: Flask
- *Database*: MongoDB with PyMongo
- *CORS*: flask-cors for cross-origin requests
- *Machine Learning*: scikit-learn for complaint categorization
- *File Handling*: werkzeug for file uploads
- *Environment*: Python 3.x

### Database
- *MongoDB*: NoSQL database for storing complaints, user data, and activity logs

### Development Tools
- *IDE*: VS Code or any code editor
- *Package Manager*: npm for frontend, pip for backend
- *Version Control*: Git

## 📁 Project Structure

project/
├── complaint/ (Frontend React App)
│   ├── public/
│   ├── src/
│   │   ├── AdminDashboard.jsx
│   │   ├── SubmitComplaint.jsx
│   │   ├── ViewComplaints.jsx
│   │   ├── login.jsx
│   │   ├── signup.jsx
│   │   ├── home.jsx
│   │   └── ... (other components)
│   ├── package.json
│   └── ...
├── rf.py (Backend Flask API)
├── uploads/ (Uploaded complaint photos)
├── random_forest_model_retrained.pkl (ML model)
├── tfidf_vectorizer_retrained.pkl (ML vectorizer)
└── label_encoder_retrained.pkl (ML label encoder)


## 🚀 Setup Guide

### Prerequisites
1. *Node.js* (v14 or higher) and npm
2. *Python* (3.7 or higher)
3. *MongoDB* (local installation or cloud instance)
4. *Git* (for version control)

### Installation Steps

#### 1. Backend Setup (Flask API)
bash
# Navigate to project root
cd /path/to/project/Final/Final

# Install Python dependencies
pip install flask flask-cors pymongo joblib

# Start MongoDB service (if using local MongoDB)
# On Windows:
net start MongoDB
# On macOS/Linux:
sudo systemctl start mongod

# Run the Flask backend
python rf.py


The backend will start on port 5000 by default.

#### 2. Frontend Setup (React App)
bash
# Navigate to frontend directory
cd /path/to/project/Final/Final/complaint

# Install npm dependencies
npm install

# Start the development server
npm start


The frontend will start on port 3000 by default.

### Environment Variables
Create a .env file in the project root if needed:
env
MONGODB_URI=mongodb://localhost:27017/


### Default Ports
- *Frontend*: http://localhost:3000
- *Backend API*: http://localhost:5000

## ▶ Running the Application

### Terminal Commands

#### Backend (Flask API)
bash
# Navigate to project directory
cd e:\miniproject\project\Final\Final

# Run the backend server
python rf.py


Expected output:

✅ MongoDB connected successfully!
 * Serving Flask app 'rf.py'
 * Debug mode: on
 * Running on http://127.0.0.1:5000


#### Frontend (React App)
bash
# Navigate to frontend directory
cd e:\miniproject\project\Final\Final\complaint

# Install dependencies (first time only)
npm install

# Run the frontend server
npm start


Expected output:

Compiled successfully!
Local: http://localhost:3000


## 📱 Key Features

### User Features
1. *Submit Complaints*: Citizens can file complaints with descriptions, photos, and locations
2. *View Complaints*: Browse all public complaints with filtering and sorting options
3. *Vote on Complaints*: Upvote important issues to increase their priority
4. *Comment on Complaints*: Engage in discussions about specific issues
5. *Track Progress*: Monitor the status of submitted complaints

### Admin Features
1. *Dashboard Analytics*: Overview of complaint statistics and trends
2. *Complaint Management*: View, update status, and assign complaints to departments
3. *Department Assignment*: Automatically or manually assign complaints to appropriate departments
4. *Admin Notes*: Add official updates and responses to complaints
5. *Priority Management*: Monitor high-priority issues requiring immediate attention

## 🔧 Key Components

### 1. SubmitComplaint.jsx
- Form for submitting new complaints
- Voice input capability
- Photo upload functionality
- Location detection
- Category prediction using ML

### 2. ViewComplaints.jsx
- Public dashboard for viewing complaints
- Filtering by category, status, and priority
- Voting and commenting system
- Complaint details modal view

### 3. AdminDashboard.jsx
- Administrative interface for managing complaints
- Analytics and reporting
- Status updates and department assignments
- Recent activity tracking

### 4. rf.py (Backend API)
- RESTful API endpoints for all frontend operations
- MongoDB data access and manipulation
- Machine learning integration for complaint categorization
- Activity logging and analytics

## 🎨 UI/UX Design

### Color Scheme
- *Primary*: Dark blue (#1a3a6d) for headers and important elements
- *Secondary*: Light blue (#4fc3f7) for accents and buttons
- *Background*: Light gray (#f8f9fa) for main content
- *Cards*: White (#ffffff) with subtle shadows
- *Status Colors*:
  - Success (Resolved): Green (#4caf50)
  - Warning (In Progress): Orange (#ff9800)
  - Info (New): Blue (#2196f3)

### Responsive Design
- Mobile-friendly layout with adaptive grids
- Flexible components that adjust to screen size
- Touch-friendly controls and buttons

## 🤖 Machine Learning Integration

The system uses a trained Random Forest model to automatically categorize complaints:
- *Categories*: Water Issues, Road Issues, Garbage Issues, Electricity, Drainage Issues
- *Features*: TF-IDF vectorized complaint text
- *Model Files*: 
  - random_forest_model_retrained.pkl
  - tfidf_vectorizer_retrained.pkl
  - label_encoder_retrained.pkl

## 🔒 Security Considerations

1. *CORS*: Properly configured to allow frontend-backend communication
2. *Data Validation*: Input validation on both frontend and backend
3. *File Uploads*: Size and type restrictions for complaint photos
4. *MongoDB*: Secure connection with proper error handling

## 📊 Database Collections

1. *complaints*: Stores all complaint data
2. *activity*: Tracks administrative actions and events
3. (Future) *users*: User account information (currently using localStorage)

## 🚨 Troubleshooting

### Common Issues

1. *Port Conflicts*:
   bash
   # Kill processes on port 3000 (frontend)
   taskkill /F /IM node.exe
   
   # Kill processes on port 5000 (backend)
   taskkill /F /IM python.exe
   

2. *MongoDB Connection Errors*:
   - Ensure MongoDB service is running
   - Check connection string in rf.py
   - Verify MongoDB installation

3. *Dependency Issues*:
   bash
   # Frontend
   cd complaint && npm install
   
   # Backend
   pip install -r requirements.txt
   

4. *ML Model Loading Errors*:
   - Ensure all .pkl files are in the project root
   - Check file permissions
   - Verify Python version compatibility

### Development Tips

1. *Hot Reloading*: Both frontend and backend support hot reloading during development
2. *Debug Mode*: Flask runs in debug mode for detailed error messages
3. *Console Logging*: Extensive logging in browser console and terminal
4. *Error Boundaries*: Graceful error handling throughout the application

This comprehensive system provides a complete solution for municipal complaint management with modern web technologies and a focus on user experience.
