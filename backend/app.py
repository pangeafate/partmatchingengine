import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from chat_service import ChatService

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Initialize chat service
chat_service = ChatService()

# Store chat histories for different sessions
chat_histories = {}

# Serve the frontend static files
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve the frontend static files."""
    if path == "":
        # Serve index.html for the root path
        return send_from_directory(app.static_folder, 'index.html')
    
    # Try to serve the requested file
    try:
        return send_from_directory(app.static_folder, path)
    except:
        # If the file doesn't exist, serve index.html for client-side routing
        return send_from_directory(app.static_folder, 'index.html')

# API endpoint that returns API information
@app.route('/api', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        "status": "online",
        "service": "Part Matching Engine API",
        "endpoints": {
            "health_check": "/api/health",
            "chat": "/api/chat",
            "reset_chat": "/api/reset"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint that handles user queries."""
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    query = data['query']
    session_id = data.get('session_id', 'default')
    
    # Get or initialize chat history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    # Generate response
    response = chat_service.generate_response(query, chat_histories[session_id])
    
    # Update chat history
    chat_histories[session_id].append({"role": "user", "content": query})
    chat_histories[session_id].append({"role": "assistant", "content": response})
    
    return jsonify({
        "response": response,
        "session_id": session_id
    })

@app.route('/api/reset', methods=['POST'])
def reset_chat():
    """Reset the chat history for a session."""
    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id in chat_histories:
        chat_histories[session_id] = []
    
    return jsonify({"status": "Chat history reset", "session_id": session_id})

# Make sure app is listening on the port that Render expects
# This port will be used when running locally directly with python app.py
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Changed default to 10000 to match render.yaml
    app.run(host='0.0.0.0', port=port, debug=True)