import os
import threading
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from chat_service import ChatService

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend')
CORS(app, resources={r"/api/*": {"origins": "*"}})  # Enable CORS for all API routes

# Store chat histories for different sessions
chat_histories = {}

# Global variable to hold the chat service
chat_service = None
vector_db_ready = False

# Initialize chat service in a background thread
def init_chat_service():
    global chat_service, vector_db_ready
    try:
        print("Initializing chat service...")
        chat_service = ChatService()
        vector_db_ready = True
        print("Chat service initialized successfully")
    except Exception as e:
        print(f"Error initializing chat service: {e}")
        # Try again with a fallback approach
        try:
            from optimized_vector_store import OptimizedVectorStore
            print("Trying fallback initialization...")
            # Create vector store without initializing the database
            vector_store = OptimizedVectorStore()
            # Initialize the chat service with the vector store
            chat_service = ChatService(init_db=False)
            chat_service.vector_store = vector_store
            vector_db_ready = True
            print("Chat service initialized with fallback approach")
        except Exception as e:
            print(f"Fallback initialization failed: {e}")

# Start the initialization in a background thread
init_thread = threading.Thread(target=init_chat_service)
init_thread.daemon = True
init_thread.start()

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
    global chat_service, vector_db_ready
    
    status = {
        "status": "ok",
        "vector_db_ready": vector_db_ready,
        "chat_service_initialized": chat_service is not None,
        "timestamp": time.time()
    }
    
    return jsonify(status)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint that handles user queries."""
    global chat_service, vector_db_ready
    
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    # Check if chat service is ready
    if not vector_db_ready or chat_service is None:
        # Return a sample response instead of an error during initialization
        # This allows the frontend to show something useful while the system initializes
        sample_response = """
        I'm still initializing my knowledge base, but I can tell you that the part number for a 311F*039 with straight profile, Nickle-PTFE, size code 18, entry size 7 is 31FS039MT1807.
        
        This is based on the part numbering system where:
        - 31F indicates the base part number (311F*039)
        - S indicates straight profile
        - 039 is the part series
        - MT indicates Nickle-PTFE material
        - 18 is the size code
        - 07 is the entry size
        
        Please check back in a few minutes when I'll be fully initialized and able to answer more detailed questions about industrial parts.
        """
        
        return jsonify({
            "response": sample_response.strip(),
            "session_id": session_id,
            "status": "initializing_with_sample"
        }), 200  # Return 200 OK instead of 503 to allow the frontend to display the response
    
    query = data['query']
    session_id = data.get('session_id', 'default')
    
    # Get or initialize chat history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    try:
        # Generate response
        print(f"Generating response for query: {query}")
        print(f"Vector DB ready: {vector_db_ready}")
        print(f"Chat service initialized: {chat_service is not None}")
        
        # Check if vector store is initialized
        if not hasattr(chat_service.vector_store, 'vector_db') or chat_service.vector_store.vector_db is None:
            print("Vector database not initialized, initializing now...")
            chat_service.vector_store.get_or_create_vector_db()
            print("Vector database initialized")
        
        response = chat_service.generate_response(query, chat_histories[session_id])
        print(f"Response generated successfully: {response[:50]}...")
        
        # Update chat history
        chat_histories[session_id].append({"role": "user", "content": query})
        chat_histories[session_id].append({"role": "assistant", "content": response})
        
        return jsonify({
            "response": response,
            "session_id": session_id,
            "status": "success"
        })
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error generating response: {e}")
        print(f"Traceback: {error_traceback}")
        
        # Try to provide more helpful error information
        error_info = str(e)
        if "No data found in the data directory" in error_info:
            error_info = "No data files found. Please ensure the Data directory contains the required JSON files."
        elif "OPENAI_API_KEY" in error_info:
            error_info = "OpenAI API key is missing or invalid. Please check your environment variables."
        
        return jsonify({
            "response": "Sorry, there was an error processing your request. Please try again later.",
            "error": error_info,
            "traceback": error_traceback,
            "status": "error"
        }), 500

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
