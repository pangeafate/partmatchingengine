import os
import threading
import time
import logging
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from chat_service import ChatService

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a file handler for persistent logging
try:
    if os.path.exists('/data'):
        log_file = '/data/app.log'
    else:
        log_file = 'app.log'
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")
except Exception as e:
    logger.error(f"Failed to set up file logging: {e}")

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
        logger.info("Initializing chat service...")
        
        # Log environment information
        logger.info(f"Current working directory: {os.getcwd()}")
        
        # Check for OpenAI API key
        api_key_exists = bool(os.environ.get("OPENAI_API_KEY"))
        logger.info(f"OpenAI API key exists: {api_key_exists}")
        
        # Check database paths
        db_path = "/data/chroma_db" if os.path.exists("/data") else "chroma_db"
        logger.info(f"Database path: {db_path}")
        logger.info(f"Database path exists: {os.path.exists(db_path)}")
        
        # Check specifically for pre-built database file
        prebuilt_db_file = os.path.join(db_path, "chroma.sqlite3")
        if os.path.exists(prebuilt_db_file):
            db_size = os.path.getsize(prebuilt_db_file) / (1024 * 1024)  # Size in MB
            logger.info(f"Pre-built database found at {prebuilt_db_file} (Size: {db_size:.2f} MB)")
        else:
            logger.warning(f"Pre-built database file not found at {prebuilt_db_file}")
            if os.path.exists(db_path):
                logger.info(f"Contents of {db_path}: {os.listdir(db_path)}")
        
        # Check for Data directory
        data_path = "/data/source" if os.path.exists("/data") else "../Data" if os.path.basename(os.getcwd()) == "backend" else "./Data"
        logger.info(f"Data directory path: {data_path}")
        logger.info(f"Data directory exists: {os.path.exists(data_path)}")
        
        # Check for files in Data directory
        try:
            if os.path.exists(data_path):
                files = os.listdir(data_path)
                logger.info(f"Files in data directory: {files}")
        except Exception as e:
            logger.error(f"Error listing data directory: {e}")
        
        # Initialize chat service - set init_db=False to avoid rebuilding the database
        chat_service = ChatService(init_db=False)
        
        # Verify vector store is properly initialized
        if hasattr(chat_service, 'vector_store'):
            # Explicitly load the database instead of rebuilding it
            logger.info("Loading pre-built vector database...")
            chat_service.vector_store.load_vector_db()
            if chat_service.vector_store.vector_db is not None:
                doc_count = 0
                if hasattr(chat_service.vector_store.vector_db, '_collection'):
                    doc_count = chat_service.vector_store.vector_db._collection.count()
                logger.info(f"Vector database loaded successfully with {doc_count} documents")
                vector_db_ready = True
            else:
                logger.warning("Vector database loaded but appears to be empty")
                vector_db_ready = False
        else:
            logger.error("Chat service initialized but vector_store not found")
            vector_db_ready = False
        
        if vector_db_ready:
            logger.info("Chat service initialized successfully")
        else:
            logger.warning("Chat service initialized but vector database not ready")
    except Exception as e:
        logger.error(f"Error initializing chat service: {e}")
        
        # Try again with a fallback approach
        try:
            from vector_store import VectorStore
            logger.info("Trying fallback initialization...")
            
            # Create vector store with explicit instructions to use pre-built database
            vector_store = VectorStore(embedding_dimensions=192)
            
            # Initialize the chat service with the vector store
            chat_service = ChatService(init_db=False)
            chat_service.vector_store = vector_store
            
            # Try to load the vector database explicitly
            logger.info("Loading vector database in fallback...")
            if os.path.exists(vector_store.db_path):
                chat_service.vector_store.load_vector_db()
                vector_db_ready = True
                logger.info("Vector database loaded in fallback approach")
            else:
                # If no database exists, try creating it
                logger.info("No database found, initializing a new one...")
                chat_service.vector_store.get_or_create_vector_db()
                vector_db_ready = True
                logger.info("New vector database created in fallback approach")
            
        except Exception as e:
            logger.error(f"Fallback initialization failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

# Start the initialization immediately instead of in a background thread
# This ensures the database is ready before accepting requests
init_chat_service()

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

@app.route('/admin')
def admin_page():
    """Admin page for database management."""
    return send_from_directory(app.static_folder, 'admin.html')

@app.route('/api/admin/rebuild-db', methods=['POST'])
def rebuild_db():
    """Start rebuilding the vector database."""
    global chat_service, vector_db_ready
    
    try:
        # Start rebuild in a background thread
        def rebuild_task():
            global chat_service, vector_db_ready
            
            try:
                vector_db_ready = False
                if chat_service and hasattr(chat_service, 'vector_store'):
                    chat_service.vector_store.rebuild_vector_db()
                else:
                    from vector_store import VectorStore
                    store = VectorStore(embedding_dimensions=192)  # Ensure consistent dimensions
                    store.rebuild_vector_db()
                vector_db_ready = True
            except Exception as e:
                logger.error(f"Error rebuilding database: {e}")
                if chat_service and hasattr(chat_service, 'vector_store') and hasattr(chat_service.vector_store, 'build_progress'):
                    chat_service.vector_store.build_progress["status"] = "error"
                    chat_service.vector_store.build_progress["last_error"] = str(e)
        
        thread = threading.Thread(target=rebuild_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Error starting rebuild: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/db-status', methods=['GET'])
def db_status():
    """Get the database build status."""
    global chat_service
    
    if chat_service and hasattr(chat_service, 'vector_store') and hasattr(chat_service.vector_store, 'build_progress'):
        return jsonify(chat_service.vector_store.build_progress)
    else:
        # Return a default status if not initialized
        return jsonify({
            "total_files": 0,
            "processed_files": 0,
            "total_batches": 0,
            "processed_batches": 0,
            "total_items": 0,
            "processed_items": 0,
            "status": "not_initialized",
            "last_error": None,
            "last_update": None
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global chat_service, vector_db_ready
    
    # Check if the data directory exists and is accessible
    if os.path.exists("/data"):
        data_path = "/data/source"
    elif os.path.basename(os.getcwd()) == "backend":
        data_path = "../Data"
    else:
        data_path = "./Data"
    
    # Check database path
    if os.path.exists("/data"):
        db_path = "/data/chroma_db"
    else:
        db_path = "chroma_db"
    
    status = {
        "status": "ok",
        "vector_db_ready": vector_db_ready,
        "chat_service_initialized": chat_service is not None,
        "timestamp": time.time(),
        "environment": {
            "current_dir": os.getcwd(),
            "data_dir_exists": os.path.exists(data_path),
            "db_dir_exists": os.path.exists(db_path),
            "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY"))
        }
    }
    
    # Add more detailed information if chat service is initialized
    if chat_service is not None and hasattr(chat_service, "vector_store"):
        try:
            # Check for pre-built database file
            db_file = os.path.join(chat_service.vector_store.db_path, "chroma.sqlite3")
            db_file_exists = os.path.exists(db_file)
            
            status["vector_store_info"] = {
                "data_dir": chat_service.vector_store.data_dir,
                "db_path": chat_service.vector_store.db_path,
                "data_dir_exists": os.path.exists(chat_service.vector_store.data_dir),
                "db_path_exists": os.path.exists(chat_service.vector_store.db_path),
                "db_file_exists": db_file_exists
            }
            
            # Add number of documents if vector_db is loaded
            if hasattr(chat_service.vector_store, 'vector_db') and chat_service.vector_store.vector_db is not None:
                if hasattr(chat_service.vector_store.vector_db, '_collection'):
                    doc_count = chat_service.vector_store.vector_db._collection.count()
                    status["vector_store_info"]["document_count"] = doc_count
        except Exception as e:
            status["vector_store_error"] = str(e)
    
    return jsonify(status)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint that handles user queries."""
    global chat_service, vector_db_ready
    
    # Debug output
    logger.debug(f"Chat endpoint called, vector_db_ready: {vector_db_ready}, chat_service initialized: {chat_service is not None}")
    
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({"error": "Query is required"}), 400
    
    # Check if chat service is ready
    if not vector_db_ready or chat_service is None:
        logger.warning("Chat service not ready yet")
        return jsonify({
            "response": "The system is still initializing. Please try again in a moment.",
            "status": "initializing"
        }), 503  # Service Unavailable
    
    query = data['query']
    session_id = data.get('session_id', 'default')
    
    # Get or initialize chat history for this session
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    try:
        # Generate response
        logger.info(f"Generating response for query: {query}")
        logger.info(f"Vector DB ready: {vector_db_ready}")
        logger.info(f"Chat service initialized: {chat_service is not None}")
        
        # Log environment information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Data directory path: {chat_service.vector_store.data_dir}")
        logger.info(f"Data directory exists: {os.path.exists(chat_service.vector_store.data_dir)}")
        logger.info(f"Database path: {chat_service.vector_store.db_path}")
        logger.info(f"Database path exists: {os.path.exists(chat_service.vector_store.db_path)}")
        
        # Try to list files in the data directory
        try:
            data_files = os.listdir(chat_service.vector_store.data_dir)
            logger.info(f"Files in data directory: {data_files}")
        except Exception as e:
            logger.error(f"Error listing data directory: {e}")
        
        # Check OpenAI API key
        logger.info(f"OpenAI API key exists: {bool(os.environ.get('OPENAI_API_KEY'))}")
        
        # Check if vector store is initialized
        if not hasattr(chat_service.vector_store, 'vector_db') or chat_service.vector_store.vector_db is None:
            logger.info("Vector database not initialized, initializing now...")
            # Try to load first instead of recreating
            try:
                chat_service.vector_store.load_vector_db()
            except:
                chat_service.vector_store.get_or_create_vector_db()
            logger.info("Vector database initialized")
        
        response = chat_service.generate_response(query, chat_histories[session_id])
        logger.info(f"Response generated successfully: {response[:50]}...")
        
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
        logger.error(f"Error generating response: {e}")
        logger.error(f"Traceback: {error_traceback}")
        
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