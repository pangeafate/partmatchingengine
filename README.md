# Part Matching Engine

A web service that allows users to ask questions about industrial parts and equipment information contained in JSON files. The service provides a chat-like interface for interacting with the data.

## Features

- Web-based chat interface for asking questions about industrial parts
- RAG (Retrieval-Augmented Generation) system powered by OpenAI's GPT-4o model
- Vector database using FAISS for efficient similarity search
- Hierarchical chunking strategy for better context retrieval
- Hybrid search combining vector similarity and keyword matching

## Project Structure

- `backend/`: Python Flask backend
  - `app.py`: Main Flask application
  - `chat_service.py`: Chat service for handling user queries
  - `vector_store.py`: Vector store for document retrieval
  - `requirements.txt`: Python dependencies
- `frontend/`: Web interface
  - `index.html`: Main HTML page
  - `styles.css`: CSS styles
  - `app.js`: JavaScript for the chat interface
- `Data/`: Directory for JSON data files

## Setup

1. Clone the repository:
```bash
git clone https://github.com/pangeafate/partmatchingengine.git
cd partmatchingengine
```

2. Install backend dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

4. Run the backend server:
```bash
cd backend
python app.py
```

5. Run the frontend server:
```bash
cd frontend
python -m http.server 8080
```

6. Open your browser and navigate to `http://localhost:8080`

## Usage

1. Type your question about industrial parts in the chat interface
2. The system will retrieve relevant information from the JSON files
3. GPT-4o will generate a response based on the retrieved context

## Technologies Used

- Backend: Python, Flask, LangChain, FAISS
- Frontend: HTML, CSS, JavaScript
- AI: OpenAI GPT-4o, OpenAI Embeddings
- Database: FAISS Vector Store

## License

MIT
