# Part Matching Engine

A web application for matching industrial parts using AI-powered semantic search.

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Create a `.env` file in the backend directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
5. Start the backend server:
   ```bash
   python app.py
   ```
6. Start the frontend server:
   ```bash
   cd frontend
   python -m http.server 8080
   ```
7. Access the application at http://localhost:8080

## Deployment to Render

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure the service:
   - Environment: Python
   - Build Command: `cd backend && pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`
5. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PORT`: 10000
6. Deploy the service

## Database Management

The application uses ChromaDB for vector storage. The database is automatically created and populated when the application starts. You can rebuild the database through the admin interface at `/admin`.

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access
- `PORT`: Port number for the server (default: 3001)

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
