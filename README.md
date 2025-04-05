# Part Matching Engine

A web service that allows users to ask questions about industrial parts and equipment using a chat-like interface. The service is powered by ChatGPT 4o and uses a RAG (Retrieval-Augmented Generation) approach with vector embeddings to provide accurate responses based on the information in the JSON files.

## Features

- Chat-like interface for asking questions about industrial parts
- RAG (Retrieval-Augmented Generation) approach for accurate responses
- Vector database for efficient similarity search
- Session management for maintaining conversation context
- Responsive design that works on desktop and mobile devices

## Tech Stack

- **Backend**: Flask, OpenAI API, LangChain, ChromaDB
- **Frontend**: HTML, CSS, JavaScript
- **Vector Database**: ChromaDB with OpenAI embeddings

## Project Structure

```
.
├── backend/               # Backend Flask API
│   ├── app.py             # Main Flask application
│   ├── chat_service.py    # Chat service using OpenAI API
│   ├── requirements.txt   # Python dependencies
│   └── vector_store.py    # Vector database implementation
├── frontend/              # Frontend web interface
│   ├── app.js             # JavaScript for the web interface
│   ├── index.html         # HTML structure
│   └── styles.css         # CSS styling
├── Data/                  # Directory for JSON data files
├── .env                   # Environment variables
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js (optional, for serving the frontend)
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pangeafate/partmatchingengine.git
   cd partmatchingengine
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your OpenAI API key:
   ```
   OPENAI_API_KEY="your-api-key-here"
   ```

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python app.py
   ```
   The server will run on http://localhost:5001

2. Open the frontend in your browser:
   - You can simply open the `frontend/index.html` file in your browser
   - Or serve it using a simple HTTP server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
     Then open http://localhost:8000 in your browser

## Usage

1. The application will load with a welcome message
2. Type your question about industrial parts in the input field
3. Press Enter or click the send button to submit your question
4. The system will retrieve relevant information and generate a response
5. You can continue the conversation with follow-up questions
6. Use the "Reset Chat" button to start a new conversation

## Data

The application uses JSON files stored in the `Data` directory. Each JSON file contains information about industrial parts and equipment. The system automatically loads all JSON files in this directory and indexes them in the vector database.

## Development

### Adding New Data

1. Add your JSON files to the `Data/` directory.
2. The system will automatically process these files when creating the vector database.

### Customizing the System

- Modify the system prompt in `chat_service.py` to change how the assistant responds.
- Adjust the vector search parameters in `vector_store.py` to tune retrieval performance.
- Customize the frontend styling in `styles.css` to match your branding.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the ChatGPT 4o model
- LangChain for the RAG implementation
- ChromaDB for the vector database
- Flask for the backend framework
