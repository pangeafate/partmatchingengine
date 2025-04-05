import os
import json
import glob
import openai
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

class SimpleOpenAIEmbeddings(Embeddings):
    """A simple implementation of OpenAI embeddings that doesn't require tiktoken."""
    
    def __init__(self, model="text-embedding-ada-002"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the OpenAI API."""
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            embeddings.append(response["data"][0]["embedding"])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the OpenAI API."""
        response = openai.Embedding.create(
            model=self.model,
            input=text
        )
        return response["data"][0]["embedding"]

class VectorStore:
    def __init__(self, data_dir: str = "../Data", db_path: str = "chroma_db"):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory containing JSON files
            db_path: Path to save the vector database
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.embeddings = SimpleOpenAIEmbeddings()
        self.vector_db = None
    
    def load_json_files(self) -> List[Dict[str, Any]]:
        """Load all JSON files from the data directory."""
        all_data = []
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")
        for file_path in json_files:
            print(f"Loading file: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        print(f"  - Loaded {len(data)} items from list")
                        all_data.extend(data)
                    else:
                        print(f"  - Loaded 1 item from object")
                        all_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Total items loaded: {len(all_data)}")
        return all_data
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Convert JSON data to Document objects for the vector store."""
        documents = []
        
        for item in data:
            # Convert the item to a string representation for the content
            content = json.dumps(item, indent=2)
            
            # Create metadata with key information for retrieval
            metadata = {
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "source": "json_data"
            }
            
            # Create a Document object
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents
    
    def create_vector_db(self) -> Chroma:
        """Create a Chroma vector database from JSON files."""
        # Load data from JSON files
        data = self.load_json_files()
        
        if not data:
            raise ValueError("No data found in the data directory")
        
        # Prepare documents
        documents = self.prepare_documents(data)
        
        # Create vector store
        vector_db = Chroma.from_documents(
            documents=documents, 
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        self.vector_db = vector_db
        
        return vector_db
    
    def save_vector_db(self) -> None:
        """Save the vector database to disk."""
        if self.vector_db:
            self.vector_db.persist()
            print(f"Vector database saved to {self.db_path}")
        else:
            raise ValueError("Vector database has not been created yet")
    
    def load_vector_db(self) -> Chroma:
        """Load the vector database from disk."""
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            return self.vector_db
        else:
            raise FileNotFoundError(f"Vector database not found at {self.db_path}")
    
    def get_or_create_vector_db(self) -> Chroma:
        """Get the vector database, creating it if it doesn't exist."""
        try:
            return self.load_vector_db()
        except FileNotFoundError:
            vector_db = self.create_vector_db()
            self.save_vector_db()
            return vector_db
    
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Perform a similarity search on the vector database."""
        if not self.vector_db:
            self.get_or_create_vector_db()
        
        return self.vector_db.similarity_search(query, k=k)
