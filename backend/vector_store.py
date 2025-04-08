import os
import json
import glob
import gc
import openai
from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

class SimpleOpenAIEmbeddings(Embeddings):
    """A simple implementation of OpenAI embeddings that doesn't require tiktoken."""
    
    def __init__(self, model="text-embedding-ada-002", dimensions=192):
        self.model = model
        self.dimensions = dimensions
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the OpenAI API."""
        embeddings = []
        for text in texts:
            response = openai.Embedding.create(
                model=self.model,
                input=text
            )
            embedding = response["data"][0]["embedding"]
            
            # Truncate dimensions if specified
            if self.dimensions is not None and self.dimensions < len(embedding):
                embedding = embedding[:self.dimensions]
                
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the OpenAI API."""
        response = openai.Embedding.create(
            model=self.model,
            input=text
        )
        embedding = response["data"][0]["embedding"]
        
        # Truncate dimensions if specified
        if self.dimensions is not None and self.dimensions < len(embedding):
            embedding = embedding[:self.dimensions]
            
        return embedding

class VectorStore:
    def __init__(self, data_dir: str = None, db_path: str = None, embedding_dimensions: int = 192):
        """Initialize the vector store.
        
        Args:
            data_dir: Directory containing JSON files
            db_path: Path to save the vector database
            embedding_dimensions: Number of dimensions to use for embeddings (smaller = less memory)
        """
        # If data_dir is not provided, use a path relative to the current directory
        if data_dir is None:
            # First check for Render environment
            render_data_path = "/data/source"
            if os.path.exists("/data"):
                self.data_dir = render_data_path
                # Ensure directory exists
                os.makedirs(self.data_dir, exist_ok=True)
                print(f"Using Render data directory: {self.data_dir}")
            elif os.path.basename(os.getcwd()) == "backend":
                self.data_dir = "../Data"
                print(f"Using relative backend data directory: {self.data_dir}")
            else:
                self.data_dir = "./Data"
                print(f"Using local data directory: {self.data_dir}")
        else:
            self.data_dir = data_dir
            print(f"Using provided data directory: {self.data_dir}")
        
        # If db_path is not provided, check for Render environment first
        if db_path is None:
            render_db_path = "/data/chroma_db"
            if os.path.exists("/data"):
                self.db_path = render_db_path
                print(f"Using Render database path: {self.db_path}")
            else:
                self.db_path = "chroma_db"
                print(f"Using local database path: {self.db_path}")
        else:
            self.db_path = db_path
            print(f"Using provided database path: {self.db_path}")
        
        # Ensure the db_path directory exists
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = SimpleOpenAIEmbeddings(dimensions=embedding_dimensions)
        self.vector_db = None
        
        # Batch processing settings
        self.batch_size = 25  # Process 25 items at a time
        
        # Add progress tracking
        self.build_progress = {
            "total_files": 0,
            "processed_files": 0,
            "total_batches": 0,
            "processed_batches": 0,
            "total_items": 0,
            "processed_items": 0,
            "status": "idle",
            "last_error": None,
            "last_update": None
        }
    
    def load_json_files(self) -> List[Dict[str, Any]]:
        """Load all JSON files from the data directory."""
        all_data = []
        
        # Load all JSON files in the data directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        print(f"Loading JSON files from {self.data_dir}")
        print(f"Found {len(json_files)} JSON files")
        
        for file_path in json_files:
            # Skip test_data.json
            if os.path.basename(file_path) == "test_data.json":
                print(f"Skipping test file: {file_path}")
                continue
                
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
        """Convert JSON data to Document objects for the vector store.
        
        This method implements a comprehensive JSON parsing strategy that handles:
        1. Nested objects and arrays
        2. Relationships between entities
        3. Metadata preservation
        4. Hierarchical structure
        """
        documents = []
        
        def sanitize_metadata(value: Any) -> Any:
            """Convert complex metadata types to simple types that ChromaDB can handle."""
            if isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return ', '.join(str(x) for x in value)
            elif isinstance(value, dict):
                return str(value)
            else:
                return str(value)
        
        def process_value(value, path: str = "", metadata: Dict = None) -> None:
            """Recursively process JSON values to create documents."""
            if metadata is None:
                metadata = {}
            
            # Sanitize metadata
            sanitized_metadata = {k: sanitize_metadata(v) for k, v in metadata.items()}
            
            if isinstance(value, dict):
                # Process dictionary
                content = []
                for key, val in value.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(val, (dict, list)):
                        process_value(val, new_path, sanitized_metadata)
                    else:
                        content.append(f"{key}: {val}")
                
                if content:
                    doc_metadata = sanitized_metadata.copy()
                    doc_metadata["path"] = path
                    doc_metadata["type"] = "object"
                    doc = Document(
                        page_content="\n".join(content),
                        metadata=doc_metadata
                    )
                    documents.append(doc)
                
            elif isinstance(value, list):
                # Process list
                for i, item in enumerate(value):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    if isinstance(item, (dict, list)):
                        process_value(item, new_path, sanitized_metadata)
                    else:
                        doc_metadata = sanitized_metadata.copy()
                        doc_metadata["path"] = path
                        doc_metadata["type"] = "list_item"
                        doc_metadata["index"] = i
                        doc = Document(
                            page_content=f"Item {i}: {item}",
                            metadata=doc_metadata
                        )
                        documents.append(doc)
            
            else:
                # Process primitive value
                doc_metadata = sanitized_metadata.copy()
                doc_metadata["path"] = path
                doc_metadata["type"] = "primitive"
                doc = Document(
                    page_content=f"{path}: {value}",
                    metadata=doc_metadata
                )
                documents.append(doc)
        
        # Process each item in the data
        for item in data:
            # Extract metadata if present
            metadata = {}
            if "metadata" in item:
                metadata = item["metadata"]
                del item["metadata"]
            
            # Process the item
            process_value(item, metadata=metadata)
        
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks of specified size."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            # Try to find a natural break point (newline or period)
            if end < len(text):
                # Look for the last newline within the chunk
                last_newline = text.rfind('\n', start, end)
                if last_newline > start + chunk_size // 2:  # Only use if it's not too close to the start
                    end = last_newline + 1
                else:
                    # Look for the last period within the chunk
                    last_period = text.rfind('. ', start, end)
                    if last_period > start + chunk_size // 2:
                        end = last_period + 2
            
            chunks.append(text[start:end])
            start = end - overlap  # Create overlap between chunks
        
        return chunks
    
    def create_vector_db(self) -> Chroma:
        """Create a Chroma vector database from JSON files."""
        import time
        
        # Check if any JSON files exist in the data directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        json_files = [f for f in json_files if os.path.basename(f) != "test_data.json"]
        
        if not json_files:
            print(f"Error: No valid JSON files found in data directory: {self.data_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Listing files in data directory:")
            try:
                files = os.listdir(self.data_dir)
                for file in files:
                    print(f"  - {file}")
            except Exception as e:
                print(f"Error listing files: {e}")
            
            self.build_progress["status"] = "error"
            self.build_progress["last_error"] = "No valid JSON files found"
            raise ValueError("No valid JSON files found in the data directory")
        
        # Initialize progress tracking
        self.build_progress["status"] = "processing"
        self.build_progress["total_files"] = len(json_files)
        self.build_progress["processed_files"] = 0
        self.build_progress["total_batches"] = 0
        self.build_progress["processed_batches"] = 0
        self.build_progress["total_items"] = 0
        self.build_progress["processed_items"] = 0
        self.build_progress["last_update"] = time.time()
        
        # Calculate total items and batches
        total_items = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                    total_items += len(data)
                    self.build_progress["total_batches"] += (len(data) + self.batch_size - 1) // self.batch_size
            except Exception as e:
                print(f"Error counting items in {json_file}: {e}")
        
        self.build_progress["total_items"] = total_items
        
        # Process data in batches
        vector_db = None
        processed_items = 0
        
        # Process each JSON file separately
        for json_file in json_files:
            print(f"Processing file: {json_file}")
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]
                    
                    # Process in batches
                    total_batches = (len(data) + self.batch_size - 1) // self.batch_size
                    print(f"Processing {len(data)} items in {total_batches} batches")
                    
                    for batch_idx in range(0, len(data), self.batch_size):
                        try:
                            batch_num = batch_idx // self.batch_size + 1
                            end_idx = min(batch_idx + self.batch_size, len(data))
                            batch_data = data[batch_idx:end_idx]
                            
                            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} items)")
                            
                            # Prepare documents for this batch
                            documents = self.prepare_documents(batch_data)
                            
                            if vector_db is None:
                                # Create new database with first batch
                                print("Creating new vector database")
                                vector_db = Chroma.from_documents(
                                    documents=documents,
                                    embedding=self.embeddings,
                                    persist_directory=self.db_path
                                )
                                vector_db.persist()
                            else:
                                # Add documents to existing database
                                print("Adding to existing vector database")
                                vector_db.add_documents(documents)
                                vector_db.persist()
                            
                            # Force garbage collection to free memory
                            documents = None
                            batch_data = None
                            gc.collect()
                            
                            # Update progress
                            self.build_progress["processed_batches"] += 1
                            processed_items += len(batch_data)
                            self.build_progress["processed_items"] = processed_items
                            self.build_progress["last_update"] = time.time()
                            
                            print(f"Completed batch {batch_num}/{total_batches}")
                            
                        except Exception as e:
                            print(f"Error processing batch {batch_num}: {e}")
                            import traceback
                            print(traceback.format_exc())
                            continue  # Continue with next batch
                    
                    # Update file progress
                    self.build_progress["processed_files"] += 1
                
            except Exception as e:
                print(f"Error processing file {json_file}: {e}")
                import traceback
                print(traceback.format_exc())
                continue  # Continue with next file
        
        if vector_db is None:
            self.build_progress["status"] = "error"
            self.build_progress["last_error"] = "Failed to create vector database"
            raise ValueError("Failed to create vector database")
        
        # Update final status
        self.build_progress["status"] = "complete"
        self.build_progress["last_update"] = time.time()
        
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
            # Check for pre-built database file
            chroma_db_file = os.path.join(self.db_path, "chroma.sqlite3")
            
            # Ensure embeddings are initialized
            if not hasattr(self, 'embeddings') or self.embeddings is None:
                print("Embeddings not initialized, creating them now")
                self.embeddings = SimpleOpenAIEmbeddings(dimensions=192)
            
            # Log database detection for debugging    
            if os.path.exists(chroma_db_file):
                print(f"Pre-built database file found at {chroma_db_file}")
                file_size = os.path.getsize(chroma_db_file) / (1024 * 1024)  # Size in MB
                print(f"Database file size: {file_size:.2f} MB")
            else:
                print(f"No pre-built database file found at {chroma_db_file}")
                print(f"Contents of {self.db_path}: {os.listdir(self.db_path) if os.path.exists(self.db_path) else 'directory not found'}")
                
            print(f"Loading vector database from {self.db_path}")
            try:
                self.vector_db = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings
                )
                # Print number of documents in the database for debugging
                if hasattr(self.vector_db, '_collection'):
                    count = self.vector_db._collection.count()
                    print(f"Vector database loaded with {count} documents")
                return self.vector_db
            except Exception as e:
                print(f"Error loading vector database: {e}")
                import traceback
                print(traceback.format_exc())
                raise
        else:
            print(f"Vector database not found at {self.db_path}")
            raise FileNotFoundError(f"Vector database not found at {self.db_path}")
    
    def get_or_create_vector_db(self) -> Chroma:
        """Get the vector database, creating it if it doesn't exist."""
        try:
            return self.load_vector_db()
        except FileNotFoundError:
            print("Database not found, creating new vector database")
            vector_db = self.create_vector_db()
            self.save_vector_db()
            return vector_db
        except Exception as e:
            print(f"Error in get_or_create_vector_db: {e}")
            # Try to create a new database as fallback
            print("Attempting to create new vector database due to loading error")
            vector_db = self.create_vector_db()
            self.save_vector_db()
            return vector_db
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform a similarity search on the vector database.
        
        Args:
            query: The query string to search for
            k: The number of documents to retrieve (default: 5)
            
        Returns:
            A list of Document objects that are most similar to the query
        """
        if not self.vector_db:
            self.get_or_create_vector_db()
        
        # Adjust k if it's larger than the number of documents in the database
        if hasattr(self.vector_db, '_collection'):
            total_docs = self.vector_db._collection.count()
            if k > total_docs:
                print(f"Adjusting k from {k} to {total_docs} (total documents in database)")
                k = max(1, total_docs)
        
        return self.vector_db.similarity_search(query, k=k)
    
    def rebuild_vector_db(self) -> None:
        """Force a complete rebuild of the vector database."""
        print("Rebuilding vector database...")
        
        # Delete the existing vector database if it exists
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
            print(f"Deleted existing vector database at {self.db_path}")
        
        # Create a new vector database
        vector_db = self.create_vector_db()
        self.save_vector_db()
        print("Vector database rebuilt successfully")
    
    def is_db_prebuilt(self) -> bool:
        """Check if a pre-built database exists at the db_path."""
        if not os.path.exists(self.db_path):
            return False
            
        # Check for the SQLite database file
        chroma_db_file = os.path.join(self.db_path, "chroma.sqlite3")
        return os.path.exists(chroma_db_file)