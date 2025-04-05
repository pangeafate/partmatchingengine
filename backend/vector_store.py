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
        """Convert JSON data to Document objects for the vector store.
        
        This method processes structured JSON data from PDF documents,
        preserving page numbers, element types, and hierarchical relationships.
        """
        documents = []
        
        # Group items by page number
        pages = {}
        for item in data:
            if "metadata" in item and "page_number" in item["metadata"]:
                page_num = item["metadata"]["page_number"]
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append(item)
        
        # Process each page
        for page_num, items in pages.items():
            # Sort items by their hierarchical structure (if available)
            items.sort(key=lambda x: x["metadata"].get("category_depth", 0) if "metadata" in x else 0)
            
            # Group items by their parent_id to maintain hierarchical relationships
            parent_groups = {}
            for item in items:
                if "metadata" in item and "parent_id" in item["metadata"]:
                    parent_id = item["metadata"]["parent_id"]
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(item)
            
            # Create a document for each page
            page_content = f"Page {page_num}:\n\n"
            
            # Add titles and headings first (if any)
            titles = [item for item in items if item.get("type") in ["Title", "Heading"]]
            for title in titles:
                if "text" in title and title["text"]:
                    page_content += f"{title['text']}\n"
            
            # Add narrative text
            narratives = [item for item in items if item.get("type") == "NarrativeText"]
            for narrative in narratives:
                if "text" in narrative and narrative["text"]:
                    page_content += f"{narrative['text']}\n"
            
            # Add other text content
            other_text = [item for item in items if item.get("type") not in ["Title", "Heading", "NarrativeText"] and "text" in item and item["text"]]
            for text_item in other_text:
                page_content += f"{text_item['text']}\n"
            
            # Create metadata with key information for retrieval
            metadata = {
                "page_number": page_num,
                "filename": items[0]["metadata"]["filename"] if items and "metadata" in items[0] and "filename" in items[0]["metadata"] else "",
                "source": "pdf_json_data"
            }
            
            # Create a Document object for the page
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
            
            # For larger pages, create additional chunks to ensure better context retrieval
            if len(page_content) > 1000:
                chunks = self._chunk_text(page_content, chunk_size=1000, overlap=200)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = i
                    chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                    documents.append(chunk_doc)
        
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
        
        return self.vector_db.similarity_search(query, k=k)
