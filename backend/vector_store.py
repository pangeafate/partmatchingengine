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
        
        # Load all JSON files in the data directory
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        print(f"Loading JSON files from {self.data_dir}")
        print(f"Found {len(json_files)} JSON files")
        
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
        
        This method implements a hierarchical chunking strategy for PDF documents:
        1. Page-Level Chunks: Base chunk for each page with metadata
        2. Section-Level Chunks: Group elements within the same section
        3. Element-Level Chunks: Preserve important individual elements
        
        It also integrates rich metadata with each chunk for better retrieval.
        """
        documents = []
        
        # Group items by filename first
        files = {}
        for item in data:
            if "metadata" in item and "filename" in item["metadata"]:
                filename = item["metadata"]["filename"]
                if filename not in files:
                    files[filename] = []
                files[filename].append(item)
        
        # Process each file
        for filename, file_items in files.items():
            # Group items by page number
            pages = {}
            for item in file_items:
                if "metadata" in item and "page_number" in item["metadata"]:
                    page_num = item["metadata"]["page_number"]
                    if page_num not in pages:
                        pages[page_num] = []
                    pages[page_num].append(item)
            
            # Process each page
            for page_num, page_items in pages.items():
                # Sort items by their hierarchical structure
                page_items.sort(key=lambda x: x["metadata"].get("category_depth", 0) if "metadata" in x else 0)
                
                # Create a document for the entire page
                page_content = f"Page {page_num}:\n\n"
                
                # Extract all titles on the page for context
                titles = [item for item in page_items if item.get("type") in ["Title"]]
                title_text = ""
                for title in titles:
                    if "text" in title and title["text"]:
                        title_text += f"{title['text']} "
                
                # Group items by their parent_id to maintain hierarchical relationships
                parent_groups = {}
                for item in page_items:
                    if "metadata" in item and "parent_id" in item["metadata"]:
                        parent_id = item["metadata"]["parent_id"]
                        if parent_id not in parent_groups:
                            parent_groups[parent_id] = []
                        parent_groups[parent_id].append(item)
                
                # Process each parent group (section) separately
                for parent_id, section_items in parent_groups.items():
                    section_content = ""
                    
                    # Add titles first
                    section_titles = [item for item in section_items if item.get("type") in ["Title", "Heading"]]
                    for title in section_titles:
                        if "text" in title and title["text"]:
                            section_content += f"{title['text']}\n"
                    
                    # Add narrative text
                    narratives = [item for item in section_items if item.get("type") == "NarrativeText"]
                    for narrative in narratives:
                        if "text" in narrative and narrative["text"]:
                            section_content += f"{narrative['text']}\n"
                    
                    # Add other text content
                    other_text = [item for item in section_items if 
                                 item.get("type") not in ["Title", "Heading", "NarrativeText"] and 
                                 "text" in item and item["text"]]
                    for text_item in other_text:
                        section_content += f"{text_item['text']}\n"
                    
                    if section_content:
                        # Add to the overall page content
                        page_content += section_content + "\n"
                        
                        # Create a section-level chunk with rich metadata
                        section_metadata = {
                            "page_number": page_num,
                            "filename": filename,
                            "source": "pdf_json_data",
                            "chunk_type": "section",
                            "parent_id": parent_id,
                            "title_context": title_text.strip()
                        }
                        
                        # Create a Document object for the section
                        section_doc = Document(page_content=section_content, metadata=section_metadata)
                        documents.append(section_doc)
                
                # Handle items without parent_id (top-level items)
                top_level_items = [item for item in page_items if 
                                  "metadata" not in item or 
                                  "parent_id" not in item["metadata"]]
                
                top_level_content = ""
                for item in top_level_items:
                    if "text" in item and item["text"]:
                        top_level_content += f"{item['text']}\n"
                
                if top_level_content:
                    page_content += top_level_content
                
                # Create metadata with key information for retrieval
                page_metadata = {
                    "page_number": page_num,
                    "filename": filename,
                    "source": "pdf_json_data",
                    "chunk_type": "page",
                    "title_context": title_text.strip()
                }
                
                # Create a Document object for the entire page
                page_doc = Document(page_content=page_content, metadata=page_metadata)
                documents.append(page_doc)
                
                # For larger pages, create additional overlapping chunks
                if len(page_content) > 800:
                    chunks = self._chunk_text(page_content, chunk_size=800, overlap=200)
                    for i, chunk in enumerate(chunks):
                        chunk_metadata = page_metadata.copy()
                        chunk_metadata["chunk_index"] = i
                        chunk_metadata["chunk_type"] = "page_chunk"
                        chunk_doc = Document(page_content=chunk, metadata=chunk_metadata)
                        documents.append(chunk_doc)
        
        # Process regular JSON data (not from PDFs)
        regular_items = [item for item in data if "metadata" not in item or "filename" not in item["metadata"]]
        for item in regular_items:
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
        
        print(f"Created {len(documents)} document chunks")
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