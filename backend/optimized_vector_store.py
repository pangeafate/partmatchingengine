# optimized_vector_store.py
import os
import json
import glob
import time
import gc
import psutil
import numpy as np
from typing import List, Dict, Any, Generator, Optional
from itertools import islice

import openai
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

# Memory monitoring function
def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

class BatchedOpenAIEmbeddings(Embeddings):
    """OpenAI embeddings with batched API calls and memory optimization."""
    
    def __init__(self, model="text-embedding-ada-002", batch_size=10, dimensions=None):
        """
        Initialize the embeddings.
        
        Args:
            model: The OpenAI embedding model to use
            batch_size: Number of texts to embed in a single API call
            dimensions: If provided, truncate embeddings to this many dimensions
        """
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions
        
        # Use a smaller model if memory is a concern
        if model == "text-embedding-ada-002" and dimensions is None:
            print("Using full 1536-dimension embeddings. Set dimensions parameter to reduce memory usage.")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batched API calls."""
        embeddings = []
        
        # Log memory usage
        start_mem = get_memory_usage()
        print(f"Memory before embedding: {start_mem:.2f} MB")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            try:
                response = openai.Embedding.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item["embedding"] for item in response["data"]]
                
                # Truncate dimensions if specified
                if self.dimensions is not None and self.dimensions < len(batch_embeddings[0]):
                    batch_embeddings = [emb[:self.dimensions] for emb in batch_embeddings]
                
                embeddings.extend(batch_embeddings)
                
                # Add a small delay to avoid rate limits
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
                
                # Log progress and memory usage periodically
                if (i // self.batch_size) % 10 == 0:
                    current_mem = get_memory_usage()
                    print(f"Embedded {i+len(batch)}/{len(texts)} texts. Memory: {current_mem:.2f} MB")
                    
                # Force garbage collection periodically
                if (i // self.batch_size) % 20 == 0:
                    gc.collect()
                    
            except Exception as e:
                print(f"Error embedding batch {i//self.batch_size}: {e}")
                # For failed batches, return zero vectors as placeholders
                vec_size = 1536 if self.dimensions is None else self.dimensions
                batch_embeddings = [[0.0] * vec_size for _ in range(len(batch))]
                embeddings.extend(batch_embeddings)
                time.sleep(1)  # Longer delay after error
        
        # Log final memory usage
        end_mem = get_memory_usage()
        print(f"Memory after embedding: {end_mem:.2f} MB (change: {end_mem-start_mem:.2f} MB)")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        response = openai.Embedding.create(
            model=self.model,
            input=text
        )
        embedding = response["data"][0]["embedding"]
        
        # Truncate dimensions if specified
        if self.dimensions is not None and self.dimensions < len(embedding):
            embedding = embedding[:self.dimensions]
            
        return embedding

class OptimizedVectorStore:
    def __init__(self, data_dir: str = "../Data", db_path: str = "chroma_db", 
                 embedding_dimensions: int = 768, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the vector store with memory optimization parameters.
        
        Args:
            data_dir: Directory containing JSON files
            db_path: Path to save the vector database
            embedding_dimensions: Number of dimensions to use for embeddings (smaller = less memory)
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.data_dir = data_dir
        self.db_path = db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use memory-optimized embeddings
        self.embeddings = BatchedOpenAIEmbeddings(
            batch_size=10,  # Smaller batch size to reduce memory usage
            dimensions=embedding_dimensions  # Truncate embeddings to reduce memory usage
        )
        
        self.vector_db = None
        
        # Log initial memory usage
        mem = get_memory_usage()
        print(f"Initial memory usage: {mem:.2f} MB")
        print(f"Using {embedding_dimensions}-dimension embeddings (vs. 1536 default)")
        print(f"Using chunk size of {chunk_size} with {chunk_overlap} overlap")
    
    def prepare_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Convert JSON data to Document objects for the vector store."""
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
                if len(page_content) > self.chunk_size / 2:  # Only chunk if content is at least half the chunk size
                    chunks = self._chunk_text(page_content)  # Use instance variables for chunk size and overlap
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
            try:
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
            except:
                # Skip items that can't be serialized
                continue
        
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks of specified size."""
        # Use instance variables if not specified
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        # Log chunking parameters
        print(f"Chunking text of length {len(text)} with size={chunk_size}, overlap={overlap}")
        
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
            
            # Force garbage collection periodically during chunking of large texts
            if len(chunks) % 100 == 0:
                gc.collect()
        
        return chunks
    
    def add_documents_in_batches(self, documents, batch_size=20):
        """Add documents to the vector store in batches."""
        # Log memory usage
        start_mem = get_memory_usage()
        print(f"Memory before adding documents: {start_mem:.2f} MB")
        
        # Initialize vector store if needed
        if self.vector_db is None:
            try:
                self.vector_db = self.load_vector_db()
                print("Using existing vector database")
            except FileNotFoundError:
                print("Creating new vector database...")
                # Create with a small initial batch
                initial_batch = documents[:min(10, len(documents))]  # Use smaller initial batch
                self.vector_db = Chroma.from_documents(
                    documents=initial_batch,
                    embedding=self.embeddings,
                    persist_directory=self.db_path
                )
                self.save_vector_db()
                # Remove the initial batch from documents
                documents = documents[min(10, len(documents)):]
        
        # Process remaining documents in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                # Add batch to vector store
                self.vector_db.add_documents(batch)
                
                # Save more frequently (every 2 batches) to manage memory
                if batch_num % 2 == 0 or batch_num == total_batches:
                    self.save_vector_db()
                    print(f"Saved vector database after batch {batch_num}")
                    # Force garbage collection
                    gc.collect()
                
                # Log memory usage periodically
                if batch_num % 5 == 0:
                    current_mem = get_memory_usage()
                    print(f"Memory after {batch_num} batches: {current_mem:.2f} MB")
                    
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                # Continue with next batch
        
        # Log final memory usage
        end_mem = get_memory_usage()
        print(f"Memory after adding all documents: {end_mem:.2f} MB (change: {end_mem-start_mem:.2f} MB)")
    
    def process_single_file(self, file_path, batch_size=20):
        """Process a single file and add it to the vector database."""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
            
        print(f"Processing file: {os.path.basename(file_path)}")
        
        # Log memory usage
        start_mem = get_memory_usage()
        print(f"Memory before processing file: {start_mem:.2f} MB")
        
        try:
            # Load the file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert to list if it's not already
            if not isinstance(data, list):
                data = [data]
                
            print(f"Loaded {len(data)} items from {os.path.basename(file_path)}")
            
            # Log memory after loading
            after_load_mem = get_memory_usage()
            print(f"Memory after loading file: {after_load_mem:.2f} MB (change: {after_load_mem-start_mem:.2f} MB)")
            
            # Prepare documents
            documents = self.prepare_documents(data)
            print(f"Created {len(documents)} document chunks")
            
            # Clear data to free memory
            data = None
            gc.collect()
            
            # Log memory after document preparation
            after_prep_mem = get_memory_usage()
            print(f"Memory after document preparation: {after_prep_mem:.2f} MB")
            
            # Add documents in batches
            self.add_documents_in_batches(documents, batch_size)
            
            # Clear documents to free memory
            documents = None
            gc.collect()
            
            # Log final memory usage
            end_mem = get_memory_usage()
            print(f"Memory after processing file: {end_mem:.2f} MB (total change: {end_mem-start_mem:.2f} MB)")
            
            print(f"Completed processing {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def process_all_files_in_batches(self, batch_size=20):
        """Process all files one at a time."""
        # Get all JSON files
        json_files = glob.glob(os.path.join(self.data_dir, "*.json"))
        
        print(f"Found {len(json_files)} JSON files in {self.data_dir}")
        
        # Log initial memory usage
        start_mem = get_memory_usage()
        print(f"Initial memory usage: {start_mem:.2f} MB")
        
        for i, file_path in enumerate(json_files):
            print(f"\nProcessing file {i+1}/{len(json_files)}: {os.path.basename(file_path)}")
            self.process_single_file(file_path, batch_size)
            
            # Force garbage collection between files
            gc.collect()
            
            # Log memory usage after each file
            current_mem = get_memory_usage()
            print(f"Memory after processing file {i+1}: {current_mem:.2f} MB")
        
        # Log final memory usage
        end_mem = get_memory_usage()
        print(f"\nFinal memory usage: {end_mem:.2f} MB (total change: {end_mem-start_mem:.2f} MB)")
        print("All files processed successfully")
    
    def save_vector_db(self):
        """Save the vector database to disk."""
        if self.vector_db:
            self.vector_db.persist()
            print(f"Vector database saved to {self.db_path}")
        else:
            raise ValueError("Vector database has not been created yet")
    
    def load_vector_db(self):
        """Load the vector database from disk."""
        if os.path.exists(self.db_path):
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            return self.vector_db
        else:
            raise FileNotFoundError(f"Vector database not found at {self.db_path}")
    
    def similarity_search(self, query: str, k: int = 5):
        """Perform a similarity search on the vector database."""
        if not self.vector_db:
            self.vector_db = self.load_vector_db()
        
        return self.vector_db.similarity_search(query, k=k)
