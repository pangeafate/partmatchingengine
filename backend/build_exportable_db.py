import os
import json
import gc
import logging
import argparse
import time
from vector_store import VectorStore, SimpleOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Define a function to filter complex metadata
def filter_complex_metadata(metadata_dict):
    """Filter out complex metadata that Chroma can't handle."""
    filtered_metadata = {}
    for key, value in metadata_dict.items():
        # Only keep simple types that Chroma can handle
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # Convert lists to strings if they contain simple types
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                filtered_metadata[key] = str(value)
    return filtered_metadata

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_file_in_chunks(file_path, chunk_size=5, db_path="./exportable_db", dimensions=192):
    """Process a large JSON file in small chunks to avoid memory issues."""
    logger.info(f"Processing file: {file_path} with chunk size {chunk_size}")
    
    # Initialize embeddings with specified dimensions
    embeddings = SimpleOpenAIEmbeddings(dimensions=dimensions)
    
    # Create or load the vector database
    vector_db = None
    if os.path.exists(db_path) and os.listdir(db_path):
        logger.info(f"Loading existing vector database from {db_path}")
        try:
            vector_db = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            logger.info("Will create a new database")
    
    # Open the file and read it line by line
    with open(file_path, 'r') as f:
        # Load the entire JSON content
        logger.info("Loading JSON file...")
        try:
            data = json.load(f)
            logger.info(f"Loaded JSON with {len(data) if isinstance(data, list) else 1} items")
            
            # Convert to list if it's not already
            if not isinstance(data, list):
                data = [data]
            
            # Process in very small chunks
            total_items = len(data)
            total_chunks = (total_items + chunk_size - 1) // chunk_size
            logger.info(f"Processing {total_items} items in {total_chunks} chunks")
            
            for chunk_idx in range(0, total_items, chunk_size):
                chunk_num = chunk_idx // chunk_size + 1
                end_idx = min(chunk_idx + chunk_size, total_items)
                chunk_data = data[chunk_idx:end_idx]
                
                logger.info(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk_data)} items)")
                
                # Convert items to documents
                documents = []
                for item in chunk_data:
                    # Extract text content
                    if "text" in item:
                        text = item["text"]
                    else:
                        # For items without text, use a JSON representation
                        text = json.dumps(item, indent=2)
                    
                    # Create metadata
                    metadata = {}
                    if "metadata" in item:
                        metadata = item["metadata"]
                    else:
                        # For items without metadata, use basic fields
                        for key in ["id", "name", "type", "page_number", "filename"]:
                            if key in item:
                                metadata[key] = item[key]
                    
                    # Filter complex metadata to avoid Chroma errors
                    metadata = filter_complex_metadata(metadata)
                    
                    # Create document
                    doc = Document(page_content=text, metadata=metadata)
                    documents.append(doc)
                
                logger.info(f"Created {len(documents)} documents for chunk {chunk_num}")
                
                # Add to vector database
                if vector_db is None:
                    logger.info("Creating new vector database")
                    try:
                        vector_db = Chroma.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            persist_directory=db_path
                        )
                    except ValueError as e:
                        logger.error(f"Error creating vector database: {e}")
                        # Try again with more aggressive metadata filtering
                        logger.info("Trying with more aggressive metadata filtering")
                        for doc in documents:
                            # Keep only simple metadata fields
                            simple_metadata = {}
                            for key, value in doc.metadata.items():
                                if isinstance(value, (str, int, float, bool)):
                                    simple_metadata[key] = value
                            doc.metadata = simple_metadata
                        
                        vector_db = Chroma.from_documents(
                            documents=documents,
                            embedding=embeddings,
                            persist_directory=db_path
                        )
                else:
                    logger.info("Adding to existing vector database")
                    try:
                        vector_db.add_documents(documents)
                    except ValueError as e:
                        logger.error(f"Error adding documents: {e}")
                        # Try again with more aggressive metadata filtering
                        logger.info("Trying with more aggressive metadata filtering")
                        for doc in documents:
                            # Keep only simple metadata fields
                            simple_metadata = {}
                            for key, value in doc.metadata.items():
                                if isinstance(value, (str, int, float, bool)):
                                    simple_metadata[key] = value
                            doc.metadata = simple_metadata
                        
                        vector_db.add_documents(documents)
                
                # Save after each chunk
                vector_db.persist()
                logger.info(f"Saved vector database after chunk {chunk_num}")
                
                # Clear memory
                documents = None
                chunk_data = None
                gc.collect()
                
                # Brief pause to let system recover
                time.sleep(1)
                
                logger.info(f"Completed chunk {chunk_num}/{total_chunks}")
            
            logger.info("Finished processing file")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

def main():
    parser = argparse.ArgumentParser(description="Build an exportable vector database with memory-efficient processing")
    parser.add_argument('--data-dir', default='./Data', help='Directory containing JSON files')
    parser.add_argument('--db-path', default='./exportable_db', help='Path to save the vector database')
    parser.add_argument('--chunk-size', type=int, default=5, help='Number of items to process in each chunk')
    parser.add_argument('--dimensions', type=int, default=192, help='Number of dimensions for embeddings')
    args = parser.parse_args()
    
    # Ensure the database directory exists
    os.makedirs(args.db_path, exist_ok=True)
    
    # Process each JSON file in the data directory
    json_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    logger.info(f"Found {len(json_files)} JSON files in {args.data_dir}")
    
    for json_file in json_files:
        file_path = os.path.join(args.data_dir, json_file)
        logger.info(f"Processing {json_file}...")
        success = process_file_in_chunks(
            file_path, 
            chunk_size=args.chunk_size,
            db_path=args.db_path,
            dimensions=args.dimensions
        )
        if success:
            logger.info(f"Successfully processed {json_file}")
        else:
            logger.error(f"Failed to process {json_file}")
    
    logger.info("Database build complete!")

if __name__ == "__main__":
    main()
