import argparse
import os
import logging
from vector_store import VectorStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Rebuild the vector database")
    parser.add_argument('--data-dir', help='Directory containing JSON files')
    parser.add_argument('--db-path', help='Path to save the vector database')
    args = parser.parse_args()
    
    # Check if we're running in Render environment
    render_env = os.path.exists('/data')
    
    if render_env:
        logger.info("Running in Render environment")
    else:
        logger.info("Running in local environment")
    
    # Initialize the vector store with appropriate paths
    store = VectorStore(
        data_dir=args.data_dir,
        db_path=args.db_path
    )
    
    # Log paths being used
    logger.info(f"Using data directory: {store.data_dir}")
    logger.info(f"Using database path: {store.db_path}")
    
    # Check if the data directory exists
    if not os.path.exists(store.data_dir):
        logger.error(f"Data directory does not exist: {store.data_dir}")
        return
    
    # Check if any files exist in the data directory
    try:
        files = os.listdir(store.data_dir)
        logger.info(f"Files in data directory: {files}")
    except Exception as e:
        logger.error(f"Error listing data directory: {e}")
    
    logger.info("Rebuilding vector database...")
    store.rebuild_vector_db()
    
    logger.info("Database rebuild complete!")

if __name__ == "__main__":
    main()