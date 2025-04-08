import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_database():
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data/chroma_db', exist_ok=True)
        
        # Copy the database files
        if os.path.exists('chroma_db'):
            logger.info("Copying database files to data directory...")
            for item in os.listdir('chroma_db'):
                src = os.path.join('chroma_db', item)
                dst = os.path.join('data/chroma_db', item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info("Database files copied successfully")
        else:
            logger.warning("No database files found to copy")
    except Exception as e:
        logger.error(f"Error exporting database: {e}")
        raise

if __name__ == "__main__":
    export_database() 