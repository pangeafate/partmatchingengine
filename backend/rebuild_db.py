import os
from dotenv import load_dotenv
from vector_store import VectorStore

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize vector store
    vector_store = VectorStore()
    
    try:
        # Force rebuild of the vector database
        print("Starting database rebuild...")
        vector_store.rebuild_vector_db()
        print("Database rebuild completed successfully")
        
        # Verify the database
        db = vector_store.load_vector_db()
        if hasattr(db, '_collection'):
            count = db._collection.count()
            print(f"Verified database contains {count} documents")
        
    except Exception as e:
        print(f"Error rebuilding database: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()