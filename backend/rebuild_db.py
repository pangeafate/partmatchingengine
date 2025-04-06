# rebuild_db.py
import argparse
from vector_store import VectorStore

def main():
    parser = argparse.ArgumentParser(description="Rebuild the vector database")
    args = parser.parse_args()
    
    store = VectorStore()
    
    print("Rebuilding vector database...")
    store.rebuild_vector_db()
    
    print("Database rebuild complete!")

if __name__ == "__main__":
    main()
