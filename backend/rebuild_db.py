from vector_store import VectorStore

if __name__ == "__main__":
    # Initialize the vector store
    vector_store = VectorStore()
    
    # Rebuild the vector database
    vector_store.rebuild_vector_db()
    
    print("Vector database rebuild complete!")