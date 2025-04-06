from vector_store import VectorStore
import os
import glob

def test_vector_db():
    # Initialize the vector store
    print("Initializing VectorStore...")
    vector_store = VectorStore()
    
    # 1. Check what files are in the data directory
    data_dir = vector_store.data_dir
    print(f"\nStep 1: Checking files in {data_dir}")
    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")
    
    # 2. Load the vector database
    print("\nStep 2: Loading vector database...")
    try:
        db = vector_store.get_or_create_vector_db()
        print("Vector database loaded successfully!")
        
        # Get the collection stats to see how many documents are indexed
        collection = db._collection
        documents_count = collection.count()
        print(f"Total documents in vector store: {documents_count}")
    except Exception as e:
        print(f"Error loading vector database: {e}")
        return
    
    # 3. Test specific queries to verify content is accessible
    print("\nStep 3: Testing specific queries...")
    
    # Test queries
    test_queries = [
        "MasterWrap",
        "shielding solution",
        "circular connector",
        # Add other relevant terms that should be in your database
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = vector_store.similarity_search(query, k=3)
            print(f"  Found {len(results)} documents")
            
            # Check if any results mention our target term
            target_term = "MasterWrap"
            found_target = False
            for i, doc in enumerate(results):
                doc_content = doc.page_content.lower()
                if target_term.lower() in doc_content:
                    print(f"  ✅ Document {i+1} contains '{target_term}'")
                    found_target = True
                    # Print a snippet of the document
                    start_idx = max(0, doc_content.find(target_term.lower()) - 50)
                    end_idx = min(len(doc_content), doc_content.find(target_term.lower()) + len(target_term) + 50)
                    snippet = doc_content[start_idx:end_idx]
                    print(f"  Snippet: '...{snippet}...'")
                    # Print metadata
                    filename = doc.metadata.get("filename", "Unknown")
                    page = doc.metadata.get("page_number", "Unknown")
                    print(f"  Source: {filename}, Page: {page}")
                    break
            
            if not found_target and query.lower() == target_term.lower():
                print(f"  ❌ '{target_term}' not found in top results")
                
        except Exception as e:
            print(f"  Error running query '{query}': {e}")
    
    print("\nVector database testing complete!")

if __name__ == "__main__":
    test_vector_db()