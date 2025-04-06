# test_optimized_db.py
from optimized_vector_store import OptimizedVectorStore

def test_masterwrap_query():
    """Test if MasterWrap is findable in the vector database."""
    print("Initializing OptimizedVectorStore...")
    store = OptimizedVectorStore()
    
    try:
        print("Testing query for 'MasterWrap'...")
        results = store.similarity_search("MasterWrap", k=5)
        
        found = False
        for i, doc in enumerate(results):
            content = doc.page_content.lower()
            if "masterwrap" in content:
                print(f"✅ Document {i+1} contains 'MasterWrap'")
                # Print metadata
                filename = doc.metadata.get("filename", "Unknown")
                page = doc.metadata.get("page_number", "Unknown")
                print(f"  Source: {filename}, Page: {page}")
                
                # Print a snippet
                index = content.find("masterwrap")
                start = max(0, index - 100)
                end = min(len(content), index + 100)
                snippet = content[start:end]
                print(f"  Snippet: '...{snippet}...'")
                
                found = True
                break
        
        if not found:
            print("❌ 'MasterWrap' not found in top results")
    except Exception as e:
        print(f"Error testing: {e}")

if __name__ == "__main__":
    test_masterwrap_query()