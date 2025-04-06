# rebuild_db.py
import os
import argparse
from optimized_vector_store import OptimizedVectorStore

def main():
    parser = argparse.ArgumentParser(description="Rebuild the vector database with optimized batching")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--file", type=str, help="Process only this specific file (optional)")
    args = parser.parse_args()
    
    store = OptimizedVectorStore()
    
    if args.file:
        file_path = os.path.join(store.data_dir, args.file)
        print(f"Processing only file: {args.file}")
        store.process_single_file(file_path, args.batch_size)
    else:
        print("Processing all files")
        store.process_all_files_in_batches(args.batch_size)
    
    print("Database rebuild complete!")

if __name__ == "__main__":
    main()