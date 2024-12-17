import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pull_data_v2_stack import *
from fill_vector_db import *
from splitv2_ppc import *
import time
from transformers import AutoTokenizer, AutoModel
from backend.utils import *
from backend.qdrant_server import *

MAX_ROWS_PER_LANG = 1

# Process each file in the repository, fetch content, split it, and attempt to index it in Qdrant.
def process_repository_files(repository, model, tokenizer):
    files_count = 0
    
    # Extract file-level information and split content
    for file_info in extract_file_info(repository):
        print(f"Processing file: {file_info['file_url']}")

        # Fetch content from the raw file URL
        file_content = fetch_file_content(file_info['raw_file_url'])
        if not file_content:
            print(f"Failed to fetch content for file: {file_info['file_url']}")
            continue

        # Add content to file info for splitting
        file_info["content"] = file_content
        splitted_samples = create_splitted_sample(file_info)

        # Index each split sample in the vector database
        for sample in splitted_samples:
            to_add = index_with_retries(model, repository, sample, tokenizer)
            
        # If a new row was added, increment the sample_count
        if to_add:
            files_count += 1
            
    print(f"Added {files_count} files to the vector DB from {repository['repo_name']}.")
    
def main():
    # Start timer
    start_time = time.time()

    # Start Qdrant server
    start_qdrant_server()
    
    # Load model and tokenizer
    print(f"We are using the {MODEL_NAME} model")
    model, tokenizer = load_model_with_retries(MODEL_NAME, CACHE_DIR)
    
    # Ensure the collection exists
    ensure_collection_exists(COLLECTION_NAME, EMBEDDING_SIZE)
    
    # Load and filter dataset for each supported language
    ds_stream = load_dataset_stream()
    filtered_data = filter_supported_language_data(ds_stream, MAX_ROWS_PER_LANG)
    
    # Process each repository row in the filtered data
    for repo in filtered_data:
        repo_info = extract_repository_info(repo)
        print("Repository Info:")
        for key, value in repo_info.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
        print("=" * 80)
        
        # Process each file within the repository
        process_repository_files(repo, model, tokenizer)
        print(f"Finished processing repository: {repo_info['repo_name']}.\n")
    
    '''
    # Get local file
    index_from_local_files(model, tokenizer, 'backend/files_info.csv')
    '''
    
    # End timer and print total time taken
    end_time = time.time()
    total_time = (end_time - start_time) / 60  # Total time in minutes
    print(f"Total time taken: {total_time:.2f} minutes")

if __name__ == '__main__':
    main()