import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from backend.fill_vector_db import load_model_with_retries, create_embedding, generate_code_embedding_generic
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from backend.utils import *

# Constants
TOP_K_RESULTS = 3

client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# Search for similar code in the Qdrant vector database based on the given embedding.
def search_similar_code(embedding):
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=TOP_K_RESULTS
    )
    return search_results

# Display the top K similar results from Qdrant.
def display_results(results):
    list_results = []

    print("\nTop similar code segments:")
    for result in results:
        label = result.payload.get('label', 'No label')
        language = result.payload.get('language', 'No language')
        part_in_code_name = result.payload.get('part_in_code_name', 'Full file')
        licenses = result.payload.get('licenses', 'No licenses')
        stars = result.payload.get('star_count', 'No Stars')
        similarity_score = result.score

        dict_result = {
            "label": label,
            "language": language,
            "part_in_code_name": part_in_code_name,
            "licenses": licenses,
            "stars": str(stars) if stars is not None else "No",
            "similarity": f"{similarity_score:.4f}"
        }

        list_results.append(dict_result)
        
        print(f"Label: {label}, Language: {language}, part_in_code_name: {part_in_code_name}, Licenses: {licenses}, Stars: {stars}, Similarity: {similarity_score:.4f}")
      
    print("\n")
    return list_results

# Process a user-input code snippet.
def process_user_input(user_code, model, tokenizer):
    list_results = []
    # embedding = create_embedding(user_code, model, tokenizer)
    embedding = generate_code_embedding_generic(user_code, model, tokenizer)
    results = search_similar_code(embedding)
    list_results = display_results(results)
        
    return list_results

def main():
    list_results =[]

    model_name = CODEBERT_MODEL_NAME
    model, tokenizer = load_model_with_retries(model_name, CACHE_DIR)

    user_code = input("Enter your code snippet: ").strip()
    list_results = process_user_input(user_code, model, tokenizer)
    
    
if __name__ == '__main__':
    main()