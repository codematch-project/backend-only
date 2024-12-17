import pandas as pd
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import time
from datetime import datetime
from backend.pull_data_v2_stack import *
from backend.splitv2_ppc import is_sample_too_short
from backend.utils import *
from backend.qdrant_server import *

# Define constants
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Global variables to store loaded model and tokenizer
loaded_model = None
loaded_tokenizer = None

# Global counter to maintain unique IDs across all splits, even if rows are skipped
global_id_counter = 0

# Checks if the qdrant collection exists
def ensure_collection_exists(collection_name, embedding_size):
    global global_id_counter
    global_id_counter = 0

    try:
        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")
            
            max_id = 0
            scroll_result, next_page_token = client.scroll(
                collection_name=collection_name,
                limit=1000,
                with_payload=False
            )

            while scroll_result:
                for point in scroll_result:
                    point_id = int(point.id)
                    max_id = max(max_id, point_id)
                
                if next_page_token:
                    scroll_result, next_page_token = client.scroll(
                        collection_name=collection_name,
                        limit=1000,
                        with_payload=False,
                        offset=next_page_token
                    )
                else:
                    break
            
            if max_id > 0:
                global_id_counter = max_id + 1
                print(f"Resuming from last ID: {max_id}. Starting at ID {global_id_counter}.")
            else:
                print("No points found in the collection, starting from ID 0.")
                    
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")

# Load the model
def load_model_with_retries(model_name, cache_dir, retries=3, wait_time=10):
    global loaded_model, loaded_tokenizer
    
    if loaded_model is not None and loaded_tokenizer is not None:
        print("Model and tokenizer are already loaded. Returning cached versions.")
        return loaded_model, loaded_tokenizer

    for attempt in range(retries):
        try:
            # Load the tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            
            if model_name == CODEBERT_MODEL_NAME or model_name == GRAPH_CODEBERT_MODEL_NAME:
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
            elif model_name == CODET5_MODEL_NAME:
                model = T5EncoderModel.from_pretrained(model_name, cache_dir=cache_dir)
            elif model_name == QWEN_MODEL_NAME:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    quantization_config=quantization_config,
                    device_map="auto"  # Automatically maps layers to available devices
                )
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
            
            loaded_model = model
            loaded_tokenizer = tokenizer
            return model, tokenizer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(wait_time)
            else:
                raise e

# Create the embedding
def create_embedding(code, model, tokenizer):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    if hasattr(outputs, 'last_hidden_state'):
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    elif hasattr(outputs, 'encoder_last_hidden_state'):
        embedding = outputs.encoder_last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    else:
        raise ValueError("Unexpected model output format.")
    return embedding

# Truncate the embedding to the target size if it exceeds the limit.
def reduce_embedding_size(embedding, target_size = EMBEDDING_SIZE):
    return embedding[:target_size] if embedding.shape[0] > target_size else embedding

# Create an embedding for the given code using the specified model with mixed precision.
def generate_code_embedding_generic(code, model, tokenizer):
    inputs = tokenizer(
        code,
        return_tensors="pt", 
        truncation=True, 
        padding=True
    )

    outputs = model(**inputs)

    # Extract embeddings based on model's output attributes
    if hasattr(outputs, "last_hidden_state"):  # Standard models (e.g., CodeBERT)
        embeddings = outputs.last_hidden_state
    elif hasattr(outputs, "logits"):  # Models outputting logits
        embeddings = outputs.logits
    elif hasattr(outputs, "hidden_states"):  # Models with intermediate hidden states
        embeddings = outputs.hidden_states[-1]  # Use the last layer
    else:
        raise AttributeError("Model output does not contain embeddings in a known attribute.")

    # Compute the mean embedding for the sequence
    embedding = embeddings.mean(dim=1).squeeze().detach().cpu().numpy()

    embedding = reduce_embedding_size(embedding, EMBEDDING_SIZE)

    return embedding

# Checks if the code sample exists in the qdrant collection
def example_exists_in_qdrant(content_id):
    query = Filter(
        must=[FieldCondition(key="content_id", match=MatchAny(any=[content_id]))]
    )

    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query,
            limit=1
        )
        return bool(points)
    except Exception as e:
        print(f"Error in checking content_id existence: {e}")
        return False

# Initialized embedding by content_id
def init_vector_db(embedding, metadata, point_id):
    point = PointStruct(
        id=point_id,
        vector=embedding.tolist(),
        payload=metadata
    )
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    print(f"Initialized embedding for content_id {metadata['content_id']} with metadata into the vector database.")

# Index the stack code samples
def index_from_stack(model, repo, sample, tokenizer):
    global global_id_counter
    to_add = True
    directory_id = repo['directory_id']

    content_id = sample['content_id']
    code = sample['content']

    if is_sample_too_short(code):
        print(f"Sample {content_id} is too short (<{MIN_TOKEN_NUM} tokens), skipping.")
        to_add = False

    elif code:
        if not example_exists_in_qdrant(content_id):
            metadata = {
                "content_id": content_id,
                "directory_id": directory_id,
                "part_in_code_name": sample['part_in_code_name'],
                "label": sample['label'], #this is the full link
                "language": sample.get('language'),
                "licenses": ', '.join(sample.get('detected_licenses', [])),
                "star_count": repo.get('star_events_count'),
                "content": code
            }
            embedding = generate_code_embedding_generic(code, model, tokenizer)
            # embedding = create_embedding(code, model, tokenizer)
            init_vector_db(embedding, metadata, global_id_counter)
            print(f"Indexed file {content_id} with label {metadata['label']} and ID {global_id_counter}.")
            global_id_counter += 1
        else:
            print(f"File {content_id} already exists in Qdrant, skipping this file.")
            to_add = False
    return to_add

# Attempt to index a sample with retries in case of failure.                
def index_with_retries(model, repo, sample, tokenizer):
    global global_id_counter
    to_add = False
    
    for attempt in range(MAX_RETRIES):
        try:
            to_add = index_from_stack(model, repo, sample, tokenizer)
            return to_add
        except Exception as e:
            print(f"Error processing sample: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"Retrying ({attempt + 1}/{MAX_RETRIES}) after {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. Skipping this sample.")
    global_id_counter += 1

# Index code examples from local files listed in the provided CSV file. Each file will be split, embedded, and added to the vector database.
def index_from_local_files(model, tokenizer, csv_path):
    global global_id_counter  # Ensure unique IDs across all entries

    # Load CSV data
    df = pd.read_csv(csv_path)

    # Iterate through each row in the CSV
    for _, row in df.iterrows():
        content_id = str(row['content_id'])
        file_path = str(row['file_path_pc'])
        file_url = row['file_url']
        licenses = row['licenses']
        lang = row['lang']
        star_count = row['star_count']
        directory_id = row['directory_id']

        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except FileNotFoundError:
            print(f"File {file_path} not found, skipping.")
            continue

        # Split the code if necessary
        splitted_samples = create_splitted_sample_from_pc({
            'content': code,
            'content_id': content_id,
            'file_url': file_url,
            'licenses': licenses,
            'language': lang,
            'star_count': star_count
        })

        # Index each split in the vector database
        for splitted_sample in splitted_samples:
            part_in_code_name = splitted_sample['part_in_code_name']
            code_part = splitted_sample['content']
            content_id = splitted_sample["content_id"]
            label = splitted_sample["label"]
            file_url = splitted_sample.get("file_url")

            # Check if the sample is too short and skip if necessary
            if is_sample_too_short(code_part):
                print(f"Sample {content_id} is too short (<{MIN_TOKEN_NUM} tokens), skipping.")
                continue

            # Check if the example already exists in Qdrant
            if not example_exists_in_qdrant(content_id):
                metadata = {
                    "content_id": content_id,
                    "directory_id": directory_id,
                    "part_in_code_name": part_in_code_name,
                    "label": label,
                    "language": lang,                    
                    "licenses": licenses,
                    "star_count": star_count,
                    "content": code_part
                }
                
                # Create embedding and store in Qdrant
                # embedding = create_embedding(code_part, model, tokenizer)
                embedding = generate_code_embedding_generic(code_part, model, tokenizer)
                init_vector_db(embedding, metadata, global_id_counter)
                
                print(f"Indexed example {content_id} with label {label} and ID {global_id_counter}.\n")
                global_id_counter += 1
            else:
                print(f"Example {content_id} already exists in Qdrant, skipping.\n")