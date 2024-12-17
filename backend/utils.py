CODEBERT_MODEL_NAME = "microsoft/codebert-base"
CODET5_MODEL_NAME = "Salesforce/codet5-base-multi-sum"
GRAPH_CODEBERT_MODEL_NAME = "microsoft/graphcodebert-base"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "code_embeddings"
EMBEDDING_SIZE = 768

MIN_TOKEN_NUM = 50
MAX_TOKEN_NUM = 510 # CODEBERT MODEL MAX TOKENS

# Supported languages - 6 languages
SUPPORTED_LANGUAGES = [
    'python', 'javascript', 'java', 'php', 'go', 'ruby'
]

import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
HUGGING_FACE_TOKEN = os.environ.get('HUGGING_FACE_TOKEN')

CACHE_DIR = "./huggingface_models"

# Final model name
MODEL_NAME = QWEN_MODEL_NAME