from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
from backend.testing_system import process_user_input
from backend.fill_vector_db import load_model_with_retries
from backend.utils import *

from transformers import AutoTokenizer, AutoModel


# Create a FastAPI app instance
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Replace with your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer once at startup
model_name = QWEN_MODEL_NAME
model, tokenizer = load_model_with_retries(model_name, CACHE_DIR)


# Define the input model for the incoming request
class CodeRequest(BaseModel):
    code: str

# Define the output model for the response
class CodeResult(BaseModel):
    # link: str
    # metadata: Dict[str, str]
    label: str 
    language: str
    licenses: str
    stars: str
    similarity: str

# # Placeholder function that simulates finding top 10 similar codes
# def find_top10(code: str):
#     print("Received code:", code)  # Print the received code to verify it's being sent correctly
#     """
#     This is a placeholder for your actual function.
#     Replace this with the real function that finds similar code snippets.
#     """
#     # Example data to simulate real results
#     return [
#         {"label": "https://example.com/code1", "language": "Python", "licenses": "lala", "stars": "1500", "similarity": "80%"},
#         {"label": "https://example.com/code2", "language": "JavaScript", "licenses": "lala", "stars": "1200", "similarity": "78%"},
#         # Add up to 10 results or dynamically generate based on your logic
#     ]

@app.post("/process_code", response_model=List[CodeResult])
async def process_code(request: CodeRequest):
    code = request.code
    try:
        # Call the find_top10 function with the received code
        similar_codes = process_user_input(code, model, tokenizer)
        # print(similar_codes)
        return similar_codes  # Directly return the similar_codes list

    except Exception as e:
        # Handle any errors and return a 500 response if needed
        # Log the error and return a 500 response
        print("Error processing code:", e)
        raise HTTPException(status_code=500, detail=str(e))

# Run the server if the file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
