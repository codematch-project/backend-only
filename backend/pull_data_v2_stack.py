import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import requests
from collections import Counter
from datasets import load_dataset
from backend.splitv2_ppc import preprocess_code
from backend.utils import *
from backend.qdrant_server import *

#Load The Stack dataset in streaming mode.
def load_dataset_stream():
    return load_dataset(
        "bigcode/the-stack-v2-train-smol-ids",
        split="train",
        streaming=True,
        token=HUGGING_FACE_TOKEN
    )

# Checks if the repo exists in the qdrant collection
def check_directory_in_qdrant(directory_id):
    query = Filter(
        must=[FieldCondition(key="directory_id", match=MatchAny(any=[directory_id]))]
    )
    try:
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=query,
            limit=1
        )
        return bool(points)
    except Exception as e:
        print(f"Error in checking directory_id existence: {e}")
        return False

# Filter the dataset to include only files in supported languages with valid licenses.
def filter_supported_language_data(ds, k):
    filtered_data = []
    for language in SUPPORTED_LANGUAGES:
        print(f"Loading up to {k} rows with valid licenses for: {language}")

        language_data = []
        count = 0
        for row in ds:
            directory_id = row.get("directory_id")
            if not check_directory_in_qdrant(directory_id):
                if row.get("gha_license_id") is not None and row.get("star_events_count") is not None and any(file.get("language", "").lower() == language for file in row.get("files", [])):
                    language_data.append(row)
                    count += 1
                    if count >= k:
                        break  # Stop after loading `k` rows with valid licenses for this language
            else:
                 print(f"Repository {directory_id} already exists in Qdrant, skipping this repo.")
        
        print(f"Loaded {len(language_data)} rows with valid licenses for {language}.\n")
        filtered_data.extend(language_data)
    
    return filtered_data

# Rest of the code remains unchanged
def determine_main_language(row):
    language_counts = Counter(
        file.get("language").lower() 
        for file in row.get("files", []) 
        if file.get("language", "").lower() in SUPPORTED_LANGUAGES
    )
    main_language = language_counts.most_common(1)[0][0] if language_counts else None
    return main_language

def extract_repository_info(row):
    repo_info = {
        "repo_name": row.get("repo_name"),
        "repo_url": row.get("repo_url"),
        "directory_id": row.get("directory_id"),
        "branch_name": row.get("branch_name"),
        "num_files": len(row.get("files", [])),
        "gha_license_id": row.get("gha_license_id"),
        "star_count": row.get("star_events_count", 0),
        "fork_count": row.get("fork_events_count", 0),
        "main_language": determine_main_language(row),
    }
    return repo_info

def extract_file_info(row):
    file_info = []
    for file in row.get("files", []):
        file_language = file.get("language", "").lower()
        if file_language in SUPPORTED_LANGUAGES:
            file_info.append({
                "file_url": f"{row.get('repo_url')}/blob/{row.get('revision_id')}{file.get('path')}",
                "raw_file_url": f"{row.get('repo_url').replace('github.com', 'raw.githubusercontent.com')}/{row.get('revision_id')}{file.get('path')}",
                "content_id": file.get("content_id"),
                "path": file.get("path"),
                "language": file_language,
                "license_type": file.get("license_type"),
                "detected_licenses": file.get("detected_licenses", []),
            })
    return file_info

# Get file content from url
def fetch_file_content(file_url):
    try:
        response = requests.get(file_url)
        response.raise_for_status()  # Raise an error for failed requests
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file content from {file_url}: {e}")
        return None

# Function to create the label for each sample
def create_label(file_url, first_line=None, last_line=None):
    if first_line is not None and last_line is not None:
        return f"{file_url}#{first_line}-{last_line}"
    return file_url

# Create and return a new sample after splitting the content
def create_splitted_sample(sample):
    content = sample["content"]
    language = sample.get("language", "").lower()
    
    # Use the original content_id without any appended suffix as the base
    base_content_id = sample["content_id"]

    # Split the content into separate functions/classes/modules
    processed_code_dict = preprocess_code(content, language)
    
    if processed_code_dict is None:
        sample["part_in_code_name"] = ""
        sample["label"] = create_label(sample["file_url"])
        sample["first_line"] = None
        sample["last_line"] = None
        sample["content_id"] = base_content_id
        return [sample]

    splitted_samples = []
    for idx, (part_name, part_info) in enumerate(processed_code_dict.items()):
        part_content = part_info.get("content", "")
        first_line = part_info.get("first_line")
        last_line = part_info.get("last_line")
        
        new_sample = sample.copy()
        new_sample["content"] = part_content
        new_sample["part_in_code_name"] = part_name
        new_sample["label"] = create_label(sample["file_url"], first_line, last_line)
        new_sample["first_line"] = first_line
        new_sample["last_line"] = last_line
        
        # Ensure unique identifier for each part by appending a single index suffix
        new_sample["content_id"] = f"{base_content_id}_{idx}"
        
        splitted_samples.append(new_sample)

    return splitted_samples

# Create and return a new sample after splitting the content from a local file
def create_splitted_sample_from_pc(file_info):
    content = file_info["content"]
    language = file_info.get("lang", "").lower()

    # Preprocess the content code by splitting it into functions/classes/modules
    processed_code_dict = preprocess_code(content, language)

    if processed_code_dict:  # If there are splits to process
        # Create new rows for each split part and add them to the splitted_sample
        splitted_samples = []

        # Modify content_id with an index for each part
        for idx, (part_name, part_data) in enumerate(processed_code_dict.items(), start=0):
            new_sample = file_info.copy()  # Copy the original file info
            new_sample["content"] = part_data.get("content")  # Update content with the split part
            new_sample["part_in_code_name"] = part_name  # Add part name from processed code
            
            # Append index to content_id to ensure uniqueness for each split
            new_sample["content_id"] = f"{new_sample['content_id']}_{idx}"
            
            # Retrieve first_line and last_line from part_data, if available
            first_line = part_data.get("first_line")
            last_line = part_data.get("last_line")
            
            # Generate the label using file_url, first_line, and last_line
            new_sample["label"] = create_label(new_sample["file_url"], first_line, last_line)
            new_sample["first_line"] = first_line
            new_sample["last_line"] = last_line
            
            splitted_samples.append(new_sample)
        
        return splitted_samples  # Return the list of split samples

    else:  # If no splitting is needed, return the original content as a single sample
        file_info["part_in_code_name"] = ""
        file_info["label"] = create_label(file_info["file_url"])  # Set label for unsplit content
        file_info["first_line"] = None
        file_info["last_line"] = None
        return [file_info]  # Wrap in a list to keep the output format consistent


def main():
    # Load and filter data for a limited number of rows per language
    k = 1  # Number of rows to load per language
    ds = load_dataset_stream()
    filtered_data = filter_supported_language_data(ds, k)

    # Process each repository row in the filtered data
    for row in filtered_data:
        repo_info = extract_repository_info(row)
        print("Repository Info:")
        for key, value in repo_info.items():
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
        
        print("=" * 80)
        
        # Print file-level information and split content
        print("Files in the repository:")
        for file in extract_file_info(row):
            print(f"File URL: {file['file_url']}")
            print(f"Language: {file['language'].capitalize()}")
            print(f"Content ID: {file['content_id']}")
            print(f"Path: {file['path']}")
            print(f"License Type: {file['license_type']}")
            print(f"Detected Licenses: {file['detected_licenses']}")
            print("-" * 30)
            
            # Fetch content and create splits
            file_content = fetch_file_content(file['raw_file_url'])
            if file_content:
                file["content"] = file_content
                splitted_samples = create_splitted_sample(file)
                
                # Print each split part
                for sample in splitted_samples:
                    print(f"Part Name: {sample['part_in_code_name']}")
                    print(f"Split Content ID: {sample['content_id']}")
                    print(f"Label: {sample['label']}")
                    print(f"First Line: {sample['first_line']}")
                    print(f"Last Line: {sample['last_line']}")
                    # print("Content:\n", sample["content"])
                    print("=" * 80)

if __name__ == '__main__':
    main()
