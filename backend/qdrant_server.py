from backend.utils import *
import subprocess
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchAny

# Initialize Qdrant client
client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}")

# Start the qdrant
def start_qdrant_server():
    docker_command = (
        'docker run -p 6333:6333 -p 6335:6334 -v ./qdrant_storage:/qdrant/storage:z qdrant/qdrant'
    )
    try:
        subprocess.Popen(['start', 'cmd', '/c', docker_command], shell=True)
        print("Qdrant server started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while starting the Qdrant server: {e}")

