import os
from dotenv import load_dotenv
from pymilvus import connections, utility

# Load environment variables
load_dotenv()

# Get Milvus address from environment
milvus_address = os.getenv("MILVUS_ADDRESS", "localhost:19530")
milvus_host, milvus_port = milvus_address.split(":")

# Connect to Milvus
print(f"Connecting to Milvus at {milvus_host}:{milvus_port}...")
try:
    connections.connect(
        alias="default", 
        host=milvus_host,
        port=milvus_port
    )
    
    # List all collections
    collections = utility.list_collections()
    print("Available collections:")
    for collection in collections:
        print(f"- {collection}")
        
    # Close connection
    connections.disconnect("default")
    
except Exception as e:
    print(f"Error connecting to Milvus: {e}")
