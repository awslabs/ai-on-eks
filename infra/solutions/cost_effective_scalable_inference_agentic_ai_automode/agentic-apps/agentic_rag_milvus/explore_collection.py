import os
from dotenv import load_dotenv
from pymilvus import connections, Collection

# Load environment variables
load_dotenv()

# Get Milvus address from environment
milvus_address = os.getenv("MILVUS_ADDRESS", "localhost:19530")
milvus_host, milvus_port = milvus_address.split(":")

# Collection name
collection_name = "rag_documents"

# Connect to Milvus
print(f"Connecting to Milvus at {milvus_host}:{milvus_port}...")
try:
    connections.connect(
        alias="default", 
        host=milvus_host,
        port=milvus_port
    )
    
    # Get collection
    collection = Collection(collection_name)
    
    # 1. Check the schema
    schema = collection.schema
    print("\n1. Collection Schema:")
    print(f"Collection name: {collection_name}")
    print("Fields:")
    for field in schema.fields:
        print(f"  - {field.name}: {field.dtype} (is_primary_key: {field.is_primary})")
        if hasattr(field, 'params') and field.params:
            print(f"    Vector dimension: {field.params.get('dim')}")
    
    # 2. Get number of entities
    num_entities = collection.num_entities
    print(f"\n2. Number of entities: {num_entities}")
    
    # 3. Examine index information
    print("\n3. Index Information:")
    try:
        index_info = collection.index().params
        print(f"Index params: {index_info}")
    except Exception as e:
        print(f"Could not get index params: {e}")
    
    # Get more detailed information about vector field indexes
    try:
        for field_name in collection.index_info.keys():
            field_index = collection.index_info.get(field_name)
            if field_index:
                print(f"\nField '{field_name}' index:")
                print(f"  Index type: {field_index.get('index_type')}")
                print(f"  Metric type: {field_index.get('metric_type')}")
                print(f"  Params: {field_index.get('params')}")
    except Exception as e:
        print(f"Could not get detailed index info: {e}")
    
    # Close connection
    connections.disconnect("default")
    
except Exception as e:
    print(f"Error exploring Milvus collection: {e}")
