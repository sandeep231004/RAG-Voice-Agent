from qdrant_client import QdrantClient
import sys

db_path = "d:/Projects/RAG_Voice_Agent_Updated/vector_db"
collection_name = "rag_docs"

try:
    print(f"🔍 Checking Qdrant database at {db_path}\n")
    client = QdrantClient(path=db_path)
    
    # Get detailed collection info
    info = client.get_collection(collection_name)
    count = client.count(collection_name=collection_name)
    
    print(f"Collection: {collection_name}")
    print(f"Status: {info.status}")
    print(f"Points count: {info.points_count}")
    print(f"Vectors count: {count.count}")
    print(f"Vector size: {info.config.params.vector_size}")
    print(f"Distance: {info.config.params.distance}")
    print(f"Optimization: {info.optimization_status}")
    
    # Check first point to verify data
    points = client.scroll(
        collection_name=collection_name,
        limit=1
    )[0]
    
    if points:
        point = points[0]
        print(f"\nSample point:")
        print(f"ID: {point.id}")
        print(f"Vector dimension: {len(point.vector)}")
        print(f"Payload keys: {list(point.payload.keys())}")
    else:
        print("\nNo points found in collection")
        
except Exception as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
