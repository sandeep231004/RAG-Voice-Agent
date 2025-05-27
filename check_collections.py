from qdrant_client import QdrantClient
import os

# Local Qdrant path
db_path = "d:/Projects/RAG_Voice_Agent_Updated/vector_db"

print("🔍 Checking Qdrant vector database...\n")

# Check if DB directory exists
if not os.path.exists(db_path):
    print(f"❌ Database path does not exist: {db_path}")
    exit(1)

print(f"✅ Database path exists: {db_path}")
print(f"📁 Contents: {os.listdir(db_path)}\n")

# Initialize Qdrant client for local DB
client = QdrantClient(path=db_path)

# Fetch and list all collections
collections = client.get_collections()
if not collections.collections:
    print("⚠️ No collections found.")
else:
    print("📚 Collections and vector counts:\n")
    for col in collections.collections:
        info = client.get_collection(col.name)
        count = client.count(collection_name=col.name)
        print(f"🗂️  Collection: {col.name}")
        print(f"   ➤ Number of vectors (info): {info.points_count}")
        print(f"   ➤ Number of vectors (count): {count.count}")
        print(f"   ➤ Vector size: {info.config.params.vector_size}")
        print(f"   ➤ Status: {info.status}")
        print(f"   ➤ Optimization status: {info.optimization_status}\n")
