import os
import logging
from pathlib import Path
import PyPDF2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('d:/Projects/RAG_Voice_Agent_Updated/rebuild_db.log')
    ]
)
logger = logging.getLogger(__name__)

def process_pdf(file_path):
    """Process a PDF file and return chunks of text"""
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file)
            logger.info(f"Processing PDF with {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text.strip():
                    # Split text into smaller chunks (roughly sentence-sized)
                    sentences = text.replace('\n', ' ').split('.')
                    current_chunk = []
                    current_size = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip() + '.'
                        if current_size + len(sentence) > 512:  # Max chunk size
                            if current_chunk:
                                chunk_text = ' '.join(current_chunk)
                                chunks.append({
                                    'text': chunk_text,
                                    'metadata': {
                                        'source': Path(file_path).name,
                                        'page': page_num
                                    }
                                })
                            current_chunk = [sentence]
                            current_size = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_size += len(sentence)
                    
                    # Add remaining chunk
                    if current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                'source': Path(file_path).name,
                                'page': page_num
                            }
                        })
            
            logger.info(f"Created {len(chunks)} chunks from PDF")
            return chunks
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")
        return []

def main():
    # Paths
    db_path = "d:/Projects/RAG_Voice_Agent_Updated/vector_db"
    docs_path = "d:/Projects/RAG_Voice_Agent_Updated/documents"
    collection_name = "rag_docs"
    
    try:
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Qdrant
        logger.info(f"Connecting to vector database at {db_path}")
        client = QdrantClient(path=db_path)
        
        # Recreate collection
        logger.info(f"Creating new collection: {collection_name}")
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        
        # Process PDF
        pdf_path = os.path.join(docs_path, "YoLo research Paper.pdf")
        chunks = process_pdf(pdf_path)
        
        if not chunks:
            logger.error("No chunks created from PDF")
            return
        
        # Create embeddings and upload in batches
        batch_size = 50
        total_points = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in batch]
            embeddings = model.encode(texts)
            
            # Create points
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                points.append(models.PointStruct(
                    id=total_points + j,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk['text'],
                        'metadata': chunk['metadata']
                    }
                ))
            
            # Upload batch
            client.upsert(collection_name=collection_name, points=points)
            total_points += len(points)
            logger.info(f"Uploaded batch of {len(points)} points. Total: {total_points}")
        
        # Verify final count
        count = client.count(collection_name=collection_name)
        logger.info(f"Ingestion complete. Collection has {count.count} points")
        
        # Test search
        test_query = "What is YOLO's main objective?"
        test_embedding = model.encode(test_query)
        results = client.search(
            collection_name=collection_name,
            query_vector=test_embedding.tolist(),
            limit=1
        )
        
        if results:
            logger.info("Test search successful. Sample result:")
            logger.info(f"Text: {results[0].payload['text'][:200]}...")
            logger.info(f"Score: {results[0].score}")
        else:
            logger.warning("Test search returned no results")
        
    except Exception as e:
        logger.error(f"Error during database rebuild: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
