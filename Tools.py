import os
import re
import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional
from qdrant_client.http import models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('d:/Projects/RAG_Voice_Agent_Updated/search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Tool:
    """Base class for agent tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, *args, **kwargs) -> str:
        raise NotImplementedError("Tool subclasses must implement run()")

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class SearchDocumentsTool(Tool):
    """Tool for searching the RAG knowledge base."""

    def __init__(self, vector_db, embedding_model, collection_name: str, top_k: int = 5):
        super().__init__(
            name="search_documents",
            description="Search the knowledge base for relevant information on a topic."
        )
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.top_k = top_k

    def run(self, query: str) -> str:
        try:
            logger.info(f"Processing search query: {query}")
              # Check available collections
            collections = self.vector_db.get_collections()
            collection_names = [c.name for c in collections.collections]
            logger.info(f"Available collections: {collection_names}")

            if self.collection_name not in collection_names:
                logger.error(f"Collection {self.collection_name} not found among {collection_names}")
                return "Error: Knowledge base not properly initialized."
                  # Count vectors
            try:
                # Get detailed collection info
                collection_info = self.vector_db.get_collection(self.collection_name)
                collection_count = self.vector_db.count(collection_name=self.collection_name)
                
                logger.info(f"Collection '{self.collection_name}' status:")
                logger.info(f"  - Points count: {collection_info.points_count}")
                logger.info(f"  - Vector count: {collection_count.count}")
                
                if collection_count.count == 0:
                    logger.warning(f"Collection '{self.collection_name}' contains 0 vectors.")
                    return "Error: Knowledge base is empty."
            except Exception as e:
                logger.error(f"Error checking collection: {e}")
                return f"Error checking collection: {str(e)}"

            # Generate embedding
            query_embedding = self.embedding_model.encode(query.strip())

            # Search (no vector_name)
            results = self.vector_db.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=self.top_k
            )

            if not results:
                return "No relevant information found in the knowledge base."

            # Format results
            formatted_results = []
            for result in results:
                metadata = result.payload.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                page = metadata.get("page", "")
                source_info = f"{source} (Page {page})" if page else source
                text = result.payload.get("text", "").strip()
                score = result.score or 0.0
                if score >= 0.3 and text:
                    formatted_results.append(f"[Relevance: {score:.2f}] From {source_info}:\n{text}\n")

            if not formatted_results:
                return "No sufficiently relevant information found in the knowledge base."

            return "Found relevant information:\n\n" + "\n---\n".join(formatted_results)

        except Exception as e:
            logger.error(f"Error during document search: {str(e)}")
            return f"An error occurred while searching the knowledge base: {str(e)}"



class SaveNoteTool(Tool):
    """Tool for saving user notes."""

    def __init__(self, notes_dir: str = "user_notes"):
        super().__init__(
            name="save_note",
            description="Save a note or reminder for the user."
        )
        self.notes_dir = notes_dir
        os.makedirs(notes_dir, exist_ok=True)

    def run(self, note_content: str, title: Optional[str] = None) -> str:
        try:
            if not title:
                title = f"Note_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            filename = os.path.join(self.notes_dir, f"{title}.txt")

            with open(filename, 'w') as f:
                f.write(note_content)

            return f"Note saved successfully as '{title}'."
        except Exception as e:
            return f"Error saving note: {str(e)}"

class WebSearchTool(Tool):
    """Tool for performing web searches."""

    def __init__(self, search_client):
        super().__init__(
            name="web_search", # Name of the tool.
            description="Search the web for up-to-date information on a topic."
        )
        self.search_client = search_client 

    def run(self, query: str) -> str:
        try:
            # Perform the web search using the search client
            search_results = self.search_client.search(queries=[query])
            
            if search_results and search_results[0].results:
                formatted_results = []  
                for i, result in enumerate(search_results[0].results):  
                    # Appends a formatted string for each result, including its title, snippet, and URL.
                    formatted_results.append(f"Result {i+1}:\nTitle: {result.source_title}\nSnippet: {result.snippet}\nURL: {result.url}\n")
                return "Web search results:\n\n" + "\n".join(formatted_results)
            else:
                return "No relevant web search results found."
        except Exception as e:
            return f"Error during web search: {str(e)}"