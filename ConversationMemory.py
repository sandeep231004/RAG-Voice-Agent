import os
import numpy as np
import json
import logging
from datetime import datetime
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationMemory:
    """
    Maintains short-term and long-term conversation memory with semantic embeddings.
    Supports relevant message retrieval based on similarity to user queries.
    """

    def __init__(self, max_history=10, embedding_model="all-MiniLM-L6-v2"):
        self.history = []  # Recent conversation messages
        self.long_term_memory = []  # Older messages moved from history
        self.max_history = max_history
        self.embeddings = {}  # Dictionary to store message embeddings by ID

        # Load sentence transformer model for semantic encoding
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.warning(f"Could not initialize embedding model: {str(e)}")
            self.embedding_model = None

    def add_message(self, role: str, content: str):
        """
        Add a new message to memory and generate its embedding.
        Older messages are moved to long-term memory once limit is reached.
        """
        msg_id = f"{datetime.now().timestamp()}-{len(self.history)}"
        message = {
            "id": msg_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.history.append(message)

        if self.embedding_model:
            try:
                self.embeddings[msg_id] = self.embedding_model.encode(content)
            except Exception as e:
                logger.error(f"Embedding generation failed: {str(e)}")

        if len(self.history) > self.max_history:
            self.long_term_memory.append(self.history.pop(0))

    def get_recent_history(self, max_turns: int = None) -> List[Dict]:
        """
        Returns recent messages up to `max_turns`.
        If `max_turns` is None, return full history.
        """
        if max_turns is None:
            return self.history
        return self.history[-min(max_turns, len(self.history)):]

    def get_relevant_history(self, query: str, top_k: int = 3, threshold: float = 0.6) -> List[Dict]:
        """
        Returns top_k most relevant messages based on cosine similarity with query embedding.
        Only messages above the similarity threshold are returned.
        """
        if not query or not self.embedding_model:
            return []

        try:
            query_embedding = self.embedding_model.encode(query)
            all_messages = self.history + self.long_term_memory
            relevant = []

            for msg in all_messages:
                msg_id = msg.get("id")
                if not msg_id or msg_id not in self.embeddings:
                    continue

                similarity = self._cosine_similarity(query_embedding, self.embeddings[msg_id])
                if similarity >= threshold:
                    relevant.append({"message": msg, "similarity": similarity})

            relevant.sort(key=lambda x: x["similarity"], reverse=True)
            return [r["message"] for r in relevant[:top_k]]

        except Exception as e:
            logger.error(f"Relevance search failed: {str(e)}")
            return []

    def save_memory(self, file_path: str) -> bool:
        """
        Save conversation messages and embeddings to disk.
        Used for persistence between sessions.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            memory_data = {
                "history": self.history,
                "long_term_memory": self.long_term_memory
            }
            with open(file_path, 'w') as f:
                json.dump(memory_data, f, indent=2)

            if self.embedding_model:
                with open(f"{file_path}.embeddings.json", 'w') as f:
                    json.dump({k: v.tolist() for k, v in self.embeddings.items()}, f)

            logger.info(f"Memory saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Saving memory failed: {str(e)}")
            return False

    def load_memory(self, file_path: str) -> bool:
        """
        Load messages and embeddings from disk.
        Rehydrates memory after restart or reload.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Memory file {file_path} does not exist.")
            return False

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.history = data.get("history", [])
                self.long_term_memory = data.get("long_term_memory", [])

            emb_path = f"{file_path}.embeddings.json"
            if os.path.exists(emb_path) and self.embedding_model:
                with open(emb_path, 'r') as f:
                    raw_embeddings = json.load(f)
                self.embeddings = {k: np.array(v) for k, v in raw_embeddings.items()}

            logger.info(f"Memory loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Loading memory failed: {str(e)}")
            return False

    def _cosine_similarity(self, vec1, vec2) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        return float(np.dot(vec1, vec2))
    

    def clear_memory(self):
        """Clear the conversation history."""
        self.history = []

    