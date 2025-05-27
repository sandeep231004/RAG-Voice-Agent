import torch

AGENT_CONFIG = {
    "asr_model": "medium",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", 
    "llm_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "tts_model": "tts_models/en/ljspeech/glow-tts", 
    "collection_name": "rag_docs", # The name of the collection/index in the Qdrant vector database for RAG documents.
    "top_k": 3,
    "temperature": 0.7,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "use_gpu": torch.cuda.is_available(),
    "audio_input_timeout": 5.0, # Float: Duration in seconds of silence after speech to automatically stop audio recording.
    "audio_sample_rate": 16000, # Integer: The sample rate (in Hz) for audio recording and playback.    "data_dir": "data",
    "db_path": "d:/Projects/RAG_Voice_Agent_Updated/vector_db", 
    "memory_size": 10, # Integer: The maximum number of recent messages to keep in the short-term conversation memory.
    "agent_name": "AURA", 
    "max_tokens": 512, # Integer: The maximum number of new tokens the LLM can generate in a single response.
    "user_notes_dir": "user_notes",
    "user_files_dir": "user_files",
    "tts_temp_dir": "tts_temp"
}

