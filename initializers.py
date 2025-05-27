import os 
import logging 
import torch 
import qdrant_client 
from qdrant_client.http import models 
from transformers import AutoModelForCausalLM, AutoTokenizer 
from faster_whisper import WhisperModel 
from sentence_transformers import SentenceTransformer
from TTS.api import TTS 

logger = logging.getLogger(__name__)

class ComponentInitializer:
    """
    Handles the initialization of all core components of the Voice RAG Agent,
    including ASR, Embedding Model, Vector Database, LLM, and TTS.
    """

    def __init__(self, config, embedding_model_instance=None):
        """
        Initializes the ComponentInitializer with the agent's configuration.
        config (dict): The configuration dictionary for the agent.
        """
        self.config = config
        self.asr_model = None
        self.embedding_model = embedding_model_instance
        self.vector_db = None
        self.llm = None
        self.tokenizer = None
        self.tts = None

        # Create necessary directories
        # os.makedirs(self.config["data_dir"], exist_ok=True)
        os.makedirs(self.config["db_path"], exist_ok=True)
        os.makedirs(self.config["tts_temp_dir"], exist_ok=True)

    def initialize_all(self):
        """Initializes all agent components."""
        logger.info("Initializing components...")
        # Calls individual initialization methods for each component.
        self._init_asr()
        self._init_embedding_model()
        self._init_llm()
        self._init_tts()

        logger.info("All components initialized successfully!")

    def _init_asr(self):
        """Initializes the ASR model using faster-whisper."""
        compute_type = 'float16' if self.config['use_gpu'] else 'int8' 
        device = 'cuda' if self.config['use_gpu'] else 'cpu'

        try:
            self.asr_model = WhisperModel(  
                self.config['asr_model'],
                device=device,
                compute_type=compute_type
            )
            logger.info(f"ASR model loaded: {self.config['asr_model']} on {device}")
        except Exception as e:
            logger.error(f"Failed to load ASR model: {str(e)}")
            raise

    def _init_embedding_model(self):
        """Initializes the Sentence Transformer model for embeddings."""
        if self.embedding_model is None:
            device = "cuda" if self.config["use_gpu"] else "cpu"
            try:
                self.embedding_model = SentenceTransformer(self.config["embedding_model"], device=device)
                logger.info(f"Embedding model loaded: {self.config['embedding_model']}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {str(e)}")
                raise

    def _init_llm(self):
        """Initializes the LLM for response generation."""
        logger.info(f"Loading LLM model: {self.config['llm_model']}") 
        
        try: 
            device = 0 if torch.cuda.is_available() and self.config["use_gpu"] else -1
            
            
            self.tokenizer = AutoTokenizer.from_pretrained( 
                self.config["llm_model"], 
                padding_side='left' # Set padding to the left, which is common for generation tasks.
            )
            
            if self.tokenizer.pad_token is None: 
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = { # Dictionary to hold keyword arguments for model loading. Prepares keyword arguments for loading the model:
                "torch_dtype": torch.float16 if self.config["use_gpu"] else torch.float32,
                "low_cpu_mem_usage": True,
            }
            
            if self.config["use_gpu"]: 
                model_kwargs["device_map"] = "auto"

            self.llm = AutoModelForCausalLM.from_pretrained(
                self.config["llm_model"], 
                **model_kwargs # Unpacks the model_kwargs dictionary as arguments.
            )
            
            logger.info(f"LLM model loaded: {self.config['llm_model']}")
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise

    def _init_tts(self):
        """Initializes the TTS model."""
        try: 
            self.tts = TTS(model_name=self.config["tts_model"]) 
            device = "cuda" if self.config["use_gpu"] else "cpu" # Determines device (GPU/CPU).
            self.tts.to(device) # Moves the TTS model to the specified device.
            logger.info(f"TTS model loaded: {self.config['tts_model']} on {device}")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            raise

    def get_components(self):
        """Returns a dictionary of all initialized components."""
        return {
            "asr_model": self.asr_model,
            "embedding_model": self.embedding_model,
            "llm": self.llm,
            "tokenizer": self.tokenizer,
            "tts": self.tts
        }
