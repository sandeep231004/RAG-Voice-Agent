import logging
import time
import os
from config import AGENT_CONFIG
from qdrant_client import QdrantClient
from initializers import ComponentInitializer
from Tools import SearchDocumentsTool, WebSearchTool
from AgentState import AgentState
from ConversationMemory import ConversationMemory
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from task_handler import TaskHandler

# --- Global Tool Placeholder (for WebSearchTool) ---
# In a real application, you would initialize a specific web search client here,
# e.g., from a library that interfaces with Google Search API, DuckDuckGo API, etc.
# For this demo, we'll assume 'google_search' is available in the environment
# as a mock or actual search utility.
# If running locally without a real search client, you'd need to mock this or
# integrate a specific API.
class MockGoogleSearch:
    def search(self, queries: list):
        logger.info(f"Mocking web search for query: {queries[0]}")
        class MockResult:
            def __init__(self, title, snippet, url):
                self.source_title = title
                self.snippet = snippet
                self.url = url

        class MockSearchResults:
            def __init__(self, query, results):
                self.query = query
                self.results = results

        return [MockSearchResults(
            query=queries[0],
            results=[
                MockResult("Mock Search Result 1", "This is a snippet from a mock web search result.", "https://example.com/mock1"),
                MockResult("Mock Search Result 2", "Another mock snippet providing some information.", "https://example.com/mock2")
            ]
        )]

google_search = MockGoogleSearch()

logger = logging.getLogger(__name__)



class VoiceRAGAgent:
    def __init__(self, config=None):
        self.config = AGENT_CONFIG.copy()
        if config:
            self.config.update(config)

        self.agent_name = self.config["agent_name"]

        # Initialize state and memory
        self.state = AgentState()
        self.memory = ConversationMemory(max_history=self.config["memory_size"])

        # Initialize core models except vector DB
        self.initializer = ComponentInitializer(self.config, embedding_model_instance=self.memory.embedding_model)
        self.initializer.initialize_all()

        # Get initialized components
        components = self.initializer.get_components()
        self.asr_model = components["asr_model"]
        self.embedding_model = components["embedding_model"]
        self.llm = components["llm"]
        self.tokenizer = components["tokenizer"]
        self.tts = components["tts"]

        # ✅ Initialize QdrantClient without recreating collection
        self.vector_db = QdrantClient(path=self.config["db_path"])
        logger.info("Connected to existing Qdrant vector database.")

        # Initialize tools and handlers
        self.tools = self._initialize_tools()

        self.llm_handler = LLMHandler(self.config, self.llm, self.tokenizer)
        self.audio_handler = AudioHandler(self.config, self.asr_model, self.tts, self.handle_user_query)
        self.task_handler = TaskHandler(
            self.config,
            self.llm_handler,
            self.tools,
            self.memory,
            self.state,
            self.audio_handler.text_to_speech
        )

        logger.info(f"{self.agent_name} initialization complete.")

    def _initialize_tools(self):
        """Initializes agent tools (SearchDocumentsTool, WebSearchTool)."""
        tools_dict = {
            "search_documents": SearchDocumentsTool(
                self.vector_db,
                self.embedding_model,
                self.config["collection_name"],
                self.config["top_k"]
            ),
            "web_search": WebSearchTool(google_search),
        }
        logger.info(f"Initialized {len(tools_dict)} tools")
        return tools_dict

    def handle_user_query(self, query):
        """
        Processes user input (from ASR or text) by delegating to the TaskHandler.
        This method serves as a bridge for the AudioHandler to send transcribed text.
        """
        return self.task_handler.handle_user_query(query)

    def process_text_input(self, text):
        """
        Processes text input directly from the user (e.g., from console).
        """
        logger.info(f"Processing text input: {text}")
        self.state.set_active(True)

        response = self.handle_user_query(text)

        return response

    def run_interactive(self):
        """Runs the agent in interactive mode, allowing text or voice input."""
        self.state.set_active(True)
        print(f"===== {self.agent_name} Interactive Mode =====")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'voice' to start voice input mode")
        print("Type 'help' for more commands")

        while self.state.active:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ['exit', 'quit']:
                    print(f"\n{self.agent_name}: Goodbye!")
                    self.state.set_active(False)
                    break

                elif user_input.lower() == 'voice':
                    print(f"\n{self.agent_name}: {self.audio_handler.start_listening(self.state)}")
                    continue

                response = self.process_text_input(user_input)
                print(f"\n{self.agent_name}: {response}")

            except KeyboardInterrupt:
                print("\nDetected keyboard interrupt. Shutting down...")
                self.state.set_active(False)
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"\nError: {str(e)}")

# Example of how to run the agent
if __name__ == "__main__":
    agent = VoiceRAGAgent() # Creates an instance of the VoiceRAGAgent with default or provided config.
    agent.run_interactive() # Starts the interactive session for the agent.
