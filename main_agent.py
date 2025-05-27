import logging
import time
import os
import re
from datetime import datetime
import pytz
from config import AGENT_CONFIG
from qdrant_client import QdrantClient
from initializers import ComponentInitializer
from Tools import SearchDocumentsTool, WebSearchTool
from AgentState import AgentState
from ConversationMemory import ConversationMemory
from audio_handler import AudioHandler
from llm_handler import LLMHandler
from task_handler import TaskHandler


from duckduckgo_search import DDGS

class DuckDuckGoSearch:
    def __init__(self):
        self.ddgs = DDGS()    
    def search(self, queries: list):
        logger.info(f"Performing web search for query: {queries[0]}")
        class SearchResult:
            def __init__(self, title, snippet, url):
                self.source_title = title
                self.snippet = snippet
                self.url = url

        class SearchResults:
            def __init__(self, query, results):
                self.query = query
                self.results = results

        results = []
        try:
            query = queries[0]            # Handle time queries directly using pytz
            if any(word in query.lower() for word in ["time", "current time", "what time"]):
                try:
                    # Extract location from query and clean it
                    location = query.lower()
                    if "in" in location:
                        location = location.split("in")[-1]
                    location = location.strip().strip('?').strip('.').strip()
                    # Remove common words that might be attached to the location
                    location = location.replace("right now", "").replace("now", "").strip()
                    
                    # Map common city names to timezone names
                    timezone_mapping = {
                        'london': 'Europe/London',
                        'new york': 'America/New_York',
                        'paris': 'Europe/Paris',
                        'tokyo': 'Asia/Tokyo',
                        # Add more mappings as needed
                    }
                    
                    # Get the timezone
                    tz_name = timezone_mapping.get(location.lower())
                    if tz_name:
                        # Get current time in the specified timezone
                        tz = pytz.timezone(tz_name)
                        current_time = datetime.now(tz)
                        formatted_time = current_time.strftime("%I:%M %p")  # 12-hour format with AM/PM
                        formatted_date = current_time.strftime("%A, %B %d, %Y")
                        
                        results.append(SearchResult(
                            title=f"Current Time in {location.title()}",
                            snippet=f"The current time in {location.title()} is {formatted_time} on {formatted_date}",
                            url=""
                        ))
                        return [SearchResults(query=queries[0], results=results)]
                    
                except Exception as e:
                    logger.error(f"Error getting time for location {location}: {str(e)}")
                    # Don't continue with web search, return error message
                    results.append(SearchResult(
                        title="Error",
                        snippet=f"Sorry, I encountered an error while getting the time for {location}. Please try again.",
                        url=""
                    ))
                    return [SearchResults(query=queries[0], results=results)]
              # Get results from DuckDuckGo
            ddg_results = []
            try:
                # For time queries, try time.is website first
                if any(word in query.lower() for word in ["time", "current time", "what time"]):
                    location = query.lower().split("in")[-1].strip()
                    ddg_results = list(self.ddgs.text(f"site:time.is current time in {location}", max_results=2))
                
                # If no results, try general search
                if not ddg_results:
                    ddg_results = list(self.ddgs.text(query, max_results=5))
                
                # For time queries, filter for relevant results
                if any(word in query.lower() for word in ["time", "current time", "what time"]):
                    filtered_results = [r for r in ddg_results if any(word in r.get('body', '').lower() for word in ["current time", "local time", "time now", "current local time"])]
                    if filtered_results:
                        ddg_results = filtered_results[:2]
            except Exception as e:
                logger.error(f"Error during DuckDuckGo search query: {str(e)}")
                ddg_results = []
            
            for r in ddg_results:                # Process and clean up the result
                try:
                    title = r.get('title', 'No Title')
                    body = r.get('body', '')
                    link = r.get('link', '')
                    
                    # For time queries, try to extract and format time information
                    if "time" in query.lower():
                        time_patterns = [
                            r'(\d{1,2}:\d{2}(?:\s*[ap]m)?)',  # 3:45 PM or 15:45
                            r'(\d{1,2}(?::\d{2})?\s*[ap]m)',   # 3 PM or 3:45 PM
                            r'(\d{2}:\d{2}\s*(?:hours?)?)',    # 15:45 or 15:45 hours
                            r'(\d{1,2}(?::\d{2})?\s*(?:hrs?)?)'  # 15:45 hr
                        ]
                        
                        for pattern in time_patterns:
                            time_match = re.search(pattern, body, re.IGNORECASE)
                            if time_match:
                                body = f"Current time is {time_match.group(1)}"
                                break
                        
                        # If no time found in body, try title
                        if "Current time is" not in body:
                            for pattern in time_patterns:
                                time_match = re.search(pattern, title, re.IGNORECASE)
                                if time_match:
                                    body = f"Current time is {time_match.group(1)}"
                                    break
                    
                    results.append(SearchResult(
                        title=title,
                        snippet=body,
                        url=link
                    ))
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {str(e)}")
            # Return at least one result even on error
            results.append(SearchResult(
                title="Error",
                snippet=f"Failed to perform web search: {str(e)}",
                url=""
            ))

        return [SearchResults(query=queries[0], results=results)]

search_client = DuckDuckGoSearch()

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
            "web_search": WebSearchTool(search_client),
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
