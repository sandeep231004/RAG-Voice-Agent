import logging 
import json 
import re 
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TaskHandler:
    """
    Manages the agent's decision-making process, including:
    - Analyzing user intent.
    - Handling knowledge-seeking queries (RAG).
    - Handling task requests (tool orchestration).
    - Handling general conversation.
    - Handling system commands.
    """

    def __init__(self, config, llm_handler, tools, memory, agent_state, text_to_speech_callback):
        """
        Initializes the TaskHandler.
        """
        self.config = config 
        self.llm_handler = llm_handler
        self.tools = tools # Stores the dictionary of available tools.
        self.memory = memory # Stores the ConversationMemory instance.
        self.state = agent_state
        self.text_to_speech = text_to_speech_callback 
        self.agent_name = config["agent_name"]

    def handle_user_query(self, query):
        """
        Processes user input through the Agent's decision pipeline.
        This is the central dispatch method for user queries.
        """
        logger.info(f"Processing user query: {query}") 

        self.memory.add_message("user", query)

        self.state.set_current_task(query)

        # Parse intent and extract action needed.
        intent_result = self._analyze_user_intent(query) # Analyzes the user's query to determine their intent.
        intent = intent_result.get("intent", "knowledge_query") # Extracts the determined intent, defaulting to "knowledge_query".

        # Execute action based on intent.
        if intent == "knowledge_query": 
            response = self._handle_knowledge_query(query) 
        elif intent == "task_request": 
            response = self._handle_task_request(query) 
        elif intent == "conversation": 
            response = self._handle_conversation(query) 
        elif intent == "system_command": 
            response = self._handle_system_command(query) 
        else: 
            response = self._handle_conversation(query) 

        self.memory.add_message("assistant", response) 

        self.text_to_speech(response)

        return response

    def _analyze_user_intent(self, query) -> Dict[str, Any]:
        """
        Analyzes the user query to determine their intent and extract any relevant parameters
        using the LLM, with a simple rule-based fallback.
        """
        logger.info("Analyzing user intent...") 
        
        # Simple rule-based intent classification as backup for when LLM fails or for quick, obvious cases.
        query_lower = query.lower()
          # Default intent classification rules with improved keyword matching
        # First check for system commands as they are most specific
        if any(word in query_lower for word in [
            "stop", "exit", "quit", "clear", "delete", "status", "list tools", "help",
            "shutdown", "restart", "forget", "memory", "voice", "settings", "configure",
            "volume", "speed", "reload"
        ]):
            default_intent = "system_command"
        
        # Then check for task requests
        elif any(word in query_lower for word in [
            "save", "note", "write", "create", "add", "remember", "update", "modify",
            "change", "set", "do", "execute", "run", "perform", "make", "start", "begin",
            "search web", "look online", "find online", "search internet"
        ]):
            default_intent = "task_request"
        
        # Check for knowledge queries
        elif any(word in query_lower for word in [
            "search", "find", "look up", "what", "who", "when", "where", "why", "how",
            "knowledge base", "information", "tell me about", "explain", "describe", 
            "define", "meaning", "purpose", "reason", "difference", "compare", "which",
            "can you", "could you", "would you", "tell me", "show me", "give me",
            "is there", "are there", "do you know"
        ]) or any(phrase in query_lower for phrase in [
            "what is", "who is", "how do", "how does", "how can", "tell me about",
            "explain to me", "can you explain", "i want to know", "What'the", "Explain this"
        ]):
            default_intent = "knowledge_query"
        
        # Only fallback to conversation if no other intent matches
        else:
            default_intent = "conversation"
        
        system_prompt = f"""You are an AI assistant named {self.agent_name}. Your task is to analyze the user's query and determine their intent.
    Available intent categories:
    1. knowledge_query - The user is asking for information or knowledge
    2. task_request - The user wants you to perform a specific task or action (e.g., search the web, save a note)
    3. conversation - The user is engaging in general conversation
    4. system_command - The user is giving you a command about your system operations (e.g., clear memory, stop)

    Return your analysis as a JSON object with fields: "intent" and optionally "parameters" (a dictionary for task_request, or a string for knowledge_query if a specific search term is needed).
    Example for knowledge_query: {{"intent": "knowledge_query", "parameters": "history of AI"}}
    Example for task_request (web search): {{"intent": "task_request", "parameters": {{"tool": "web_search", "query": "latest news on AI"}} }}
    Example for conversation: {{"intent": "conversation"}}
    Example for system_command: {{"intent": "system_command", "command": "clear memory"}}"""

        prompt = f"""User query: "{query}"

    Analyze the intent and extract parameters.""" 

        result = self.llm_handler.generate_response(system_prompt, prompt) # Calls the LLM to perform intent analysis.
        
        try:
            # Try to extract JSON from the response (handles markdown code block format and direct JSON).
            json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL) # Tries to find JSON within a markdown code block.
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = re.search(r'\{.*\}', result, re.DOTALL).group(0) # Tries to find a direct JSON object in the string.
            
            intent_data = json.loads(json_str)
            logger.info(f"Successfully parsed intent: {intent_data}")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Could not parse intent analysis as JSON: {e}, using default")
            intent_data = {"intent": default_intent} # Falls back to the default intent if parsing fails.
        
        # Ensure the intent is one of the valid options.
        valid_intents = ["knowledge_query", "task_request", "conversation", "system_command"] 
        if "intent" not in intent_data or intent_data["intent"] not in valid_intents:
            logger.warning(f"Invalid intent '{intent_data.get('intent', 'unknown')}', using default")
            intent_data["intent"] = default_intent # Overrides with default intent if invalid.

        logger.info(f"Intent analysis: {intent_data}")
        return intent_data

    def _handle_knowledge_query(self, query):
        """
        Handles knowledge-seeking queries using a RAG (Retrieval Augmented Generation) approach.
        It searches the knowledge base and uses the retrieved context to answer the user's question.
        """
        logger.info("Handling knowledge query with RAG...")
        

        # Clean and preprocess the query to improve search relevance
        cleaned_query = ' '.join(query.strip().split())  # Normalize whitespace
        logger.info(f"Preprocessed query: {cleaned_query}")
            
        # Use document search tool with error handling
        if "search_documents" not in self.tools:
            logger.error("search_documents tool not available")
            return self._handle_conversation(query, include_disclaimer=True)
            
        tool = self.tools["search_documents"]
        
        # Execute search with the cleaned query
        search_results = tool.run(cleaned_query)
        self.state.record_tool_use("search_documents")
        
        # Log raw results for debugging
        logger.info(f"Raw search results: {str(search_results)[:200]}...")  # Log first 200 chars
        
        # Validate search results
        if not search_results:
            logger.warning("Search returned empty results")
            return self._handle_conversation(query, include_disclaimer=True)
                
        if isinstance(search_results, str):
            if search_results.strip() == "No relevant information found in the knowledge base.":
                logger.warning("Search explicitly returned no relevant information")
                return self._handle_conversation(query, include_disclaimer=True)
                
            if len(search_results.strip()) < 50:  # Minimum content threshold
                logger.warning(f"Search returned very short result: {search_results}")
                return self._handle_conversation(query, include_disclaimer=True)
                
        # Additional relevance checks
        if isinstance(search_results, str) and any(phrase in search_results.lower() for phrase in 
            ["no results", "no documents", "not found", "error occurred"]):
            logger.warning(f"Search results indicate an error or no results")
            return self._handle_conversation(query, include_disclaimer=True)
            
        # Process multipart results if applicable
        if isinstance(search_results, list):
            # Combine multiple results with clear separation
            search_results = "\n---\n".join([str(r) for r in search_results])
        
        # Get minimal conversation history for context
        recent_history = self._format_conversation_history(1)
        # Generate response using RAG with refined instructions for the LLM.
        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. Answer the user's question based on the provided context information.

    STRICT GUIDELINES:
    1. If the context contains the information needed, use it to give a precise answer.
    2. If the context is partially relevant, use what's relevant and say what information is missing.
    3. If the context doesn't contain relevant information, clearly state that you don't have the information.
    4. DO NOT make up or hallucinate information that isn't in the context.
    5. Keep your answer focused ONLY on the user's specific question.
    6. Use direct quotes from the context when possible to support your answer.
    7. If multiple documents are relevant, synthesize the information clearly.
    8. DO NOT ask follow-up questions.
    9. DO NOT make suggestions about other information the user might want.
    10. If you're uncertain about any information, express that uncertainty clearly."""
        prompt = f"""Context information:
```json
{search_results}
```

    Previous conversation:
    {recent_history}

    User question: "{query}" """ 
        response = self.llm_handler.generate_response(system_prompt, prompt)

        return response
    def _handle_task_request(self, query):
        """
        Handles a task or action request by using the LLM to select and execute the appropriate tool.
        This method relies on the LLM's ability to reason about tool usage.
        """
        logger.info(f"Handling task request: {query}")
        
        # Direct handling for time queries without using LLM
        if any(phrase in query.lower() for phrase in ["what time", "current time", "time in", "what's the time"]):
            location = None
            query_lower = query.lower()
            if "in " in query_lower:
                location = query_lower.split("in ")[-1].strip().strip('?').strip('.')
                if location:
                    logger.info(f"Detected time query for location: {location}")
                    tool = self.tools["web_search"]
                    result = tool.run(f"current exact time in {location}")
                    self.state.record_tool_use("web_search")
                    return self._format_time_response(result, location)
          # First check for direct mappings to improve reliability
        direct_mappings = {
            'news': ('web_search', 'latest news'),
            'weather': ('web_search', 'current weather'),
            'stock': ('web_search', 'stock market'),
            'sports': ('web_search', 'latest sports'),
            'ai': ('web_search', 'latest artificial intelligence news'),
            'tech': ('web_search', 'latest technology news')
        }

        # Check if query matches any direct mappings
        query_lower = query.lower()
        for key, (tool, base_query) in direct_mappings.items():
            if key in query_lower:
                logger.info(f"Using direct mapping for {key}")
                return self._execute_tool(tool, base_query)

        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. Analyze the user's request and respond with a valid JSON object ONLY.

    Available tools:
    {self._get_tools_description()}

    RESPONSE REQUIREMENTS:
    1. Your response MUST be a JSON object with exactly two fields: "tool" and "parameters"
    2. The "tool" field must be one of: "web_search" or "search_documents"
    3. The "parameters" field must be a string (the search query)

    TOOL SELECTION RULES:
    - Use "web_search" for:
      * Current events and news
      * Real-time information
      * Latest updates
      * Weather information
      * Time queries
      * Stock prices
      * Sports scores
    
    - Use "search_documents" for:
      * Historical information
      * Conceptual knowledge
      * Documentation
      * Stored information
      * Past events

    EXAMPLE RESPONSES:
    {{"tool": "web_search", "parameters": "latest artificial intelligence news and developments"}}
    {{"tool": "search_documents", "parameters": "history of quantum computing"}}
    {{"tool": "web_search", "parameters": "current stock market updates"}}"""# Defines system prompt for LLM to select tools.

        prompt = f"""User request: "{query}"

    Determine which tool to use and the parameters for it."""

        tool_selection = self.llm_handler.generate_response(system_prompt, prompt)
        try:
            # First try to parse the entire response as JSON
            try:
                tool_data = json.loads(tool_selection)
            except json.JSONDecodeError:
                # Try to extract JSON from code block
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', tool_selection, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON object in the text
                    json_str = re.search(r'\{[^{}]*\}', tool_selection, re.DOTALL)
                    if json_str:
                        json_str = json_str.group(0)
                    else:
                        raise ValueError("No JSON found in response")
                
                # Clean up the string and parse JSON
                json_str = json_str.strip()
                tool_data = json.loads(json_str)
            
            tool_name = tool_data.get("tool", "")
            parameters = tool_data.get("parameters", "")
            
            # For time-related queries, ensure we're using web search
            if any(word in query.lower() for word in ["time", "current time", "what time"]):
                tool_name = "web_search"
                if isinstance(parameters, str):
                    parameters = f"current time in {parameters}" if "time" not in parameters.lower() else parameters
            
            logger.info(f"LLM selected tool: {tool_name} with parameters: {parameters}")
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Could not parse tool selection as JSON: {e}")
            return "I'm having trouble understanding which action to take. Could you please be more specific about what you'd like me to do?"

        if tool_name in self.tools:
            logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")

            tool = self.tools[tool_name] 
            try: 
                if tool_name == "web_search" or tool_name == "search_documents":
                    result = tool.run(parameters)
                else:
                    if isinstance(parameters, dict):
                        result = tool.run(**parameters)
                    else:
                        result = tool.run(parameters)
                
                self.state.record_tool_use(tool_name)
                self.state.last_tool_result = result
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                result = f"I encountered an error when trying to use the '{tool_name}' tool: {str(e)}"
        else:
            available_tools = ", ".join(self.tools.keys())
            result = f"I don't have a tool named '{tool_name}'. Available tools are: {available_tools}"        # Generate a response incorporating the tool result.
        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. You've used a tool to help with the user's request.

    RESPONSE GUIDELINES:
    1. Format your response naturally and conversationally.
    2. For time queries: Extract and clearly state the current time from the search results.
    3. For weather queries: Focus on current temperature and conditions.
    4. For news or current events: Summarize the most recent information.
    5. Don't mention that you performed a web search unless necessary.
    6. Keep responses concise and directly focused on the user's question.
    7. If the search results are not relevant or don't contain the specific information asked for, say so clearly."""

        prompt = f"""User request: "{query}"

    Tool used: {tool_name}
    Tool result: {result}

    Create a helpful and natural-sounding response that incorporates the tool's result."""

        response = self.llm_handler.generate_response(system_prompt, prompt)
        return response

    def _handle_system_command(self, command):
        """
        Handles system commands directed at the agent.
        """
        logger.info(f"Handling system command: {command}") 
        
        command_lower = command.lower()
        
        if "clear memory" in command_lower or "forget conversation" in command_lower: 
            self.memory.clear_memory()
            return "I've cleared my memory of our conversation." 
        
        elif "stop" in command_lower or "shutdown" in command_lower or "turn off" in command_lower:
            self.state.set_active(False)
            return "I'm shutting down now. Goodbye!"

        elif "status" in command_lower or "how are you" in command_lower:
            status = self.state.get_status_summary()
            return f"I'm currently {'active' if status['active'] else 'inactive'}. My current task is: {status['current_task'] or 'None'}."

        elif "list tools" in command_lower or "what can you do" in command_lower:
            tools_desc = self._get_tools_description()
            return f"Here are the tools I can use to help you:\n{tools_desc}"

        elif "help" in command_lower:
            return f"""
I'm {self.agent_name}, your voice-activated AI assistant. Here's what I can do:

1. Answer questions using my knowledge base (documents you've added)
2. Search the web for information
3. Engage in general conversation

You can control me with commands like:
- "Voice" - Start voice input mode
- "Exit" or "Quit" - End the session 
- "Clear memory" - Forget our conversation
- "Status" - Check my current state
- "List tools" - See my available capabilities

How can I assist you today?
"""
        else:
            system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. The user has given you a command that doesn't match any predefined system commands.
Try to interpret what they want and explain how they can interact with you."""

            prompt = f"""The user command was: "{command}"

Respond helpfully, explaining what commands or questions you can handle.""" # Defines user-specific prompt for unknown commands.

            return self.llm_handler.generate_response(system_prompt, prompt) # Calls LLM to interpret and respond to unknown command.
        

    def _handle_conversation(self, query, include_disclaimer=False):
        """
        Handles general conversational input using the LLM directly.
        """
        logger.info("Handling conversation input")
        
        # Get limited conversation history for context to keep the LLM focused.
        recent_history = self._format_conversation_history(2)  # Gets the last 2 turns of conversation for context.
        
        # Generate conversational response with stricter instructions for the LLM.
        system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. Your task is to provide a DIRECT and CONCISE response to the user's input.

    IMPORTANT GUIDELINES:
    - Respond ONLY to what the user has explicitly asked.
    - Keep your response under 3 sentences unless detailed information is requested.
    - Do not ask follow-up questions at the end of your response.
    - Do not try to continue the conversation.
    - Do not make assumptions about what the user might want to know next.""" 

        prompt = f"""Recent conversation:
    {recent_history}

    Current user question: "{query}" """

        response = self.llm_handler.generate_response(system_prompt, prompt)
        
        if include_disclaimer:
            response += "\n\nI'm answering based on my general knowledge, as I couldn't find specific information about this in my knowledge base."

        return response

    def _format_conversation_history(self, max_turns=None):
        """
        Formats recent conversation history for inclusion in LLM prompts.
        """
        history = self.memory.get_recent_history(max_turns)
        formatted = [] 
        
        for item in history: 
            role = "User" if item["role"] == "user" else self.agent_name
            formatted.append(f"{role}: {item['content']}")

        return "\n".join(formatted)

    def _get_tools_description(self) -> str:
        """
        Generates a description of the available tools for the LLM.
        """
        descriptions = []
        for tool_name, tool in self.tools.items():
            descriptions.append(f"{tool_name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _format_time_response(self, search_result, location):
        """
        Formats the response for time-related queries.
        """
        if not search_result or "Error" in search_result:
            return f"I'm sorry, I couldn't find the current time in {location}. Please try again or rephrase your question."

        # Parse the response to find time information
        result_lower = search_result.lower()
        time_patterns = [
            r'(\d{1,2}:\d{2}(?:\s*[ap]m)?)',  # matches patterns like "3:45 PM" or "15:45"
            r'(\d{1,2}(?::\d{2})?\s*[ap]m)',   # matches patterns like "3 PM" or "3:45 PM"
            r'(\d{2}:\d{2}\s*(?:hours?)?)'     # matches 24-hour format like "15:45" or "15:45 hours"
        ]

        found_time = None
        for pattern in time_patterns:
            matches = re.findall(pattern, result_lower)
            if matches:
                found_time = matches[0]
                break

        if found_time:
            return f"The current time in {location} is {found_time}."
        else:
            logger.warning(f"Could not extract time from result: {search_result}")
            return f"I found information about {location}, but I couldn't determine the exact current time. Here's what I found:\n{search_result}"

    def _execute_tool(self, tool_name: str, parameters: str) -> str:
        """Helper method to execute a tool with given parameters."""
        try:
            if tool_name not in self.tools:
                available_tools = ", ".join(self.tools.keys())
                return f"I don't have a tool named '{tool_name}'. Available tools are: {available_tools}"

            tool = self.tools[tool_name]
            result = tool.run(parameters)
            self.state.record_tool_use(tool_name)
            self.state.last_tool_result = result

            # Generate response incorporating the tool result
            system_prompt = f"""You are {self.agent_name}, a helpful AI assistant. Create a clear and natural response based on the tool results.

RESPONSE GUIDELINES:
1. Be concise and direct
2. For news: Summarize the main points
3. For time: State the time clearly
4. For weather: Focus on current conditions
5. No need to mention that you used a tool
6. Don't ask follow-up questions
7. Keep it focused on the user's request"""

            prompt = f"""Tool result: {result}

Create a helpful response that naturally incorporates this information."""

            response = self.llm_handler.generate_response(system_prompt, prompt)
            return response

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return f"I encountered an error while trying to help you: {str(e)}"
