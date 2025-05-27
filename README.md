# RAG Voice Agent

A sophisticated voice-enabled AI assistant that combines Retrieval Augmented Generation (RAG) with speech capabilities to provide intelligent, context-aware responses through both voice and text interactions.

## 🌟 Key Features

- **Voice Interaction**: Seamless voice input and output using advanced ASR (Automatic Speech Recognition) and TTS (Text-to-Speech) models
- **RAG-Powered Knowledge Base**: Intelligent document retrieval and response generation using vector database technology
- **Multi-Modal Interface**: Supports both voice and text-based interactions
- **Web Search Integration**: Capability to search the internet for up-to-date information
- **Conversational Memory**: Maintains context through conversation history
- **Extensible Tool System**: Modular architecture with support for adding new tools and capabilities

## 🛠️ Technology Stack

- **ASR Model**: Faster Whisper (Medium variant) for accurate speech recognition
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2) for document embeddings
- **LLM**: TinyLlama-1.1B-Chat for response generation
- **TTS Model**: Glow-TTS for natural speech synthesis
- **Vector Database**: Qdrant for efficient similarity search
- **Web Search**: DuckDuckGo integration for real-time information

## 🏗️ Architecture

The system consists of several key components:

1. **VoiceRAGAgent**: Core agent class that orchestrates all components and manages the interaction flow
2. **AudioHandler**: Manages voice input/output, including recording and speech synthesis
3. **TaskHandler**: Processes user queries and determines appropriate actions
4. **ComponentInitializer**: Handles initialization of all AI models and components
5. **Tools**: Modular system including:
   - SearchDocumentsTool: RAG knowledge base search
   - WebSearchTool: Internet search capability
   - SaveNoteTool: Note-taking functionality

## 🚀 Key Features In-Depth

### Voice Interaction
- Real-time voice input processing with automatic silence detection
- Natural-sounding speech output using advanced TTS
- Seamless switching between voice and text modes

### Knowledge Processing
- RAG-based document retrieval for accurate information access
- Web search integration for real-time information
- Context-aware response generation
- Conversation memory for maintaining context

### System Commands
- Voice input control ("voice" to start/stop)
- System status checks
- Memory management
- Tool listing and help commands

## 💡 Use Cases

1. **Knowledge Base Queries**: Access information from ingested documents with natural language
2. **Real-time Information**: Get updated information through web searches
3. **Interactive Conversations**: Engage in context-aware dialogue
4. **Voice-First Interaction**: Hands-free operation for various tasks

## 🔧 Implementation Details

### Data Ingestion
- PDF document processing with chunking
- Vector embedding generation
- Efficient storage in Qdrant vector database

### Query Processing
- Speech-to-text conversion
- Query understanding and routing
- Context-aware response generation
- Text-to-speech synthesis

## 🎯 Main Components

1. **main_agent.py**: Core agent implementation
2. **audio_handler.py**: Voice I/O management
3. **task_handler.py**: Query processing and routing
4. **Tools.py**: Implementation of various tools
5. **data_ingestion.py & rebuild_db.py**: Document processing and storage
6. **AgentState.py**: State management
7. **config.py**: System configuration

## 📝 Configuration

The system is highly configurable through `config.py`, allowing customization of:
- Model selections and parameters
- Audio processing settings
- Database configurations
- System behaviors and timeouts

## 🌟 Features in Development

- Enhanced multi-document support
- Improved context understanding
- Additional tool integrations
- Extended web search capabilities

## 👨‍💻 Developer

- **Sandeep** ([@sandeep231004](https://github.com/sandeep231004))
- Last Updated: 2025-05-27


This voice-enabled RAG agent represents a sophisticated approach to combining various AI technologies into a cohesive, interactive system that can process both voice and text inputs while providing intelligent, context-aware responses.
