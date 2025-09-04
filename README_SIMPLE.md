# 🤖 Simplified Agentic RAG System

A concise, readable agentic RAG system that enhances your existing `simple_rag_system.py` with intelligent agents and Wikipedia integration.

## 🎯 What's New?

Your original RAG system now has a **simplified agentic enhancement**:

- **🤖 Smart Query Processing**: Automatic intent analysis and query expansion
- **🌐 Wikipedia Integration**: External knowledge without complexity
- **💾 Conversation Memory**: Simple context tracking
- **🔄 Fallback System**: Graceful degradation to your original system

## 📁 Simplified Structure

```
python-demo/
├── simple_rag_system.py     # Your original RAG (unchanged)
├── agentic_rag.py          # New simplified agentic system
├── requirements.txt        # Dependencies
├── pdfs/                   # Your PDF documents
├── chroma_db/             # Vector database
└── agents/, tools/, memory/, config/  # Empty folders for future extensions
```

## 🚀 Quick Start

### Basic Usage

```python
from agentic_rag import AgenticRAG

# Initialize with agentic features
rag = AgenticRAG(use_agents=True, use_wikipedia=True)

# Process your PDFs (same as before)
rag.process_pdfs()

# Enhanced query with agents + Wikipedia
result = rag.query("What is JavaScript?")
print(result['response'])
print(f"Confidence: {result['confidence']}")
print(f"Sources: {result['sources']}")
```

### Simple Mode (Original Functionality)

```python
# Use without agents (same as your original system)
rag = AgenticRAG(use_agents=False)
result = rag.query("Your question")
```

### Run Directly

```bash
python agentic_rag.py
```

## 🤖 Agentic Features

### Intelligent Query Processing
- **Intent Analysis**: Automatically detects explanation, instruction, comparison queries
- **Query Expansion**: Expands queries for better retrieval (e.g., "JavaScript" → "JavaScript tutorial", "JavaScript guide")
- **Multi-source Retrieval**: Combines local documents + Wikipedia knowledge

### Wikipedia Integration
- **Real-time Search**: Live Wikipedia article retrieval
- **Smart Summarization**: Extracts relevant 3-sentence summaries
- **Graceful Fallback**: Works without internet connection

### Conversation Memory
- **Context Tracking**: Remembers recent queries and responses
- **Session State**: Maintains conversation flow

## 📊 Capabilities Comparison

| Feature | Simple RAG | Agentic RAG |
|---------|------------|-------------|
| **Local Documents** | ✅ | ✅ |
| **External Knowledge** | ❌ | ✅ Wikipedia |
| **Query Understanding** | ❌ | ✅ Intent analysis |
| **Query Expansion** | ❌ | ✅ Smart expansion |
| **Context Memory** | ❌ | ✅ Conversation tracking |
| **Fallback Support** | ❌ | ✅ Graceful degradation |
| **Code Complexity** | Simple | **Still Simple!** |

## 🔧 Configuration

All configuration is handled through simple parameters:

```python
# Full customization
rag = AgenticRAG(
    use_agents=True,        # Enable agentic features
    use_wikipedia=True      # Enable Wikipedia integration
)

# Enable/disable dynamically
rag.disable_agents()  # Switch to simple RAG
rag.enable_agents()   # Re-enable agentic features
```

## 📈 Example Output

```python
result = rag.query("What is JavaScript?")

# Result structure:
{
    "response": "JavaScript is a dynamic programming language...",
    "confidence": 0.9,
    "sources": ["javascript_tutorial.pdf", "JavaScript", "JavaScript library"],
    "mode": "agentic",
    "intent": "explanation",
    "local_count": 4,
    "external_count": 2,
    "total_sources": 6
}
```

## 🛡️ Error Handling

The system includes robust error handling:
- **Agent Failure**: Automatically falls back to simple RAG
- **Wikipedia Unavailable**: Uses local documents only
- **Network Issues**: Continues with available sources

## 🎯 Key Simplifications

1. **Single File**: All agentic functionality in one readable file (`agentic_rag.py`)
2. **No Complex Classes**: Simple functions and minimal abstractions
3. **Integrated Components**: Wikipedia, memory, and agents built-in
4. **Clear Logic Flow**: Easy to understand and modify
5. **Preserved Functionality**: All capabilities maintained with less code

## 🚀 Future Extensions

The folder structure is preserved for easy extensions:
- Add new agents in `agents/`
- Add new tools in `tools/`
- Add memory enhancements in `memory/`
- Add configurations in `config/`

## 📝 Dependencies

Same as before, plus:
```bash
pip install wikipedia
```

---

**🎉 Result**: You now have a **simplified, readable agentic RAG system** that maintains all the intelligence while being much easier to understand and modify!
