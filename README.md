# Generic RAG System 🚀

A simple, flexible RAG system where you can use **ANY models you want**. Just change the model names!

## 🎯 What This Does

1. **`process_all_pdfs()`** - Processes all PDFs in the `pdfs/` folder and creates embeddings
2. **`retrieve_data(query)`** - Finds relevant documents based on your question
3. **`generate_response(query, docs)`** - Creates user-friendly answers using AI

## 🔧 Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Models
Edit the model settings at the top of `simple_rag_system.py`:

```python
# Embedding Model - Use ANY embedding model you want
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # HuggingFace model name
EMBEDDING_MODEL_TYPE = "huggingface"       # "huggingface" or "openai"

# Language Model - Use ANY language model you want  
LANGUAGE_MODEL_NAME = "llama3.2:1b"        # Model name
LANGUAGE_MODEL_TYPE = "ollama"             # "ollama", "openai", "huggingface"
```

### 3. Setup Your Models

#### For HuggingFace Models (FREE):
```bash
# Models download automatically on first use
# No additional setup needed!
```

#### For Ollama Models (FREE):
```bash
# Install Ollama from https://ollama.ai/
ollama pull llama3.2:1b  # or whatever model you want
ollama serve
```

#### For OpenAI Models (PAID):
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### 4. Add PDFs and Run
```bash
# Add PDF files to pdfs/ folder
python simple_rag_system.py
```

## 💡 Usage Example

```python
from simple_rag_system import SimpleRAGSystem

# Initialize system (uses whatever models you configured)
rag = SimpleRAGSystem()

# Process all PDFs (do this once)
rag.process_all_pdfs()

# Ask questions
query = "What is the main topic?"
docs = rag.retrieve_data(query)
answer = rag.generate_response(query, docs)

print(answer)
```

## 🎛️ Model Options

You can use **ANY** of these models by just changing the names:

### Embedding Models:
- **HuggingFace**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `sentence-transformers/all-roberta-large-v1`
- **OpenAI**: `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002`

### Language Models:
- **Ollama**: `llama3.2:1b`, `llama3.2:3b`, `phi3:mini`, `gemma2:2b`, `mistral:7b`
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- **HuggingFace**: `microsoft/DialoGPT-medium`, `google/flan-t5-base`

## 📁 File Structure

```
python-demo/
├── simple_rag_system.py    # Main system (configure models here)
├── requirements.txt        # All dependencies
├── .env                   # API keys (if needed)
├── pdfs/                  # Put your PDF files here
└── chroma_db/            # Vector database (auto-created)
```

## 🛠️ Configuration

All settings are at the top of `simple_rag_system.py`:

```python
# Model Configuration - Change these to use any models!
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_TYPE = "huggingface"
LANGUAGE_MODEL_NAME = "llama3.2:1b"
LANGUAGE_MODEL_TYPE = "ollama"

# Other Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVAL_RESULTS = 4
PDFS_FOLDER = "./pdfs"
DATABASE_FOLDER = "./chroma_db"
```

## 🎉 Ready to Use!

The system is completely generic - just change the model names to use whatever models you prefer!
- `openai` - OpenAI API client