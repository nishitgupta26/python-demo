"""
Simple RAG System for Beginners
===============================

A simple, flexible RAG system where you can use ANY models you want:
1. Processes all PDFs in the 'pdfs' folder
2. Creates embeddings and stores them in a vector database
3. Retrieves relevant information based on user queries
4. Generates user-friendly responses

Just change the model names below to use whatever models you prefer!
"""

import os
import glob
import logging
from typing import List, Dict, Any
from pathlib import Path

# PDF processing
import PyPDF2

# LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# 🔧 MODEL CONFIGURATION - CHANGE THESE TO USE ANY MODELS YOU WANT
# =============================================================================

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model name
EMBEDDING_MODEL_TYPE = "huggingface"       # "huggingface" or "openai"

# Language Model Configuration  
LANGUAGE_MODEL_NAME = "llama3.2:1b"        # Model name (Ollama, OpenAI, etc.)
LANGUAGE_MODEL_TYPE = "ollama"             # "ollama", "openai", "huggingface"

# Text Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
NUM_RETRIEVAL_RESULTS = 4

# Folder Settings
PDFS_FOLDER = "./pdfs"
DATABASE_FOLDER = "./chroma_db"

# =============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleRAGSystem:
    """
    A generic RAG system that works with any models you specify.
    
    Main components:
    1. PDF processor - reads and chunks PDF files
    2. Vector database - stores document embeddings
    3. Retriever - finds relevant documents
    4. Response generator - creates user-friendly answers
    """
    
    def __init__(self):
        """Initialize the RAG system with your specified models."""
        logger.info(f"🚀 Initializing Generic RAG System...")
        logger.info(f"   📊 Embedding: {EMBEDDING_MODEL_NAME} ({EMBEDDING_MODEL_TYPE})")
        logger.info(f"   🤖 Language: {LANGUAGE_MODEL_NAME} ({LANGUAGE_MODEL_TYPE})")
        
        # Create folders
        os.makedirs(PDFS_FOLDER, exist_ok=True)
        os.makedirs(DATABASE_FOLDER, exist_ok=True)
        
        # Initialize components
        self._setup_text_splitter()
        self._setup_embedding_model()
        self._setup_language_model()
        self._setup_vector_database()
        
        logger.info("✅ RAG System initialized successfully!")
    
    def _setup_text_splitter(self):
        """Set up the text splitter."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        logger.info(f"📝 Text splitter configured (chunk_size={CHUNK_SIZE})")
    
    def _setup_embedding_model(self):
        """Set up embedding model based on the type you specified."""
        logger.info(f"🔢 Setting up embedding model: {EMBEDDING_MODEL_NAME}")
        
        if EMBEDDING_MODEL_TYPE == "huggingface":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info(f"✅ HuggingFace embedding model loaded: {EMBEDDING_MODEL_NAME}")
            except ImportError:
                logger.error("❌ Install HuggingFace dependencies: pip install sentence-transformers")
                raise
                
        elif EMBEDDING_MODEL_TYPE == "openai":
            try:
                from langchain_openai import OpenAIEmbeddings
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("❌ Set OPENAI_API_KEY in .env file for OpenAI models")
                
                self.embeddings = OpenAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    openai_api_key=api_key
                )
                logger.info(f"✅ OpenAI embedding model configured: {EMBEDDING_MODEL_NAME}")
            except ImportError:
                logger.error("❌ Install OpenAI dependencies: pip install langchain-openai")
                raise
        else:
            raise ValueError(f"❌ Unsupported embedding model type: {EMBEDDING_MODEL_TYPE}")
    
    def _setup_language_model(self):
        """Set up language model based on the type you specified."""
        logger.info(f"🤖 Setting up language model: {LANGUAGE_MODEL_NAME}")
        
        if LANGUAGE_MODEL_TYPE == "ollama":
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=LANGUAGE_MODEL_NAME,
                    temperature=0.0,
                )
                
                # Test connection
                test_response = self.llm.invoke("Hello")
                logger.info(f"✅ Ollama model connected: {LANGUAGE_MODEL_NAME}")
            except Exception as e:
                logger.error(f"❌ Ollama connection failed: {e}")
                raise
                
        elif LANGUAGE_MODEL_TYPE == "openai":
            try:
                from langchain_openai import ChatOpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("❌ Set OPENAI_API_KEY in .env file for OpenAI models")
                
                self.llm = ChatOpenAI(
                    model=LANGUAGE_MODEL_NAME,
                    temperature=0.1,
                    openai_api_key=api_key
                )
                logger.info(f"✅ OpenAI language model configured: {LANGUAGE_MODEL_NAME}")
            except ImportError:
                logger.error("❌ Install OpenAI dependencies: pip install langchain-openai")
                raise
                
        elif LANGUAGE_MODEL_TYPE == "huggingface":
            try:
                from langchain_community.llms import HuggingFacePipeline
                from transformers import pipeline
                
                pipe = pipeline("text-generation", 
                              model=LANGUAGE_MODEL_NAME, 
                              max_new_tokens=512,
                              device_map="auto")
                self.llm = HuggingFacePipeline(pipeline=pipe)
                logger.info(f"✅ HuggingFace language model loaded: {LANGUAGE_MODEL_NAME}")
            except ImportError:
                logger.error("❌ Install HuggingFace dependencies: pip install transformers torch")
                raise
        else:
            raise ValueError(f"❌ Unsupported language model type: {LANGUAGE_MODEL_TYPE}")
    
    def _setup_vector_database(self):
        """Set up ChromaDB vector database."""
        try:
            chroma_client = chromadb.PersistentClient(
                path=DATABASE_FOLDER,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.vector_store = Chroma(
                client=chroma_client,
                collection_name="pdf_documents",
                embedding_function=self.embeddings,
                persist_directory=DATABASE_FOLDER
            )
            
            logger.info("🗄️ Vector database configured (ChromaDB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup vector database: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        logger.info(f"📖 Reading PDF: {Path(pdf_path).name}")
        
        try:
            text_content = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text_content += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"⚠️ Could not read page {page_num + 1}: {e}")
                        continue
            
            if not text_content.strip():
                raise ValueError("No text found in PDF")
            
            logger.info(f" Extracted {len(text_content)} characters")
            return text_content.strip()
            
        except Exception as e:
            logger.error(f" Error reading {pdf_path}: {e}")
            raise
    
    def process_all_pdfs(self) -> bool:
        """
        Process all PDFs in the pdfs folder and create embeddings.
        
        This function:
        1. Finds all PDF files in the pdfs folder
        2. Extracts text from each PDF
        3. Splits text into chunks
        4. Creates embeddings for each chunk
        5. Stores embeddings in the vector database
        
        Returns:
            True if successful, False otherwise
        """
        logger.info(f" Looking for PDFs in: {PDFS_FOLDER}")
        
        # Find all PDF files
        pdf_files = glob.glob(os.path.join(PDFS_FOLDER, "*.pdf"))
        
        if not pdf_files:
            logger.warning(f" No PDF files found in {PDFS_FOLDER}")
            logger.info(f" Please add PDF files to the '{PDFS_FOLDER}' folder")
            return False
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        successful_files = []
        failed_files = []
        
        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                # Extract text from PDF
                text_content = self._extract_text_from_pdf(pdf_file)
                
                # Split text into chunks
                text_chunks = self.text_splitter.split_text(text_content)
                
                # Create Document objects with metadata
                filename = Path(pdf_file).name
                for i, chunk in enumerate(text_chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_file,
                            "filename": filename,
                            "chunk_id": i,
                            "total_chunks": len(text_chunks)
                        }
                    )
                    all_documents.append(doc)
                
                successful_files.append(filename)
                logger.info(f" Processed {filename}: {len(text_chunks)} chunks created")
                
            except Exception as e:
                failed_files.append(Path(pdf_file).name)
                logger.error(f" Failed to process {Path(pdf_file).name}: {e}")
        
        # Store all documents in vector database
        if all_documents:
            try:
                logger.info(f" Storing {len(all_documents)} document chunks...")
                self.vector_store.add_documents(all_documents)
                
                logger.info(f" Successfully processed {len(successful_files)} PDFs!")
                logger.info(f" Total document chunks stored: {len(all_documents)}")
                
                if failed_files:
                    logger.warning(f" Failed to process: {failed_files}")
                
                return True
                
            except Exception as e:
                logger.error(f" Failed to store documents in database: {e}")
                return False
        else:
            logger.error(" No documents were successfully processed")
            return False
    
    def retrieve_data(self, query: str, num_results: int = None) -> List[Document]:
        """
        Retrieve relevant documents from the vector database.
        
        Args:
            query: User's question or search query
            num_results: Number of relevant documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if num_results is None:
            num_results = NUM_RETRIEVAL_RESULTS
            
        logger.info(f" Searching for relevant documents for: '{query}'")
        
        try:
            # relevant_docs = self.vector_store.similarity_search(
            #     query=query,
            #     k=num_results
            # )

            # Max Marginal Relevance Search
            relevant_docs = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=num_results,       # how many final docs you want
                fetch_k=20,          # how many to fetch before filtering redundancy
                lambda_mult=0.7      # balance relevance (1.0) vs diversity (0.0)
            )
            
            logger.info(f" Found {len(relevant_docs)} relevant document chunks")
            
            # Log which documents were found
            for i, doc in enumerate(relevant_docs):
                filename = doc.metadata.get('filename', 'Unknown')
                chunk_id = doc.metadata.get('chunk_id', 'Unknown')
                logger.info(f"  {i+1}. {filename} (chunk {chunk_id})")
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f" Error retrieving documents: {e}")
            return []
    
    def generate_response(self, query: str, relevant_docs: List[Document]) -> str:
        """
        Generate a user-friendly response based on the query and relevant documents.
        
        Args:
            query: User's original question
            relevant_docs: List of relevant documents from retrieve_data()
            
        Returns:
            User-friendly response string
        """
        logger.info(f" Generating response using {LANGUAGE_MODEL_NAME}...")
        
        if not relevant_docs:
            return " I couldn't find any relevant information in your PDFs to answer that question."
        
        try:
            # Create context from relevant documents
            context_parts = []
            for i, doc in enumerate(relevant_docs):
                filename = doc.metadata.get('filename', 'Unknown document')
                context_parts.append(f"--- Document {i+1}: {filename} ---")
                context_parts.append(doc.page_content)
                context_parts.append("")  # Empty line for readability
            
            context = "\n".join(context_parts)
            
            # Create an enhanced prompt for better GenAI responses
            prompt = f"""You are a knowledgeable and friendly assistant.
Use the context below to answer the user's question.

CONTEXT:
{context}

USER QUESTION: {query}

Guidelines:
- Answer ONLY using the context above
- If the context doesn't contain the answer, say "I couldn’t find that in the documents"
- Write in clear, conversational style
- Use bullet points or short paragraphs
- Be concise, but complete

Final Answer::"""
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            logger.info(" Response generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f" Error generating response: {e}")
            return f" I encountered an error while generating the response: {e}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system configuration."""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "status": "ready",
                "document_count": count,
                "embedding_model": f"{EMBEDDING_MODEL_NAME} ({EMBEDDING_MODEL_TYPE})",
                "language_model": f"{LANGUAGE_MODEL_NAME} ({LANGUAGE_MODEL_TYPE})",
                "pdfs_folder": PDFS_FOLDER,
                "database_folder": DATABASE_FOLDER,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            }
        except Exception as e:
            logger.error(f" Error getting system info: {e}")
            return {"status": "error", "error": str(e)}


def main():
    """
    Main function demonstrating how to use the Generic RAG System.
    """
    # Show current configuration
    print(f"🔧 Current Configuration:")
    print(f"    Embedding: {EMBEDDING_MODEL_NAME} ({EMBEDDING_MODEL_TYPE})")
    print(f"    Language: {LANGUAGE_MODEL_NAME} ({LANGUAGE_MODEL_TYPE})")
    print(f"    PDFs: {PDFS_FOLDER}")
    print(f"    Database: {DATABASE_FOLDER}")
    print()
    
    try:
        # Initialize the RAG system
        rag = SimpleRAGSystem()
        
        # Check current system status
        info = rag.get_system_info()
        print(f" System Status:")
        print(f"   Status: {info['status']}")
        print(f"   Documents: {info['document_count']}")
        print(f"   Embedding: {info['embedding_model']}")
        print(f"   Language: {info['language_model']}")
        
        # Process all PDFs
        print(f"\n Processing PDFs...")
        rag.process_all_pdfs()
        
        print(" PDF processing completed!")
        
        # Update info
        info = rag.get_system_info()
        print(f" Total documents: {info['document_count']}")
        
        # Example query
        user_query = "What is Javascript?"
        
        print(f"\n Example Query: '{user_query}'")
        print(" Retrieving relevant documents...")
        
        # Retrieve and generate response
        relevant_documents = rag.retrieve_data(user_query)
        
        print(" Generating response...")
        response = rag.generate_response(user_query, relevant_documents)
        
        print(f"\n Response:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
    
    except Exception as e:
        logger.error(f" Error: {e}")
        print(f" Error: {e}")
        print(f"\n Current setup uses:")
        print(f"   {EMBEDDING_MODEL_NAME} ({EMBEDDING_MODEL_TYPE})")
        print(f"   {LANGUAGE_MODEL_NAME} ({LANGUAGE_MODEL_TYPE})")


if __name__ == "__main__":
    main()