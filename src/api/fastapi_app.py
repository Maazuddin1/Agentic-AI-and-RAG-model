# src/api/fastapi_app.py
from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import logging
import os
import json
import time
from typing import Dict, Any, List, Optional

from src.data_retrieval.fetcher import PaperFetcher
from src.data_retrieval.pdf_parser import PdfParser
from src.indexing.embeddings import DocumentEmbedder, VectorStore
from src.summarization.llm_handler import LlamaHandler
from src.memory.user_memory import UserMemory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    sources: List[str] = ["arxiv", "google_scholar"]

class SummarizeRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    message: str
    query: str
    status: str

class ResearchAssistantAPI:
    """
    API for the Research Assistant.
    """
    
    def __init__(self):
        """
        Initialize the Research Assistant API.
        """
        logger.info("Initializing ResearchAssistantAPI")
        
        # Initialize components
        self.paper_fetcher = PaperFetcher(max_results=10)
        self.pdf_parser = PdfParser()
        self.document_embedder = DocumentEmbedder()
        self.vector_store = VectorStore(dimension=self.document_embedder.dimension)
        self.llama_handler = None  # Initialized on demand
        self.user_memory = UserMemory()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Research Assistant API",
            description="API for searching and summarizing research papers",
            version="1.0.0"
        )
        self.setup_routes()
        
        # Create data directories
        os.makedirs('data/indexes', exist_ok=True)
        os.makedirs('data/memory', exist_ok=True)
        
        # Load existing data if available
        self.load_data()
        
        logger.info("ResearchAssistantAPI initialized")
    
    def setup_routes(self):
        """
        Set up FastAPI routes.
        """
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy"}
        
        @self.app.post("/search", response_model=SearchResponse)
        async def search_papers(
            search_request: SearchRequest,
            background_tasks: BackgroundTasks,
            x_user_id: str = Header(default="anonymous")
        ):
            if not search_request.query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            # Add to user memory
            self.user_memory.add_query(x_user_id, search_request.query)
            
            # Start search in background
            background_tasks.add_task(
                self.background_search,
                search_request.query,
                search_request.sources,
                x_user_id
            )
            
            return {
                "message": "Search started",
                "query": search_request.query,
                "status": "processing"
            }
        
        @self.app.get("/results/{user_id}")
        async def get_results(user_id: str):
            results = self.user_memory.get_results(user_id)
            if results:
                return results
            else:
                raise HTTPException(status_code=404, detail="No results available")
        
        @self.app.post("/summarize")
        async def summarize(
            summarize_request: SummarizeRequest,
            x_user_id: str = Header(default="anonymous")
        ):
            if not summarize_request.query:
                raise HTTPException(status_code=400, detail="Query is required")
            
            # Get results from memory
            results = self.user_memory.get_results(x_user_id)
            if not results or not results.get('papers'):
                raise HTTPException(status_code=404, detail="No search results available")
            
            # Initialize Llama handler if needed
            if self.llama_handler is None:
                self.initialize_llama()
            
            # Generate summary
            summary = self.llama_handler.generate_summary(results['papers'], summarize_request.query)
            
            # Generate citations
            citations = self.llama_handler.generate_citations(results['papers'])
            
            # Update user memory
            self.user_memory.add_summary(x_user_id, summarize_request.query, summary, citations)
            
            return {
                "summary": summary,
                "citations": citations,
                "query": summarize_request.query
            }
        
        @self.app.get("/history/{user_id}")
        async def get_history(user_id: str):
            history = self.user_memory.get_history(user_id)
            return history
    
    async def background_search(self, query: str, sources: List[str], user_id: str) -> None:
        """
        Perform search in background.
        
        Args:
            query: Search query
            sources: List of sources to search
            user_id: User ID
        """
        try:
            # Fetch papers
            papers = self.paper_fetcher.fetch_papers(query, sources)
            
            # Embed papers
            embedded_papers = self.document_embedder.embed_documents(papers)
            
            # Add to vector store
            self.vector_store.add_documents(embedded_papers)
            
            # Save vector store
            self.vector_store.save('data/indexes/vector_store')
            
            # Update user memory with papers
            self.user_memory.add_results(user_id, {"papers": papers, "query": query})
            
            logger.info(f"Search completed for user {user_id}")
        except Exception as e:
            logger.error(f"Error in background search: {str(e)}")
            self.user_memory.add_results(user_id, {"error": str(e), "query": query})
    
    def initialize_llama(self) -> None:
        """
        Initialize the Llama handler.
        """
        logger.info("Initializing Llama handler")
        try:
            self.llama_handler = LlamaHandler()
            logger.info("Llama handler initialized")
        except Exception as e:
            logger.error(f"Error initializing Llama handler: {str(e)}")
            raise
    
    def load_data(self) -> None:
        """
        Load existing data from disk.
        """
        logger.info("Loading existing data")
        try:
            # Load vector store if it exists
            if os.path.exists('data/indexes/vector_store_documents.json'):
                self.vector_store.load('data/indexes/vector_store')
                logger.info("Vector store loaded")
            
            # Load user memory if it exists
            if os.path.exists('data/memory/user_memory.json'):
                self.user_memory.load('data/memory/user_memory.json')
                logger.info("User memory loaded")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")

app = FastAPI()
api = ResearchAssistantAPI()
app = api.app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)