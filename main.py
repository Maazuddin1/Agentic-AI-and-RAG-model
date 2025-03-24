# main.py
import os
import logging
import argparse
from src.api.fastapi_app import app as fastapi_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Research Assistant")
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000, 
        help="Port to run the API on"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to run the API on"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="meta-llama/Llama-3-8b-hf", 
        help="Model to use for summarization"
    )
    parser.add_argument(
        "--max-results", 
        type=int, 
        default=10, 
        help="Maximum number of search results"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set environment variables
    os.environ["RESEARCH_ASSISTANT_MODEL"] = args.model
    os.environ["RESEARCH_ASSISTANT_MAX_RESULTS"] = str(args.max_results)
    
    # Create data directories
    os.makedirs('data/indexes', exist_ok=True)
    os.makedirs('data/memory', exist_ok=True)
    os.makedirs('data/papers', exist_ok=True)
    
    # Start FastAPI server
    import uvicorn
    logger.info(f"Starting FastAPI server on {args.host}:{args.port}")
    uvicorn.run(fastapi_app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()