# src/memory/user_memory.py
import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserMemory:
    """
    Manages user interaction history and search results.
    """
    
    def __init__(self):
        """
        Initialize user memory.
        """
        logger.info("Initializing UserMemory")
        self.memory = {}
        
        # Create data directory
        os.makedirs('data/memory', exist_ok=True)
    
    def add_query(self, user_id: str, query: str) -> None:
        """
        Add a query to user memory.
        
        Args:
            user_id: User ID
            query: Search query
        """
        if user_id not in self.memory:
            self.memory[user_id] = {
                "queries": [],
                "results": {},
                "summaries": {}
            }
        
        # Add query with timestamp
        self.memory[user_id]["queries"].append({
            "query": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save memory
        self.save()
    
    def add_results(self, user_id: str, results: Dict[str, Any]) -> None:
        """
        Add search results to user memory.
        
        Args:
            user_id: User ID
            results: Search results
        """
        if user_id not in self.memory:
            self.memory[user_id] = {
                "queries": [],
                "results": {},
                "summaries": {}
            }
        
        # Add results with timestamp
        query = results.get("query", "unknown")
        self.memory[user_id]["results"][query] = {
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save memory
        self.save()
    
    def add_summary(self, user_id: str, query: str, summary: str, citations: str) -> None:
        """
        Add a summary to user memory.
        
        Args:
            user_id: User ID
            query: Search query
            summary: Generated summary
            citations: Generated citations
        """
        if user_id not in self.memory:
            self.memory[user_id] = {
                "queries": [],
                "results": {},
                "summaries": {}
            }
        
        # Add summary with timestamp
        self.memory[user_id]["summaries"][query] = {
            "summary": summary,
            "citations": citations,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save memory
        self.save()
    
    def get_results(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest search results for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Latest search results or None
        """
        if user_id not in self.memory or not self.memory[user_id]["results"]:
            return None
        
        # Get the latest query
        if not self.memory[user_id]["queries"]:
            return None
        
        latest_query = self.memory[user_id]["queries"][-1]["query"]
        
        # Get results for the latest query
        if latest_query in self.memory[user_id]["results"]:
            return self.memory[user_id]["results"][latest_query]["data"]
        
        # If no results for the latest query, return the latest results
        latest_results = sorted(
            self.memory[user_id]["results"].items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        if latest_results:
            return latest_results[0][1]["data"]
        
        return None
    
    def get_history(self, user_id: str) -> Dict[str, Any]:
        """
        Get the complete history for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User history
        """
        if user_id not in self.memory:
            return {
                "queries": [],
                "summaries": {}
            }
        
        # Return a simplified version of the memory
        return {
            "queries": self.memory[user_id]["queries"],
            "summaries": {
                query: {
                    "summary": data["summary"],
                    "timestamp": data["timestamp"]
                }
                for query, data in self.memory[user_id]["summaries"].items()
            }
        }
    
    def save(self, path: str = 'data/memory/user_memory.json') -> None:
        """
        Save user memory to disk.
        
        Args:
            path: Path to save the memory
        """
        try:
            with open(path, 'w') as f:
                json.dump(self.memory, f)
            
            logger.info(f"User memory saved to {path}")
        except Exception as e:
            logger.error(f"Error saving user memory: {str(e)}")
    
    def load(self, path: str = 'data/memory/user_memory.json') -> None:
        """
        Load user memory from disk.
        
        Args:
            path: Path to load the memory from
        """
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.memory = json.load(f)
                
                logger.info(f"User memory loaded from {path}")
            else:
                logger.warning(f"User memory file not found at {path}")
        except Exception as e:
            logger.error(f"Error loading user memory: {str(e)}")