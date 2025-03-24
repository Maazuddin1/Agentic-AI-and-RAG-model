# src/indexing/embeddings.py
import os
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentEmbedder:
    """
    Embeds documents into vector space for semantic search.
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialize the document embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        logger.info(f"Initializing DocumentEmbedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model dimension: {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text document.
        
        Args:
            text: The text to embed
            
        Returns:
            A numpy array containing the embedding
        """
        try:
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            # Return a zero vector if embedding fails
            return np.zeros(self.dimension)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Embed a list of documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of documents with added embeddings
        """
        logger.info(f"Embedding {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            # Combine title and summary for embedding
            text_to_embed = f"{doc['title']} {doc.get('summary', '')}"
            doc['embedding'] = self.embed_text(text_to_embed).tolist()
            
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(documents)} documents")
        
        logger.info(f"Embedded all {len(documents)} documents")
        return documents


class VectorStore:
    """
    Stores and indexes document embeddings for similarity search.
    """
    
    def __init__(self, dimension: int = 768, index_type: str = 'faiss'):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of index to use ('faiss' or 'memory')
        """
        logger.info(f"Initializing VectorStore with dimension: {dimension}, type: {index_type}")
        self.dimension = dimension
        self.index_type = index_type
        self.documents = []
        
        if index_type == 'faiss':
            # Initialize FAISS index
            self.index = faiss.IndexFlatL2(dimension)
        else:
            # Use in-memory list for small datasets
            self.index = []
        
        # Create directory for storing indexes
        os.makedirs('data/indexes', exist_ok=True)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with embeddings
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Store documents
        start_id = len(self.documents)
        self.documents.extend(documents)
        
        # Extract embeddings as numpy array
        embeddings = np.array([doc['embedding'] for doc in documents], dtype=np.float32)
        
        if self.index_type == 'faiss':
            # Add to FAISS index
            self.index.add(embeddings)
        else:
            # Add to in-memory list
            for i, embedding in enumerate(embeddings):
                self.index.append((start_id + i, embedding))
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query embedding
            top_k: Number of results to return
            
        Returns:
            List of similar documents
        """
        logger.info(f"Searching vector store for top {top_k} results")
        
        if len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        if self.index_type == 'faiss':
            # Search FAISS index
            distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents) and idx >= 0:
                    doc = self.documents[idx].copy()
                    doc['score'] = float(distances[0][i])
                    results.append(doc)
            
            return results
        else:
            # Search in-memory list
            scores = []
            for i, (doc_id, embedding) in enumerate(self.index):
                score = np.linalg.norm(query_embedding - embedding)
                scores.append((score, doc_id))
            
            # Sort by score (lower is better)
            scores.sort()
            
            results = []
            for score, doc_id in scores[:top_k]:
                doc = self.documents[doc_id].copy()
                doc['score'] = float(score)
                results.append(doc)
            
            return results
    
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Path to save the vector store
        """
        logger.info(f"Saving vector store to {path}")
        
        # Save documents
        with open(f"{path}_documents.json", 'w') as f:
            # Remove embeddings from documents for storage
            docs_to_save = [
                {k: v for k, v in doc.items() if k != 'embedding'}
                for doc in self.documents
            ]
            json.dump(docs_to_save, f)
        
        # Save embeddings separately
        embeddings = np.array([doc['embedding'] for doc in self.documents], dtype=np.float32)
        with open(f"{path}_embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        
        if self.index_type == 'faiss':
            # Save FAISS index
            faiss.write_index(self.index, f"{path}_faiss.index")
        
        logger.info(f"Saved vector store to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path: Path to load the vector store from
        """
        logger.info(f"Loading vector store from {path}")
        
        try:
            # Load documents
            with open(f"{path}_documents.json", 'r') as f:
                self.documents = json.load(f)
            
            # Load embeddings
            with open(f"{path}_embeddings.pkl", 'rb') as f:
                embeddings = pickle.load(f)
            
            # Add embeddings back to documents
            for i, doc in enumerate(self.documents):
                doc['embedding'] = embeddings[i].tolist()
            
            if self.index_type == 'faiss':
                # Load FAISS index
                self.index = faiss.read_index(f"{path}_faiss.index")
            else:
                # Recreate in-memory index
                self.index = []
                for i, embedding in enumerate(embeddings):
                    self.index.append((i, embedding))
            
            logger.info(f"Loaded vector store with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")