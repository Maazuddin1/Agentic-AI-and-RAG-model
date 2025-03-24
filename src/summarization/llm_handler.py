# src/summarization/llm_handler.py
import os
import sys
import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaHandler:
    """
    Handles interactions with Llama 3 model for summarization and citation generation.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3-8b-hf"):
        """
        Initialize the Llama handler.
        
        Args:
            model_name: Name of the Llama model to use
        """
        logger.info(f"Initializing LlamaHandler with model: {model_name}")
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate quantization if on CPU
            if self.device == "cpu":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    load_in_8bit=True
                )
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Create LangChain pipeline
            self.llm = HuggingFacePipeline(pipeline=self.pipeline)
            
            logger.info("LlamaHandler initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing LlamaHandler: {str(e)}")
            raise
    
    def generate_summary(self, documents: List[Dict[str, Any]], query: str) -> str:
        """
        Generate a summary of the documents based on the query.
        
        Args:
            documents: List of document dictionaries
            query: The user's query
            
        Returns:
            A summary of the documents
        """
        logger.info(f"Generating summary for {len(documents)} documents")
        
        try:
            # Prepare documents for LangChain
            docs = []
            for doc in documents:
                content = f"Title: {doc['title']}\nAuthors: {', '.join(doc['authors'])}\nPublished: {doc['published']}\nSummary: {doc.get('summary', 'N/A')}"
                docs.append(Document(page_content=content, metadata={"source": doc.get('source', 'unknown')}))
            
            # Create map prompt
            map_prompt_template = """You are a helpful research assistant. Your task is to analyze the following research paper and provide a brief summary of its key points, methods, and findings:

{text}

SUMMARY:"""
            map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
            
            # Create combine prompt
            combine_prompt_template = """You are a helpful research assistant. Given the following summaries of research papers, provide a comprehensive answer to the user's query: "{query}"

{text}

In your answer, please make sure to:
1. Synthesize information from multiple papers
2. Address the user's question directly
3. Provide specific details and examples
4. Highlight any contradictions or debates in the literature
5. Suggest potential directions for further research

COMPREHENSIVE ANSWER:"""
            combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text", "query"])
            
            # Create summary chain
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                verbose=True
            )
            
            # Run the chain
            summary = chain.run({"input_documents": docs, "query": query})
            
            logger.info(f"Generated summary with {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def generate_citations(self, documents: List[Dict[str, Any]], style: str = "APA") -> str:
        """
        Generate citations for the documents.
        
        Args:
            documents: List of document dictionaries
            style: Citation style (APA, MLA, etc.)
            
        Returns:
            Formatted citations
        """
        logger.info(f"Generating {style} citations for {len(documents)} documents")
        
        try:
            # Prepare documents for citation generation
            docs_text = ""
            for i, doc in enumerate(documents):
                docs_text += f"Document {i+1}:\n"
                docs_text += f"Title: {doc['title']}\n"
                docs_text += f"Authors: {', '.join(doc['authors'])}\n"
                docs_text += f"Published: {doc['published']}\n"
                docs_text += f"Source: {doc.get('source', 'N/A')}\n"
                docs_text += f"URL: {doc.get('url', 'N/A')}\n\n"
            
            # Create prompt
            prompt_template = f"""You are a citation expert. Generate {style} citations for the following documents:

{docs_text}

CITATIONS IN {style} FORMAT:"""
            
            # Generate citations
            result = self.pipeline(prompt_template)[0]['generated_text']
            
            # Extract only the citations part
            citations = result.split(f"CITATIONS IN {style} FORMAT:")[1].strip()
            
            logger.info(f"Generated {len(citations)} characters of citations")
            return citations
            
        except Exception as e:
            logger.error(f"Error generating citations: {str(e)}")
            return f"Error generating citations: {str(e)}"