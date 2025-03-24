# src/data_retrieval/pdf_parser.py
import requests
import io
from PyPDF2 import PdfReader
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PdfParser:
    """
    Parses PDF documents to extract full text content.
    """
    
    def __init__(self):
        pass
    
    def download_pdf(self, url: str) -> Optional[bytes]:
        """
        Download a PDF from a URL.
        """
        try:
            logger.info(f"Downloading PDF from {url}")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to download PDF, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text content from a PDF.
        """
        try:
            logger.info("Extracting text from PDF")
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def parse_pdf(self, url: str) -> Dict[str, Any]:
        """
        Download and parse a PDF from a URL.
        """
        pdf_content = self.download_pdf(url)
        if pdf_content:
            text = self.extract_text_from_pdf(pdf_content)
            return {
                'url': url,
                'text': text,
                'char_count': len(text)
            }
        else:
            return {
                'url': url,
                'text': '',
                'char_count': 0
            }
        
