# src/data_retrieval/fetcher.py
import arxiv
import requests
from bs4 import BeautifulSoup
from scholarly import scholarly
import logging
from typing import List, Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PaperFetcher:
    """
    Fetches research papers from multiple sources.
    """
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
    
    def fetch_from_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch papers from ArXiv based on a query.
        """
        logger.info(f"Fetching papers from ArXiv with query: {query}")
        
        try:
            # Create the search client
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Fetch results
            results = []
            for paper in search.results():
                results.append({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'summary': paper.summary,
                    'url': paper.pdf_url,
                    'source': 'arxiv',
                    'id': paper.get_short_id(),
                    'full_text_url': paper.pdf_url
                })
            
            logger.info(f"Retrieved {len(results)} papers from ArXiv")
            return results
        
        except Exception as e:
            logger.error(f"Error fetching from ArXiv: {str(e)}")
            return []
    
    def fetch_from_google_scholar(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch papers from Google Scholar based on a query.
        """
        logger.info(f"Fetching papers from Google Scholar with query: {query}")
        
        try:
            # Use scholarly to search Google Scholar
            search_query = scholarly.search_pubs(query)
            results = []
            count = 0
            
            for paper in search_query:
                if count >= self.max_results:
                    break
                
                try:
                    # Get detailed information
                    detailed_paper = scholarly.fill(paper)
                    
                    results.append({
                        'title': detailed_paper.get('bib', {}).get('title', 'N/A'),
                        'authors': detailed_paper.get('bib', {}).get('author', []),
                        'published': str(detailed_paper.get('bib', {}).get('pub_year', 'N/A')),
                        'summary': detailed_paper.get('bib', {}).get('abstract', 'N/A'),
                        'url': detailed_paper.get('pub_url', 'N/A'),
                        'source': 'google_scholar',
                        'id': detailed_paper.get('author_id', 'N/A'),
                        'citations': detailed_paper.get('num_citations', 0)
                    })
                    count += 1
                    
                    # Add a delay to avoid getting blocked
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Error processing Google Scholar paper: {str(e)}")
                    continue
            
            logger.info(f"Retrieved {len(results)} papers from Google Scholar")
            return results
        
        except Exception as e:
            logger.error(f"Error fetching from Google Scholar: {str(e)}")
            return []
    
    def fetch_from_pubmed(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch papers from PubMed based on a query.
        """
        logger.info(f"Fetching papers from PubMed with query: {query}")
        
        try:
            # Search PubMed
            search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query.replace(' ', '+')}&size={self.max_results}"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract paper information
            papers = soup.find_all('article', class_='full-docsum')
            results = []
            
            for paper in papers:
                try:
                    title_element = paper.find('a', class_='docsum-title')
                    title = title_element.text.strip() if title_element else 'N/A'
                    
                    authors_element = paper.find('span', class_='docsum-authors')
                    authors = [author.strip() for author in authors_element.text.split(',')] if authors_element else []
                    
                    published_element = paper.find('span', class_='docsum-journal-citation')
                    published = published_element.text.strip() if published_element else 'N/A'
                    
                    paper_id = paper.get('data-article-id', 'N/A')
                    paper_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}" if paper_id != 'N/A' else 'N/A'
                    
                    results.append({
                        'title': title,
                        'authors': authors,
                        'published': published,
                        'summary': 'N/A',  # PubMed doesn't provide summaries in the search results
                        'url': paper_url,
                        'source': 'pubmed',
                        'id': paper_id
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing PubMed paper: {str(e)}")
                    continue
            
            logger.info(f"Retrieved {len(results)} papers from PubMed")
            return results
        
        except Exception as e:
            logger.error(f"Error fetching from PubMed: {str(e)}")
            return []
    
    def fetch_papers(self, query: str, sources: List[str] = ['arxiv', 'google_scholar', 'pubmed']) -> List[Dict[str, Any]]:
        """
        Fetch papers from multiple sources based on a query.
        """
        logger.info(f"Fetching papers from sources {sources} with query: {query}")
        
        results = []
        
        if 'arxiv' in sources:
            results.extend(self.fetch_from_arxiv(query))
        
        if 'google_scholar' in sources:
            results.extend(self.fetch_from_google_scholar(query))
        
        if 'pubmed' in sources:
            results.extend(self.fetch_from_pubmed(query))
        
        logger.info(f"Retrieved a total of {len(results)} papers from all sources")
        return results

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
        
