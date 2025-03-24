# Agentic AI Research Assistant

An AI-powered research assistant that fetches, indexes, and summarizes research papers from multiple sources.

## Features

- **Multi-source Paper Retrieval**: Search for papers from ArXiv, Google Scholar, and PubMed
- **Semantic Search**: Find papers based on meaning, not just keywords
- **AI-powered Summarization**: Generate comprehensive summaries of research papers
- **Citation Generation**: Automatically generate citations in various formats
- **User Memory**: Track search history and previous summaries
- **FastAPI Backend**: High-performance API with async support
- **Streamlit Frontend**: User-friendly interface for interacting with the assistant

## Architecture

The application is built with a modular architecture:

1. **Data Retrieval**: Fetches papers from multiple sources
2. **Document Indexing**: Converts papers into vector representations for semantic search
3. **LLM Summarization**: Uses Llama 3 to generate summaries and citations
4. **FastAPI Backend**: Exposes functionality as a RESTful API
5. **Streamlit Frontend**: Provides a user-friendly interface

## Installation

### Prerequisites

- Python 3.10+
- Git

### Option 1: Using Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# Build and run the Docker container
docker build -t research-assistant .
docker run -p 8000:8000 -p 8501:8501 research-assistant
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/research-assistant.git
cd research-assistant

# Run the