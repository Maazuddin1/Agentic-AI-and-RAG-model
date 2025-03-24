# app.py
import streamlit as st
import requests
import json
import time
import os
from typing import Dict, Any, List

# Set API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Set page config
st.set_page_config(
    page_title="Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Function to search papers
def search_papers(query: str, sources: List[str]) -> Dict[str, Any]:
    response = requests.post(
        f"{API_URL}/search",
        json={"query": query, "sources": sources},
        headers={"X-User-ID": st.session_state.user_id}
    )
    return response.json()

# Function to get results
def get_results() -> Dict[str, Any]:
    response = requests.get(
        f"{API_URL}/results/{st.session_state.user_id}"
    )
    if response.status_code == 200:
        return response.json()
    return None

# Function to generate summary
def generate_summary(query: str) -> Dict[str, Any]:
    response = requests.post(
        f"{API_URL}/summarize",
        json={"query": query},
        headers={"X-User-ID": st.session_state.user_id}
    )
    return response.json()

# Function to get user history
def get_history() -> Dict[str, Any]:
    response = requests.get(
        f"{API_URL}/history/{st.session_state.user_id}"
    )
    return response.json()

# Initialize session state
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

if "results" not in st.session_state:
    st.session_state.results = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "citations" not in st.session_state:
    st.session_state.citations = None

if "tab" not in st.session_state:
    st.session_state.tab = "search"

# App title
st.title("📚 Research Assistant")
st.write("Search for research papers and generate summaries")

# Create tabs
tabs = st.tabs(["Search", "Results", "Summary", "History"])

# Search tab
with tabs[0]:
    st.header("Search")
    
    with st.form("search_form"):
        query = st.text_input("Search Query", value=st.session_state.search_query)
        sources = st.multiselect(
            "Sources",
            ["arxiv", "google_scholar", "pubmed"],
            default=["arxiv", "google_scholar"]
        )
        
        submitted = st.form_submit_button("Search")
        
        if submitted and query:
            with st.spinner("Searching..."):
                st.session_state.search_query = query
                result = search_papers(query, sources)
                st.success(f"Search started: {result['message']}")
                st.session_state.tab = "results"

# Results tab
with tabs[1]:
    st.header("Results")
    
    if st.button("Refresh Results"):
        with st.spinner("Getting results..."):
            st.session_state.results = get_results()
    
    if st.session_state.results:
        papers = st.session_state.results.get("papers", [])
        st.write(f"Found {len(papers)} papers")
        
        for i, paper in enumerate(papers):
            with st.expander(f"{i+1}. {paper['title']}"):
                st.write(f"**Authors:** {', '.join(paper['authors'])}")
                st.write(f"**Published:** {paper['published']}")
                st.write(f"**Source:** {paper['source']}")
                st.write(f"**URL:** {paper['url']}")
                st.write("**Summary:**")
                st.write(paper.get('summary', 'No summary available'))
    else:
        st.info("No results available yet. Please search first or refresh.")

# Summary tab
with tabs[2]:
    st.header("Summary")
    
    if st.session_state.results:
        with st.form("summary_form"):
            summary_query = st.text_input("Summary Query", value=st.session_state.search_query)
            submitted = st.form_submit_button("Generate Summary")
            
            if submitted:
                with st.spinner("Generating summary..."):
                    result = generate_summary(summary_query)
                    st.session_state.summary = result["summary"]
                    st.session_state.citations = result["citations"]
    
    if st.session_state.summary:
        st.subheader("Generated Summary")
        st.write(st.session_state.summary)
        
        st.subheader("Citations")
        st.write(st.session_state.citations)
    else:
        st.info("No summary available yet. Please search for papers first and then generate a summary.")

# History tab
with tabs[3]:
    st.header("History")
    
    if st.button("Refresh History"):
        with st.spinner("Getting history..."):
            history = get_history()
            
            if history:
                st.subheader("Past Queries")
                for query_data in history["queries"]:
                    st.write(f"- {query_data['query']} ({query_data['timestamp']})")
                
                st.subheader("Past Summaries")
                for query, summary_data in history["summaries"].items():
                    with st.expander(f"Summary for: {query}"):
                        st.write(f"**Generated at:** {summary_data['timestamp']}")
                        st.write(summary_data["summary"])
            else:
                st.info("No history available yet.")