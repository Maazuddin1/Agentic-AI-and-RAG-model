# app.py (for Hugging Face Spaces)
import os
import sys
import subprocess
import streamlit as st

# Check if the environment is set up
if not os.path.exists(".env_setup"):
    st.write("Setting up environment...")
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Create data directories
    os.makedirs("data/indexes", exist_ok=True)
    os.makedirs("data/memory", exist_ok=True)
    os.makedirs("data/papers", exist_ok=True)
    
    # Create .env_setup file
    with open(".env_setup", "w") as f:
        f.write("Environment set up")
    
    st.write("Environment set up complete. Please refresh the page.")
    st.stop()

# Import FastAPI app
from src.api.fastapi_app import app as fastapi_app
import threading
import uvicorn

# Start FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

# Start FastAPI server if not already running
if not os.environ.get("FASTAPI_STARTED"):
    thread = threading.Thread(target=run_fastapi)
    thread.daemon = True
    thread.start()
    os.environ["FASTAPI_STARTED"] = "1"
    os.environ["API_URL"] = "http://localhost:8000"

# Run Streamlit app
from src.frontend.streamlit_app import main
main()