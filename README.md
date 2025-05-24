# RAG App — Document Q&A

A FastAPI-based Retrieval-Augmented Generation (RAG) application for uploading documents and asking questions about their content using Google Generative AI.

## Features

- Upload documents (PDF, DOCX, TXT)
- Store and manage document metadata
- Ask questions and get answers based on uploaded documents
- Simple web frontend

## Requirements

- Python 3.9+
- [Google Generative AI API Key](https://makersuite.google.com/app/apikey)

## Setup

1. *Clone the repository*

   sh
   git clone <your-repo-url>
   cd rag_app/rag_app
   

2. *Create and activate a virtual environment*

   sh
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # Or
   source .venv/bin/activate  # On macOS/Linux
   

3. *Install dependencies*

   sh
   pip install -r requirements.txt
   

4. *Configure environment variables*

   - Copy .env and set your Google API key:
     
     GOOGLE_API_KEY="your-google-api-key"
     

5. *Run the application*

   sh
   uvicorn main:app --reload
   

6. *Open in your browser*

   Visit [http://localhost:8000](http://localhost:8000)

## Project Structure


rag_app/
  main.py
  requirements.txt
  .env
  db/
  routes/
  templates/
  uploads/
  faiss_indices/
  vectorstore/
  tests/


## Running Tests

sh
pytest


---

*Note:* This setup guide is for running the app locally without Docker.
