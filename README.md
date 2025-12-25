
## Installation or how to use will be bellow:  ## For page  usage, ## Troubleshooting & FAQ, Technologies Used , Features and Requirements . this all are for project introduction. 

# ............................................................................................
#                          This is for just better understanding
# ...........................................................................................

## For page  usage:
- 1. first open the Streamlit page in your browser 
- 2. second upload your file 
- 3. Submit and process 
- 4. When it shows Successfully processed and indexed your document! 
- 5. Then ask your question  and Enter ( note:  this is lite frontend . please give some time ) 

## note 
- Since the file is uploaded locally, the question cannot be answered until the fine is uploaded.It will showing "answer is not available in the context "

##  Troubleshooting & FAQ

### Why do I sometimes see previous "Source Chunks" after uploading new documents?

This happens because Streamlit uses caching to speed up loading the FAISS index. If the cache is not cleared or the page is not refreshed after uploading new documents, the app may still use the old cached index, causing it to show chunks from previous uploads.

**Solution:**
- The app tries to clear the cache automatically after processing new documents. However, if you still see old chunks, simply refresh the Streamlit page in your browser after uploading and processing new files. This will ensure the latest FAISS index is loaded and your new documents are used for answering questions.

---
## Notes
- Make sure the backend is running before using the frontend.
- The first time you upload documents, a FAISS index will be created.
- Your Google API key is required for Gemini model access.


#  Retrieval-Augmented Generation (RAG) API: Chat with Any Document

This project is a smart, containerized RAG API and web app that lets you upload, process, and chat with any document type—including PDF, DOCX, TXT, images (JPG, PNG), CSV, and SQLite DB files. It uses FastAPI for the backend, Streamlit for the frontend, FAISS for vector search, and Google's Gemini LLM for answering your questions. Bonus: Supports image-based questions (OCR) and multi-document querying!

## Features
- Upload and process PDF, DOCX, TXT, image (JPG, JPEG, PNG), CSV, and SQLite DB files
- Extracts and preprocesses text and image content (OCR for images and scanned PDFs)
- Embeds content using Google Gemini (or swap for OpenAI/HuggingFace if desired)
- Stores embeddings and metadata in FAISS vector store
- Ask questions via text or image (base64) and get answers from your documents
- Multi-document querying: search across multiple files at once
- File-type icons and rich metadata in responses
- Minimal, modern web frontend (Streamlit)
- Fully containerized with Docker

## Technologies Used
- Python 3.9+
- FastAPI (backend API)
- Streamlit (frontend UI)
- FAISS (vector search)
- LangChain (chaining/orchestration)
- Google Gemini LLM (via API)
- pytesseract, pdfplumber, python-docx, pandas, sqlite3 (for extraction)
- Docker (for easy deployment)

# ............................................................................................
# till this is for just better understanding
# ............................................................................................

## Requirements and installation
- Python 3.9+
- Google Generative AI API key (set as `GOOGLE_API_KEY` in a `.env` file)
- (For Docker) Docker installed on your system

##  Installation
1. Clone this repository or download the code.

2. Create a virtual environment `.venv` file in the project root and activate this virtual environment 

3. Install dependencies in the virtual environment:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here

   ``` 

##  Usage

### Option 1: Local (Recommended for Development)

1. **Start the FastAPI backend:**
   ```
   uvicorn fastapi_app:app --reload --port 8000
   ```
2. **In a new terminal, start the Streamlit frontend:**
   ```
   streamlit run main.py
   ```
3. Open the Streamlit URL in your browser. Upload PDF, TXT, DOCX, image, CSV, or DB files, process them, and start chatting!

### Option 2: Docker (Production/All-in-One)
1. Build the Docker image:
   ```
   docker build -t rag-api .
   ```
2. Run the container:
   ```
   docker run -p 8000:8000 -p 8501:8501 --env-file .env rag-api
   ```
3. Access FastAPI docs at [http://localhost:8000/docs](http://localhost:8000/docs) and Streamlit UI at [http://localhost:8501](http://localhost:8501)

## 🗂️ File Structure
- `main.py` - Streamlit frontend
- `fastapi_app.py` - FastAPI backend
- `document_ingestor.py` - Document text extraction logic (PDF, TXT, DOCX, images, CSV, DB)
- `faiss_index/` - Stores the FAISS vector index
- `uploads/` - Uploaded files
- `requirements.txt` - Python dependencies
- `Dockerfile` - Containerization

## 📄 Supported File Types

The app can extract and process text from the following file types:

- **PDF** (`.pdf`)
- **Word Document** (`.docx`)
- **Text File** (`.txt`)
- **Images** (`.jpg`, `.jpeg`, `.png`) — using OCR
- **CSV** (`.csv`)
- **SQLite Database** (`.db`)

## Sample Workflow

1. Upload File: Upload a .pdf, .docx, .jpg, .csv, or .db via the Streamlit UI or `/upload` API endpoint

2. Ask a Question:
   - “What are the product specs mentioned in the attached PDF?”
   - “What is written in this image?” (image passed in base64)

3. API Returns:
   - Context
   - Final Answer
   - Source info (e.g., page 3 of invoice.pdf)

## 🏅 Bonus Features
- Image+text multimodal prompt support (OCR for images; can be extended to GPT-4 Vision, Claude, etc.)
- Multi-document querying (search across multiple files)
- File-type icons and detailed metadata in API responses
- Minimal, modern web frontend (Streamlit)
- Fully containerized with Docker

##  Notes
- Make sure the backend is running before using the frontend.
- The first time you upload documents, a FAISS index will be created.
- Your Google API key is required for Gemini model access.
- For OCR/image support, system dependencies are installed in Docker (see Dockerfile).


## Requirements
- Python 3.9+
- Google Generative AI API key (set as `GOOGLE_API_KEY` in a `.env` file)

## Installation
1. Clone this repository or download the code.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage
1. **Start the FastAPI backend:**
   ```
   uvicorn fastapi_app:app --reload --port 8000
   ```
2. **In a new terminal, start the Streamlit frontend:**
   ```
   streamlit run main.py
   ```
3. Open the Streamlit URL in your browser. Upload PDF, TXT, DOCX, image, CSV, or DB files, process them, and start chatting!

## File Structure
- `main.py` - Streamlit frontend
- `fastapi_app.py` - FastAPI backend
- `document_ingestor.py` - Document text extraction logic (PDF, TXT, DOCX, images, CSV, DB)
- `faiss_index/` - Stores the FAISS vector index
- `uploads/` - Uploaded files
- `requirements.txt` - Python dependencies
## Supported File Types

The app can extract and process text from the following file types:

- **PDF** (`.pdf`)
- **Word Document** (`.docx`)
- **Text File** (`.txt`)
- **Images** (`.jpg`, `.jpeg`, `.png`) — using OCR
- **CSV** (`.csv`)
- **SQLite Database** (`.db`)





