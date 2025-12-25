from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import base64
from document_ingestor import DocumentIngestor
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from main import get_conversational_chain
from typing import List, Optional


app = FastAPI()
# Add CORS for local dev (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a root endpoint for a friendly welcome message
@app.get("/")
def root():
    return {"message": "Welcome to the RAG API! Use /docs for API documentation."}
UPLOAD_DIR = "uploads"
INDEX_DIR = "faiss_index"

os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None
    file_id: Optional[str] = None  # can be comma-separated for multi-doc

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Extract text
        raw_text = DocumentIngestor.extract(file_path)
        # Handle list or string return from extract
        if isinstance(raw_text, list):
            # If it returns list of tuples, join them
            raw_text = "\n".join([item[0] if isinstance(item, tuple) else item for item in raw_text])
        if not raw_text or not str(raw_text).strip():
            raise HTTPException(status_code=400, detail="No extractable text found in file.")
        # Chunk text
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = text_splitter.split_text(raw_text)
        if not text_chunks:
            raise HTTPException(status_code=400, detail="No text chunks generated from file.")
        # Prepare metadatas for each chunk
        ext = os.path.splitext(file.filename or "")[1].lower() if file.filename else ""
        metadatas = [
            {"filename": file.filename, "chunk": i, "filetype": ext}
            for i in range(len(text_chunks))
        ]
        # Embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas)
        # Save index with file_id
        file_id = os.path.splitext(file.filename or "")[0] if file.filename else "unknown"
        index_path = os.path.join(INDEX_DIR, file_id)
        os.makedirs(index_path, exist_ok=True)
        vectorstore.save_local(index_path)
        # Optionally, clean up uploaded file to save space
        try:
            os.remove(file_path)
        except Exception:
            pass
        return {"file_id": file_id, "filename": file.filename, "filetype": ext, "icon": filetype_icon(ext)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/query")
async def query_api(request: QueryRequest):
    # Validate input
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    context = ""
    metadatas = []
    docs = []
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Multi-document support: file_id can be comma-separated or None (search all)
    file_ids = []
    if request.file_id:
        file_ids = [fid.strip() for fid in request.file_id.split(",") if fid.strip()]
    else:
        # If no file_id, search all indexes
        file_ids = [d for d in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, d))]
    if not file_ids:
        raise HTTPException(status_code=404, detail="No documents found to search.")
    # Limit number of docs per query for performance
    max_chunks = 10
    for fid in file_ids:
        index_path = os.path.join(INDEX_DIR, fid)
        if not os.path.exists(index_path):
            continue
        try:
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            docs.extend(db.similarity_search(request.question, k=max_chunks))
        except Exception as e:
            continue
    # If image is provided, extract text using OCR
    if request.image_base64:
        try:
            from PIL import Image
            import io
            import pytesseract
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            context += pytesseract.image_to_string(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
    # Add context from docs and collect metadata (limit to max_chunks)
    docs = docs[:max_chunks]
    context += "\n".join([doc.page_content for doc in docs])
    metadatas = [doc.metadata for doc in docs]
    # LLM QA
    try:
        chain = get_conversational_chain()
        # Format context as string
        context_text = "\n".join([doc.page_content for doc in docs])
        response = chain.invoke({"context": context_text, "question": request.question})
        answer = response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {str(e)}")
    # Add icon and filetype to each metadata
    for m in metadatas:
        m["icon"] = filetype_icon(m.get("filetype", ""))
    return {
        "context": context,
        "answer": answer,
        "sources": metadatas
    }

# Helper for file-type icon
def filetype_icon(ext):
    icons = {
        ".pdf": "📄",
        ".docx": "📝",
        ".txt": "📃",
        ".jpg": "🖼️",
        ".jpeg": "🖼️",
        ".png": "🖼️",
        ".csv": "📊",
        ".db": "🗄️",
    }
    return icons.get(ext.lower(), "❓")
