# Helper to extract text chunks with page numbers from PDF
import pdfplumber

def extract_pdf_chunks_with_page_numbers(file_path, chunk_size=4000, chunk_overlap=300):
    """
    Extracts text from a PDF and returns a list of (chunk, page_number) tuples.
    """
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Split page text into chunks
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                if chunk.strip():
                    chunks.append((chunk, page_num))
                start += chunk_size - chunk_overlap
    return chunks
