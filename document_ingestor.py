import os
import pdfplumber
from docx import Document
import pytesseract
from PIL import Image
import pandas as pd
import sqlite3

class DocumentIngestor:
    @staticmethod
    def extract_text_from_pdf_with_pages(file_path, chunk_size=1000, chunk_overlap=200):
        from pdf_chunk_helper import extract_pdf_chunks_with_page_numbers
        return extract_pdf_chunks_with_page_numbers(file_path, chunk_size, chunk_overlap)

    @staticmethod
    def extract_text_from_docx(file_path, chunk_size=1000, chunk_overlap=200):
        doc = Document(file_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        # Join paragraphs into chunks
        chunks = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = para
            else:
                if current:
                    current += "\n" + para
                else:
                    current = para
        if current:
            chunks.append(current)
        return [(chunk, idx+1) for idx, chunk in enumerate(chunks)]

    @staticmethod
    def extract_text_from_txt(file_path, chunk_size=1000, chunk_overlap=200):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Join lines into chunks
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = line
            else:
                if current:
                    current += "\n" + line
                else:
                    current = line
        if current:
            chunks.append(current)
        return [(chunk, idx+1) for idx, chunk in enumerate(chunks)]

    @staticmethod
    def extract_text_from_image(file_path):
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    @staticmethod
    def extract_text_from_csv(file_path, chunk_size=1000, chunk_overlap=200):
        df = pd.read_csv(file_path)
        lines = df.to_string(index=False).splitlines()
        # Join lines into chunks
        chunks = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = line
            else:
                if current:
                    current += "\n" + line
                else:
                    current = line
        if current:
            chunks.append(current)
        return [(chunk, idx+1) for idx, chunk in enumerate(chunks)]

    @staticmethod
    def extract_text_from_db(file_path):
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        text = ""
        for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall():
            table_name = table[0]
            rows = cursor.execute(f"SELECT * FROM {table_name}").fetchall()
            text += f"Table: {table_name}\n"
            for row in rows:
                text += str(row) + "\n"
        conn.close()
        return text

    @staticmethod
    def extract(file_path, chunk_size=1000, chunk_overlap=200):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return DocumentIngestor.extract_text_from_pdf_with_pages(file_path, chunk_size, chunk_overlap)
        elif ext == '.docx':
            return DocumentIngestor.extract_text_from_docx(file_path, chunk_size, chunk_overlap)
        elif ext == '.txt':
            return DocumentIngestor.extract_text_from_txt(file_path, chunk_size, chunk_overlap)
        elif ext in ['.jpg', '.jpeg', '.png']:
            text = DocumentIngestor.extract_text_from_image(file_path)
            return [(text, 1)]
        elif ext == '.csv':
            return DocumentIngestor.extract_text_from_csv(file_path, chunk_size, chunk_overlap)
        elif ext == '.db':
            text = DocumentIngestor.extract_text_from_db(file_path)
            return [(text, 1)]
        else:
            raise ValueError(f"Unsupported file type: {ext}")
