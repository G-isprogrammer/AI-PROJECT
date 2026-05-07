from pypdf import PdfReader
import chromadb
import os
import uuid
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="sbc")


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, size=1000):
    return [text[i:i+size] for i in range(0, len(text), size)]


def build():
    for file in os.listdir("knowledge"):
        if file.endswith(".pdf"):
            path = os.path.join("knowledge", file)
            print(f"Reading {file}")

            text = extract_text(path)
            chunks = chunk_text(text)

            for chunk in chunks:
                embedding = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                ).data[0].embedding

                collection.add(
                    ids=[str(uuid.uuid4())],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": file}]
                )

    print("DONE ✅")


if __name__ == "__main__":
    build()