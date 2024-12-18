import os
import openai
import chromadb
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configuration
PERSIST_DIR = r"C:\Users\vannus8553\PycharmProjects\ChatWithMatt\chroma_db"
COLLECTION_NAME = "my_lecture_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"


def preprocess_text(text):
    """
    Clean and prepare text for embedding
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Break text into overlapping chunks
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


def get_embeddings(text_list):
    """
    Generate embeddings for a list of texts
    """
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=text_list
    )
    return [item["embedding"] for item in response["data"]]


def add_document_to_database(file_path):
    """
    Add a single document to the ChromaDB collection
    """
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Preprocess the text
    cleaned_text = preprocess_text(raw_text)

    # Chunk the text
    chunks = chunk_text(cleaned_text)

    # Generate embeddings
    chunk_embeddings = get_embeddings(chunks)

    # Prepare metadata and IDs
    filename = os.path.basename(file_path)
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    # Add to ChromaDB
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=chunk_embeddings
    )

    print(f"Added {len(chunks)} chunks from {filename} to the database")
    print(f"Total documents in collection: {collection.count()}")


def add_multiple_documents(directory_path):
    """
    Add all text files from a specified directory
    """
    # Ensure the directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a valid directory")
        return

    # Find all .txt files in the directory
    text_files = [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith('.txt')
    ]

    # Add each file to the database
    for file_path in text_files:
        try:
            add_document_to_database(file_path)
        except Exception as e:
            print(f"Error adding {file_path}: {e}")


# Main execution
if __name__ == "__main__":
    # Set up OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Specify the directory containing your text files
    DOCUMENTS_DIR = r"C:\Users\vannus8553\PycharmProjects\ChatWithMatt\.venv\data"

    # Add documents
    add_multiple_documents(DOCUMENTS_DIR)