import os
import openai
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

TRANSCRIPT_FILE = "./data/204.txt"
CHUNK_SIZE = 300
PERSIST_DIR = r"C:\Users\vannus8553\PycharmProjects\ChatWithMatt\chroma_db"
COLLECTION_NAME = "my_lecture_collection"

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_embeddings(text_list):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text_list
    )
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings

if __name__ == "__main__":
    # Read and chunk the transcript
    full_text = read_file(TRANSCRIPT_FILE)
    chunks = chunk_text(full_text, CHUNK_SIZE)

    # Embed the chunks
    chunk_embeddings = get_embeddings(chunks)

    # Initialize Chroma client with new configuration
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # Get or create the collection
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Add documents to the collection
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": TRANSCRIPT_FILE, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
        embeddings=chunk_embeddings
    )

    print("Document count after adding:", collection.count())
    all_docs = collection.get()
    print("All docs:", all_docs["documents"])
    print("Data preparation completed successfully!")
    print(f"Collection '{COLLECTION_NAME}' now contains {collection.count()} documents.")
