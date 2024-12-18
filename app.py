import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import chromadb
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Flask App Configuration
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Database Configuration
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "my_lecture_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def get_query_embedding(query: str):
    """Convert a user query into an embedding using OpenAI."""
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    return response["data"][0]["embedding"]


def query_chroma(embedding, top_k=5):
    """Query ChromaDB with the query embedding."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    return results


def generate_answer(question: str, context_chunks: list):
    """Generate an answer using GPT with context."""
    system_message = (
        "You are a helpful assistant analyzing a specific document. "
        "Use the provided context to answer the question as precisely as possible. "
        "If the exact answer isn't available, provide the closest relevant information."
    )

    context_text = "\n\n".join(context_chunks)
    user_message = f"Context:\n{context_text}\n\nQuestion: {question}"

    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()


@app.route('/query', methods=['POST'])
def process_query():
    """Process user query and return answer"""
    data = request.json
    user_question = data.get('question', '')

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Generate query embedding
        query_embedding = get_query_embedding(user_question)

        # Query ChromaDB
        results = query_chroma(query_embedding)

        # Check results
        if not results.get("documents") or not results["documents"][0]:
            return jsonify({
                "answer": "No relevant information found in the database.",
                "context": []
            })

        # Prepare context and generate answer
        retrieved_chunks = results["documents"][0]
        answer = generate_answer(user_question, retrieved_chunks)

        return jsonify({
            "answer": answer,
            "context": retrieved_chunks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/')
def home():
    """Simple home route"""
    return "Matt's Lecture Chatbot is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)