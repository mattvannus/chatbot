import os
import openai
import chromadb
from dotenv import load_dotenv

##############################
# Configuration
##############################
# Load environment variables
load_dotenv()

# Get API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

PERSIST_DIR = r"C:\Users\vannus8553\PycharmProjects\ChatWithMatt\chroma_db"
COLLECTION_NAME = "my_lecture_collection"
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"  # or gpt-4 if you have access
TOP_K = 5  # Number of relevant chunks to retrieve


##############################
# Functions
##############################

def get_query_embedding(query: str):
    """Convert a user query into an embedding using OpenAI."""
    response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    return response["data"][0]["embedding"]


def get_comprehensive_context(results, top_k=5):
    """
    Extract and prepare context with more information
    """
    # Collect all retrieved documents
    all_documents = results.get("documents", [[]])[0]

    # Collect metadata to understand context
    all_metadatas = results.get("metadatas", [[]])[0]

    # Combine documents with their metadata for more context
    enhanced_context = [
        f"Document Chunk {i + 1} (Source: {meta.get('source', 'Unknown')}):\n{doc}"
        for i, (doc, meta) in enumerate(zip(all_documents, all_metadatas))
    ]

    return enhanced_context


def query_chroma(embedding, query_text, top_k=5):
    """
    Improve query with more sophisticated retrieval and debugging
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Debug: Print collection information
    print(f"Total documents in collection: {collection.count()}")

    try:
        results = collection.query(
            query_embeddings=[embedding],  # Semantic similarity
            n_results=top_k
        )

        # Additional debugging
        print("Debug - Query Results:")
        print("Documents:", results.get("documents", "No documents"))
        print("Distances:", results.get("distances", "No distances"))

        return results

    except Exception as e:
        print(f"Error in query_chroma: {e}")
        raise


def process_query(user_question):
    """
    Process a user's query with more robust error handling and debugging
    """
    try:
        # 1. Generate query embedding
        query_embedding = get_query_embedding(user_question)

        # 2. Perform query
        results = query_chroma(
            embedding=query_embedding,
            query_text=user_question,
            top_k=7
        )

        # 3. Check results
        if not results.get("documents") or not results["documents"][0]:
            print("Debugging information:")
            print("Embedding length:", len(query_embedding))
            print("Embedding first few values:", query_embedding[:5])
            return "No relevant documents found in the database. The database might be empty or the query is not matching any documents."

        # 4. Prepare context
        retrieved_chunks = results["documents"][0]

        # 5. Generate answer
        answer = generate_answer(user_question, retrieved_chunks)

        return answer

    except Exception as e:
        print(f"Comprehensive error in processing query: {e}")
        return f"An error occurred: {e}"


def process_query(user_question):
    # 1. Generate query embedding
    query_embedding = get_query_embedding(user_question)

    # 2. Perform advanced query with both embedding and text
    results = query_chroma(
        embedding=query_embedding,
        query_text=user_question,
        top_k=7  # Increased from 5 to get more potential context
    )

    # 3. Get comprehensive context
    if not results.get("documents") or not results["documents"][0]:
        return "No relevant documents found in the database."

    # 4. Prepare enhanced context
    retrieved_chunks = get_comprehensive_context(results)

    # 5. Generate answer with more context
    answer = generate_answer(user_question, retrieved_chunks)

    return answer

def generate_answer(question: str, context_chunks: list):
    """
    More robust answer generation with fallback strategies
    """
    system_message = (
        "You are a helpful assistant analyzing a specific document. "
        "The Document is notes from my lecture."
        "The student will have questions about assignments, due date exams and other educational realted material"
        "Use the provided context to answer the question as precisely as possible. "
        "If the exact answer isn't available, provide the closest relevant information. "
        "If truly no information is found, explain what information is missing."
    )

    # Combine context chunks with more context
    context_text = "\n\n".join(context_chunks)

    # Add more detailed context to help GPT understand
    user_message = (
        f"Document Context:\n{context_text}\n\n"
        f"Specific Question: {question}\n\n"
        "Please analyze the context carefully and provide the most relevant information possible."
    )

    try:
        response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,  # Slightly lower temperature for more precise answers
            max_tokens=1500  # Increased max tokens for more comprehensive responses
        )

        answer = response.choices[0].message.content.strip()

        # Additional fallback logic
        if not answer or len(answer) < 10:
            return "I couldn't find specific information about this in the document. The context might be too limited."

        return answer

    except Exception as e:
        return f"An error occurred while generating the answer: {str(e)}"


##############################
# Main
##############################

def process_query(user_question):
    """
    Process a user's query and return an answer from the document
    """
    # 1. Generate query embedding
    query_embedding = get_query_embedding(user_question)

    # 2. Perform advanced query with both embedding and text
    results = query_chroma(
        embedding=query_embedding,
        query_text=user_question,
        top_k=7  # Increased from 5 to get more potential context
    )

    # 3. Get comprehensive context
    if not results.get("documents") or not results["documents"][0]:
        return "No relevant documents found in the database."

    # 4. Prepare enhanced context
    retrieved_chunks = get_comprehensive_context(results)

    # 5. Generate answer with more context
    answer = generate_answer(user_question, retrieved_chunks)

    return answer


def main():
    print("Welcome to the Document Query Assistant!")
    print("Type your questions, or type 'exit' to quit.")

    while True:
        # Prompt for user input
        user_question = input("\nAsk your question (or 'exit' to quit): ").strip()

        # Check for exit condition
        if user_question.lower() in ['exit', 'quit', 'q']:
            print("Thank you for using the Document Query Assistant. Goodbye!")
            break

        # Validate input
        if not user_question:
            print("Please enter a valid question.")
            continue

        try:
            # Process the query and print the answer
            answer = process_query(user_question)
            print("\nAnswer:\n", answer)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()