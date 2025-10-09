import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv

load_dotenv()

# Step 1: Setup Groq LLM
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    temperature=0.5,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)


# Step 2: Connect LLM with FAISS and Create chain

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
)

STRICT_MEDICAL_PROMPT = PromptTemplate(
    template="""You are a medical assistant. 
    Use ONLY the provided context to answer the user's question. 
    Do NOT guess or provide information that is not in the context. 
    If the answer is not present in the context, politely respond: 
    "I'm sorry, but this information is not available in my dataset."

    Context:
    {context}

    Question:
    {input}

    Instructions:
    - Provide a **detailed and comprehensive answer** based on the context.
    - If the user asks a general question (e.g., "Tell me about X"), provide all relevant information available in the context in a coherent, well-structured, and thorough manner.
    - If the user asks something specific (e.g., "What are the symptoms?"), provide only that information, but still **elaborate as much as possible**.
    - Do NOT mention section headers like Definition, Symptoms, Diagnosis, or Notes. Just give the content naturally.
    - Explain clearly, give examples if relevant, and make it educational.
    - Use complete sentences and paragraphs for readability.

    Answer:
    """,
    input_variables=["context", "input"],
)

# Document combiner chain (stuff documents into prompt)
combine_docs_chain = create_stuff_documents_chain(llm, STRICT_MEDICAL_PROMPT)

# Retrieval chain (retriever + doc combiner)
rag_chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
)


# Now invoke with a single query
user_query = input("Write Query Here: ")
response = rag_chain.invoke({"input": user_query})
print("RESULT: ", response["answer"])
print("\nSOURCE DOCUMENTS:")
for doc in response["context"]:
    print(f"- {doc.metadata} -> {doc.page_content[:200]}...")
