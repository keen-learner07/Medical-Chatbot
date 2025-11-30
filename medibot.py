import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from chat_manager import (
    initialize_chat_system,
    sidebar_chats,
    get_current_chat_messages,
    add_message_to_current_chat,
)

from dotenv import load_dotenv

load_dotenv()


DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return db


def main():
    initialize_chat_system()

    with st.sidebar:
        sidebar_chats()

    st.markdown(
        """ <style> .block-container { padding-top: 1rem; /* Reduce top padding */ } </style> """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p style='text-align: center; font-size: 16px; color: grey; margin-bottom: -5px;'>
            A medical question-answering chatbot powered by a curated medical dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.title("Ask MedicoBot!")

    for message in get_current_chat_messages():
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        add_message_to_current_chat("user", prompt)
        st.chat_message("user").markdown(prompt)

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"
            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY,
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
            combine_docs_chain = create_stuff_documents_chain(
                llm, STRICT_MEDICAL_PROMPT
            )

            # Retrieval chain (retriever + doc combiner)
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
            )

            response = rag_chain.invoke({"input": prompt})
            bot_reply = response["answer"]

            add_message_to_current_chat("assistant", bot_reply)
            st.chat_message("assistant").markdown(bot_reply)

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
