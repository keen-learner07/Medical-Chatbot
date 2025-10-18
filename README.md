🩺 MedicoBot — AI Medical Chat Assistant

MedicoBot is an AI-powered chatbot designed to answer medical questions using curated knowledge from the Gale Encyclopedia of Medicine. It leverages LangChain, Groq’s Llama 3.1 model, and FAISS-based semantic retrieval to provide accurate, context-grounded responses.

🚀 Features

   Conversational medical Q&A interface built with Streamlit

   Context-aware retrieval using FAISS

   Powered by Groq Llama 3.1 and HuggingFace embeddings

   Provides answers strictly from the Gale Encyclopedia of Medicine


🧩 Tech Stack

LangChain — Retrieval-Augmented Generation (RAG)

Groq API — LLM inference (Llama 3.1 8B Instant)

HuggingFace — Sentence embeddings (all-MiniLM-L6-v2)

FAISS — Vector similarity search

Streamlit — Interactive web UI

🧠 Note

MedicoBot can only answer questions available in the Gale Encyclopedia of Medicine.
If a query is outside its scope, it will respond:

“I'm sorry, but this information is not available in my dataset.”

🌐 Live Demo

🔗 Try it here: https://medical-chatbot-nd6e2t28z3swyu6mjg6lr9.streamlit.app/
