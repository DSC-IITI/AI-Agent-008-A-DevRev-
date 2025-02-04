import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from main import search_query, get_answer

# Initialize model and collection first
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_collection("paragraph_collection")

def return_answer(question):
    
    context = search_query(
        question,
        collection,
        embedding_model,
        return_raw=True,
        top_k=2
    )

    # Get answer
    answer = get_answer(
        question,
        context,
        collection,
        embedding_model
    )

    return answer

# Streamlit App
st.title("QA Bot")
st.write("Ask any question below:")

question = st.text_input("Your Question", placeholder="Type your question here...")
if st.button("Get Answer"):
    if question:
        answer = return_answer(question)
        st.write(f"**Answer:** {answer}")
    else:
        st.error("Please enter a question.")
