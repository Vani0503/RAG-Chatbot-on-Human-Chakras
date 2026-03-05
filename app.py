import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

vector_store = load_vectorstore()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

st.title("🧠 RAG Chatbot on Human Chakras")

query = st.text_input("Enter your question:")

if query:
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    final_prompt = f"""
    Answer the question using the context below.
    If not found, say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(final_prompt)

    st.subheader("Answer")
    st.write(response.content)

    st.subheader("Sources")
    for source in sources:
        st.write("-", source)
