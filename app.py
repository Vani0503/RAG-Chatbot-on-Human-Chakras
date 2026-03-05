import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.title("🧠 RAG Chatbot on Human Chakras")

query = st.text_input("Ask a question")

if query:

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k":3})

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source","unknown") for doc in docs]))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    Answer using only the context below.
    If answer not present say you don't know.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.subheader("Answer")
    st.write(response.content)

    st.subheader("Sources")
    for s in sources:
        st.write("-", s)
