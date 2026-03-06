import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 RAG Chatbot on Human Chakras")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Build chat history string for context
    history_text = ""
    for m in st.session_state.messages[:-1]:  # exclude current query
        role = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{role}: {m['content']}\n"

    # Retrieve relevant docs
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    # Prompt with memory
    final_prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say you don't know.

Previous conversation:
{history_text}

Context:
{context}

Question: {query}
"""

    response = llm.invoke(final_prompt)
    answer = response.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("📄 Sources"):
            for source in sources:
                st.write("-", source)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()
