import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    st.session_state.chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        return_source_documents=True
    )

st.title("🧠 RAG Chatbot on Human Chakras")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Enter your question:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    result = st.session_state.chain({"question": query})
    answer = result["answer"]
    docs = result["source_documents"]
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.subheader("Sources")
        for source in sources:
            st.write("-", source)

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    del st.session_state["chain"]
    st.rerun()
