import streamlit as st
import asyncio
from MEDI_PDF_CHATBOT.data_ingestion import Data_loading
from MEDI_PDF_CHATBOT.embedding import download_embedding
from MEDI_PDF_CHATBOT.model_api import load_model
from MEDI_PDF_CHATBOT.retriver import redriever_model

# Streamlit App Config
st.set_page_config(page_title="MEDI PDF Chatbot", layout="wide")
st.title("🤖 MEDI PDF Chatbot Interface")

# Ensure necessary session states
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Upload PDF
with st.sidebar:
    st.header("📁 Upload PDF")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# Process uploaded PDF
if uploaded_file:
    with st.spinner("Processing PDF..."):
        if st.session_state.vector_store is None:
            # Use asyncio.run to properly handle the async function
            documents = asyncio.run(Data_loading(uploaded_file))

            # Load models (LLM and embeddings)
            llm, embeddings = load_model()
            st.session_state.llm = llm
            st.session_state.embeddings = embeddings

            # Generate embeddings and upload to Qdrant
            vector_store = download_embedding(embeddings, documents)
            st.session_state.vector_store = vector_store

            st.success("PDF processed successfully! Start chatting below.")
        else:
            st.info("PDF already processed. Ready to chat!")

# Display Chat History
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

# Input Box for Chat
if prompt := st.chat_input("How can I help?"):
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("Generating response..."):
        response = redriever_model(
            st.session_state.vector_store,
            st.session_state.llm,
            prompt,
            chain_type_kwargs={}
        )
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(response)
