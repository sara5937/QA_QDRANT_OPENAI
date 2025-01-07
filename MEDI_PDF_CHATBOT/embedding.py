from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


from MEDI_PDF_CHATBOT.data_ingestion import Data_loading
from MEDI_PDF_CHATBOT.model_api import load_model
from dotenv import load_dotenv
import os

import sys
from exception import customexception
from logger import logging



def download_embedding(model, document):
    try:
        collection_name = "qa_medi_chatbot1"
        logging.info("Creating Chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(document)
        
        logging.info("Extracting as plain text...")
        plain_texts = [doc.page_content for doc in document]

        logging.info("Loading environment variables...")
        load_dotenv()
        qdrant_url = os.getenv("QTRAND_URL")
        qdrant_api_key = os.getenv("QDTRAND_API")
        
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QTRAND_URL or QDTRAND_API not set in environment variables.")
        
        logging.info("Connecting to Qdrant...")
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        logging.info("Checking if collection exists...")
        if not qdrant_client.collection_exists(collection_name):
            logging.info("Collection does not exist. Creating collection...")
            vector_size = 768
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": "Cosine"}
            )
        else:
            logging.info("Collection already exists. Reusing collection.")

        logging.info("Adding texts to vector store...")
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=collection_name,
            embedding=model
        )
        vector_store.add_texts(plain_texts)

        logging.info("Returning vector store...")
        return vector_store

    except Exception as e:
        logging.error(f"Error in download_embedding: {e}")
        raise customexception(e, sys)
