from logger import logging
from exception import customexception
import sys
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader










def Data_loading(filename):
    #loading the PDF files
    try:

       

        logging.info("data loading started...")

        loader = PyPDFLoader("D:\MLOPS_END_PROJECT\LLM_PROJECT\QA_MEDI_CHATBOT\Data\Medical_book.pdf")
        documents = loader.aload()
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        #texts = text_splitter.split_documents(documents)
        return documents

    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e, sys)
    





