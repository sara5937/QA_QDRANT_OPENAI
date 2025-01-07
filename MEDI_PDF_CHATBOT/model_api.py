import os
from dotenv import load_dotenv
import sys

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI 
#from langchain.embeddings import SentenceTransformerEmbeddings

from exception import customexception
from logger import logging

load_dotenv()

#OPEN_AI_API_KEY= os.getenv('HUGGINGFACE_API_KEY')



def load_model():
    

    try:
        logging.info("Initialize Embedding model")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        #llm_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("Embedding Model Loaded")

       
        logging.info("Chat Model Loaded")
        logging.info("loading API key...")
        load_dotenv()
        #load_dotenv(dotenv_path="D:\MLOPS_END_PROJECT\LLM_PROJECT\MEDICAL_QA_CHATBOT\.env")
        OPEN_AI_API_KEY= os.getenv('OPEN_AI_API')

        logging.info("Saving API...")
       # Chat_model=CTransformers(model="Model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  ##model_type="llama",
                  #config={'max_new_tokens':512,
                          #'temperature':0.5})
        llm = ChatOpenAI( temperature=0.7, model_name="gpt-4o-mini", max_tokens=600,openai_api_key=OPEN_AI_API_KEY)
        logging.info("Returning Model")
       # return Chat_model,llm_embedding
        return llm, embeddings
    except Exception as e:
        raise customexception(e,sys)