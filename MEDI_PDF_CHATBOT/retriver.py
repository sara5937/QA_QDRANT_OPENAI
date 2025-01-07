from langchain.chains import RetrievalQA
from logger import logging
from exception import customexception
import sys

def redriever_model(vector_store, Chat_model, user_question, chain_type_kwargs):
    try:
        logging.info("Retriever object created...")
        #retriever = vectordb.as_retriever(search_kwargs={"k": 2})
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        logging.info("Setting parameters for chain...")
        #qa_chain = RetrievalQA.from_chain_type(
            #llm=Chat_model,
            #chain_type="stuff",
            #retriever=retriever,
            #return_source_documents=True,
            #chain_type_kwargs=chain_type_kwargs
        #)
        qa_chain = RetrievalQA.from_chain_type(
        llm=Chat_model,
        retriever=retriever,
        return_source_documents=False  # To include source context in the output
)
        
        logging.info("Input user query...")
        #res = qa_chain.invoke(user_question)
        response = qa_chain.invoke({"query": user_question})
        
        logging.info("Query response received...")
        #print("Answer:", response['result'])
        #return res['result']
        return response['result']
    except Exception as e:
        logging.error(f"Error in retriever model: {e}")
        raise customexception(e, sys)