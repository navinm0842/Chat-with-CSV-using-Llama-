from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import CTransformers
import transformers
from transformers import AutoTokenizer 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.document_loaders.csv_loader import CSVLoader 
from langchain_community.vectorstores import FAISS 
from langchain.chains import RetrievalQA 
import torch 
import textwrap 
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tempfile

def run_llama(query,uploaded_file):
    llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
                            model_type="llama",
                            max_new_tokens=512,
                            temperature=0.1)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={'device': 'cpu'}) 
    
    if uploaded_file :
   #use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
         tmp_file.write(uploaded_file.getvalue())
         tmp_file_path = tmp_file.name

    loader = CSVLoader(tmp_file_path, 
                    encoding="utf-8", 
                    csv_args={'delimiter': ','}) 

    # Load the data
    data=loader.load()
    vectorstore = FAISS.from_documents(data, embeddings)

    chain = RetrievalQA.from_chain_type(llm=llm,
                    chain_type = "stuff",
                    return_source_documents=True,
                    retriever=vectorstore.as_retriever()) 
    while True:
        #query = input("Enter your query:")
        #print(query)

        result=chain(query) 

        wrapped_text = textwrap.fill(result['result'], width=500) 
        return(wrapped_text)