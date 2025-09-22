import requests
import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from time import time


def start_vdb():
    df = pd.read_csv("./dataset.csv")
    print("[*] CSV loaded")
    
    docs = [
        Document(
            page_content=f"[QUESTION] {row['body']}\n\n[ANSWER] {row['ideal_answer']}",
            metadata={"source": row["snippets"]}
        )
        for index, row in df.iterrows()
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Vector Store com FAISS e embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    print("[*] VDB loaded")
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
    return retriever

t0 = time()
r = start_vdb()
t1 = time()
print(t0-t1)
test_question = "What is the use of P85-Ab?"
retrieved_docs = r.invoke(test_question)

for doc in retrieved_docs:
    print(f"Page Content: {doc.page_content[:150]}...")
    print(f"Source: {doc.metadata['source']}\n")
