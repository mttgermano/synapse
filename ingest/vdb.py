import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def start_vdb():
    df = pd.read_csv("./data/dataset.csv")
    print("[*] CSV loaded")

    docs = [
        Document(
            #page_content=f"[QUESTION] {row['body']} [ANSWER] {row['ideal_answer']}",
            page_content=f"{row['body']} {row['ideal_answer']}",
            metadata={"source": row["snippets"]},
        )
        for _, row in df.iterrows()
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
    print("[*] VDB loaded")

    return vectorstore.as_retriever(search_kwargs={"k": 4})
