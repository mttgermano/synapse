import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from ingest.vdb import start_vdb
from src.workflow import Workflow, WorkflowLLM

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import PrivateAttr

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

from typing import Any, List

# ---------------------------------------------------------------
# Carrega variáveis de ambiente
load_dotenv()

# ---------------------------------------------------------------
# Carrega dataset original e ajusta colunas
df = pd.read_csv("./data/dataset.csv")
df.rename(
    columns={
        "body": "question",
        "ideal_answer": "ground_truth",  # gabarito
    },
    inplace=True,
)
df = df.drop("exact_answer", axis=1)

evaluation_dataset = Dataset.from_pandas(df)
print("Dataset carregado com sucesso!")
print(evaluation_dataset)

# ---------------------------------------------------------------
# Inicializa retriever e workflow
retriever = start_vdb()
workflow = Workflow(retriever)
eval_llm = WorkflowLLM(workflow)
eval_embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-small")

# ---------------------------------------------------------------
# Prompt base do RAG
template = """
Use os seguintes trechos de contexto para responder à pergunta no final.
Se não souber a resposta, apenas diga que não sabe, não tente inventar uma.

Contexto: {context}

Pergunta: {question}

Resposta:
"""
prompt = ChatPromptTemplate.from_template(template)

generator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("API_KEY"),
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | generator_llm
    | StrOutputParser()
)

# ---------------------------------------------------------------
# Geração de respostas do modelo
answers = []
contexts = []

print("\nGerando respostas com o modelo RAG (Gemini)...")
for row in evaluation_dataset:
    question = row["question"]

    retrieved_docs = retriever.invoke(question)
    contexts.append([doc.page_content for doc in retrieved_docs])

    response = rag_chain.invoke(question)
    answers.append(response)

# Adiciona colunas esperadas pelo Ragas
dataset_with_results = evaluation_dataset.add_column("answer", answers)
dataset_with_results = dataset_with_results.add_column("contexts", contexts)

print("[*] Dataset com resultados do modelo RAG\n\n")
print(dataset_with_results)

# ---------------------------------------------------------------
# Avaliação com Ragas
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]

print("[*] Iniciando a avaliação com Ragas (usando Gemini como juiz)")

result = evaluate(
    dataset=dataset_with_results,
    metrics=metrics,
    llm=generator_llm,
    embeddings=eval_embeddings,
)

print("[*] Avaliação concluída")

# ---------------------------------------------------------------
# Resultados finais
print("[*] Resultados da Avaliação Ragas")
print(result)

df_results = result.to_pandas()
print(df_results)

df_results.to_csv("./evaluate/ragas_evaluation.csv", index=False)
exit()
