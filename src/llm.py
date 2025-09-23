from typing import List
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import os

from dotenv import load_dotenv
load_dotenv()


def init_llm(retriever):
    llm = llm_base()
    llm, tool_map = bind_tools(llm, retriever)
    return llm, tool_map


def llm_base():
    #return ChatGoogleGenerativeAI(
    #    model="gemini-2.5-flash",
    #    temperature=0.0,
    #    google_api_key=os.getenv("API_KEY"),
    #)
    return ChatOllama(
        model="llama3",
        temperature=0.0,
    )

def bind_tools(llm, retriever):
    @tool
    def retriever_agent(question: str):
        """Busca documentos relevantes de biomediciana para responder à pergunta."""
        print("--[Tool] RetrieverAgent")
        documents = retriever.invoke(question)
        for doc in documents:
            if "source" in doc.metadata and "page" in doc.metadata:
                doc.metadata["source"] = f"{os.path.basename(doc.metadata['source'])}, pág. {doc.metadata['page']}"
        return documents

    @tool
    def answer_agent(question: str, documents: List[dict]):
        """Gera resposta com base apenas nos documentos fornecidos, citando fontes."""
        print("--[Tool]: AnswerAgent")
        prompt_template = """
        Você é um assistente especializado em Biomedicina.
        Responda à pergunta SOMENTE com base no contexto fornecido.
        Se a informação não estiver no contexto, diga: 
        "Não encontrei informações sobre isso nos documentos fornecidos."
        Cite as fontes no formato [fonte: metadata.source].

        Contexto: {context}
        Pergunta: {question}
        Resposta:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = prompt | llm | StrOutputParser()
        context_str = "\n\n".join(
            f"[fonte: {doc['metadata']['source']}]\n{doc['page_content']}"
            for doc in documents
        )
        answer = rag_chain.invoke({"context": context_str, "question": question})
        return {"answer": answer, "documents": documents}

    @tool
    def self_check_agent(answer: str, documents: List[dict]):
        """Verifica se a resposta é suportada pelo contexto fornecido."""
        print("--[Tool]: SelfCheckAgent")
    
        class Verdict(BaseModel):
            is_grounded: bool = Field(
                description="A resposta é totalmente suportada pelos documentos? (true/false)"
            )
    
        parser = JsonOutputParser(pydantic_object=Verdict)
    
        prompt_template = """
        Verifique se a resposta está totalmente fundamentada no contexto.
        Responda APENAS com JSON no formato:
        {{"is_grounded": true}} ou {{"is_grounded": false}}
    
        Contexto: {context}
        Resposta: {answer}
        {format_instructions}
        """
    
        prompt = ChatPromptTemplate.from_template(
            prompt_template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
    
        context_str = "\n\n".join(doc["page_content"] for doc in documents)
        result = chain.invoke({"context": context_str, "answer": answer})
    
        return "grounded" if result.get("is_grounded", False) else "not_grounded"

    @tool
    def safety_agent(answer: str):
        """Adiciona aviso de segurança à resposta final."""
        print("--[Tool]: SafetyAgent")
        disclaimer = (
            "\n\n**Aviso:** Este assistente é apenas para fins informativos "
            "e não substitui a consulta com um profissional de saúde."
        )
        return answer + disclaimer

    tools = [retriever_agent, answer_agent, self_check_agent, safety_agent]
    tool_map = {t.name: t for t in tools}

    return llm.bind_tools(tools), tool_map
