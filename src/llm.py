from langchain_google_genai import ChatGoogleGenerativeAI


def init_llm_pipeline():
    pipeline = [llm_base, bind_tools]
    out = ""
    for process in pipeline:
        out = process(out)

    return out 


def llm_base(kw): 
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        google_api_key="AIzaSyD5rbP-fsG3NB2gQo2tgfDjyRPMRlNpB-0"
    )
    return llm

def bind_tools(llm):
    @tool
    def retriever_agent(question: str):
        """
        Busca em uma base de dados de documentos de saúde pública (como o Guia Alimentar)
        para encontrar informações relevantes para responder à pergunta do usuário.
        """
        print("---FERRAMENTA: RetrieverAgent---")
        documents = retriever.invoke(question)
        for doc in documents:
            doc.metadata['source'] = f"{os.path.basename(doc.metadata['source'])}, pág. {doc.metadata['page']}"
        return documents
    
    @tool
    def answer_agent(question: str, documents: List[dict]):
        """
        Gera uma resposta detalhada e baseada em fatos para a pergunta do usuário,
        utilizando exclusivamente os documentos fornecidos como contexto.
        Sempre cita as fontes para cada informação fornecida.
        """
        print("---FERRAMENTA: AnswerAgent---")
        prompt_template = """
        Você é um assistente especializado em responder perguntas sobre saúde pública com base em documentos oficiais.
        Responda à pergunta do usuário utilizando SOMENTE as informações contidas no contexto abaixo.
        Se a informação não estiver no contexto, responda: "Não encontrei informações sobre isso nos documentos fornecidos."
        Para cada afirmação que você fizer, cite a fonte usando o formato [fonte: metadata.source].
    
        Contexto: {context}
        Pergunta: {question}
        Resposta:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        rag_chain = prompt | llm | StrOutputParser()
        context_str = "\n\n".join(f"[fonte: {doc['metadata']['source']}]\n{doc['page_content']}" for doc in documents)
        answer = rag_chain.invoke({"context": context_str, "question": question})
        return {"answer": answer, "documents": documents}
    
    @tool
    def self_check_agent(answer: str, documents: List[dict]):
        """
        Verifica se uma resposta generada é totalmente suportada pelos documentos de contexto fornecidos.
        Use esta ferramenta para garantir a veracidade e evitar alucinações.
        Retorna 'grounded' se a resposta for fiel ou 'not_grounded' caso contrário.
        """
        print("---FERRAMENTA: SelfCheckAgent---")
        class Verdict(BaseModel):
            is_grounded: bool = Field(description="A resposta é totalmente suportada pelos trechos de contexto fornecidos? (true/false)")
        parser = JsonOutputParser(pydantic_object=Verdict)
        prompt_template = """
        Você é um avaliador especialista. Sua tarefa é verificar se a 'Resposta' fornecida é totalmente suportada pelas informações no 'Contexto'.
        Responda APENAS com um objeto JSON contendo a chave 'is_grounded' (true ou false).
    
        Contexto: {context}
        Resposta a ser verificada: {answer}
        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template, partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt | llm | parser
        context_str = "\n\n".join(doc['page_content'] for doc in documents)
        result = chain.invoke({"context": context_str, "answer": answer})
        if result.get('is_grounded', False):
            print("---DECISÃO: RESPOSTA FIEL---")
            return "grounded"
        else:
            print("---DECISÃO: RESPOSTA NÃO FIEL---")
            return "not_grounded"
    
    @tool
    def safety_agent(answer: str):
        """
        Adiciona um disclaimer de segurança e informativo a uma resposta final antes de exibi-la ao usuário.
        Use esta ferramenta como a última etapa do processo.
        """
        print("---FERRAMENTA: SafetyAgent---")
        disclaimer = "\n\n**Aviso:** Este assistente é apenas para fins informativos e não substitui a consulta com um profissional de saúde. As informações são baseadas em documentos públicos e podem não estar completas ou atualizadas."
        return answer + disclaimer

    tools = [retriever_agent, answer_agent, self_check_agent, safety_agent]
    return llm.bind_tools(tools)
