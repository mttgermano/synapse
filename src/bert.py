class Bert:
    def __init__(self,retriever):
        biobert_model_name = "dmis-lab/biobert-large-cased-v1.1-squad"
        self.tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(biobert_model_name)
    
    def ask(self,question):
        p = pipeline(
            "question-answering",
            model=biobert_model,
            tokenizer=biobert_tokenizer
        )

        # TODO tem que ver o output disso aqui  
        rag_result = self.retriever.invoke(question)

        docs = [r["text"] for r in rag_result['retrieved']]
        metadatas = [r["metadata"] for r in rag_result['retrieved']]

        sep = " <SEP> "
        context = sep.join(docs)

        result = p({
            "context": context,
            "question": question
        })

        start, end = result["start"], result["end"]
        answer = result["answer"]

        cumulative = 0
        doc_index = None
        for i, doc in enumerate(docs):
            doc_len = len(doc)
            if cumulative <= start < cumulative + doc_len:
                doc_index = i
                break
            cumulative += doc_len + len(sep)


        citation = metadatas[doc_index] if doc_index is not None else None
        citation = ast.literal_eval(citation['source']) if citation is not None else " "

        output = "-[BioBert]--\nAnswer: "+answer + "\n\nBased Articles:\n"

        for c in citation: output += c[0] + "\n"

        return output
