from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline
#from ast import ast

class Bert:
    def __init__(self,retriever):
        #biobert_model_name = 
        self.retriever = retriever
        init_models()

    def init_models():
        # BioBert
        biobert_model_name = "dmis-lab/biobert-v1.1"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_model_name)
        self.biobert = AutoModelForQuestionAnswering.from_pretrained(biobert_model_name)

        # BioBertSquad
        biobert_squad_model_name = "dmis-lab/biobert-large-cased-v1.1-squad"
        self.biobert_squad_tokenizer = AutoTokenizer.from_pretrained(biobert_squad_model_name)
        self.biobert_squad = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)

        # BertBase
        bert_model_name = "google-bert/bert-base-uncased"
        slef.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModelForQuestionAnswering.from_pretrained(bert_model_name)

    
    def ask(self,question,model):
        if(model == "Bert"):
            return self.query(question,(
                self.bert,
                self.bert_tokenizer
            ))
        elif(model == "BioBert"):
            return self.query(question,(
                self.biobert,
                self.biobert_tokenizer
            ))
        elif(model == "BioBertSquad"):
            return self.query(question,(
                self.biobert_squad,
                self.biobert_squad_tokenizer
            ))

        return "Unsupported Model selected"


    def query(self,question,model):
        p = pipeline(
            "question-answering",
            model=model[0],
            tokenizer=model[1]
        )

        rag_result = self.retriever.invoke(question)

        docs = [doc.page_content for doc in rag_result] 
        metadatas = [doc.metadata for doc in rag_result]

        sep = " <SEP> "
        context = sep.join(docs)
        print(context)

        result = p({
            "context": context,
            "question": question
        })

        start, end = result["start"], result["end"]
        answer = result["answer"]

        intervals = []
        idx = 0
        for d in docs:
            i = (idx, idx + len(d))
            intervals.append(i)
            idx += len(d)

        docs = []
        for i, (s,e) in enumerate(intervals):
            if not (end <= s or start >= e): 
                docs.append(i)

        citation = ""
        for i in docs:
            citation += metadatas[i].get("source"," ") + "\n" 

        output = answer + "\n\nBased Articles: \n\n" + citation[1:-2]

        return output
