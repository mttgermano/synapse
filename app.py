from src.bert import Bert
from ingest.vdb import start_vdb

import streamlit as st

@st.cache_resource
def load_workflow():
    retriever = start_vdb()  
    bert = Bert(retriever)
    return bert 

workflow = load_workflow()

def app():

    st.markdown(
        """
        <div style='
            background-color: #007BFF; 
            padding: 20px; 
            border-radius: 10px;
        '>
            <h1 style='color: white;'>🪼Synapse: BioAsk RAG Client</h1>
            <p style='color: white; text-align: center;'>
            AI assistant specialized in Evidence-Based BioMedicine
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    model_choice = st.selectbox("", ["Bert", "BioBert", "BioBertSquad"])

    st.session_state["messages"] = [{"role": "assistant", "content": "Pergunte-me sobre informações biomédicas do estado da arte!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    
    #for msg in st.session_state.messages:
    #    st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        user_msg = {"role": "user", "content": prompt}
        st.chat_message("user").write(prompt)
    
        msg = workflow.ask(prompt,model_choice)
    
        assistant_msg = {"role": "assistant", "content": msg}
        st.chat_message("assistant").write(msg)

app()
