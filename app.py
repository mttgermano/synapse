from src.workflow import Workflow
from ingest.vdb import start_vdb

import streamlit as st


@st.cache_resource
def load_workflow():
    retriever = start_vdb()  
    workflow = Workflow(retriever)
    return workflow

workflow = load_workflow()

def app():
    st.title("[🪼Synapse] BioAsk RAG Client")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me about science!"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        st.chat_message("user").write(prompt)
    
        msg = workflow.ask(prompt)
    
        assistant_msg = {"role": "assistant", "content": msg}
        st.session_state.messages.append(assistant_msg)
        st.chat_message("assistant").write(msg)

app()
