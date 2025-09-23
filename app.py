from src.workflow import Workflow
from ingest.vdb import start_vdb

import streamlit as st

retriever = start_vdb()
workflow = Workflow(retriever)

def app():
    st.title("[🪼Synapse] BioAsk RAG Client")
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Ask me about science!"}]
    
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        st.session_state.messages.append(prompt)
        st.chat_message("user").write(prompt)

        msg = workflow.ask(st.session_state.messages[-1])

        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

app()
