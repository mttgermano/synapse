from src.bert import Bert
from src.workflow import Workflow
from ingest.vdb import start_vdb

import streamlit as st

@st.cache_resource
def load_workflow():
    retriever = start_vdb()  
    bert = Bert(retriever)
    w = Workflow(retriever)
    return bert, w

workflow, w= load_workflow()

def app():
    st.set_page_config(
        page_title="Synapse: BioAsk RAG Client",
        #layout="centered"
        layout="wide"
    )

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

    models = ["Bert", "BioBert", "BioBertSquad","BioBertSquad + Mistral"]
    model_choice = st.selectbox("", models+["All"])

    st.session_state["messages"] = [{"role": "assistant", "content": "Pergunte-me sobre informações biomédicas do estado da arte!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input():
        user_msg = {"role": "user", "content": prompt}
        st.chat_message("user").write(prompt)
        
        if model_choice == "All":
            col1, col2, col3 = st.columns(3)

            with col1:
                msg1 = workflow.ask(prompt,models[0])
                st.chat_message("assistant").write(f"**[{models[0]}]**\n\n {msg1}")

            with col2:
                msg2 = workflow.ask(prompt,models[1])
                st.chat_message("assistant").write(f"**[{models[1]}]**\n\n {msg2}")

            with col3:
                msg3 = workflow.ask(prompt,models[2])
                st.chat_message("assistant").write(f"**[{models[2]}]**\n\n {msg3}")
        elif model_choice == "BioBertSquad + Mistral":
            w.ask(prompt)

        else:
            msg = workflow.ask(prompt,model_choice)
            st.chat_message("assistant").write(msg)

app()
