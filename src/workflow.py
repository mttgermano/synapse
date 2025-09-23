from src.llm import init_llm 
import requests
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import operator
from typing import List, TypedDict, Annotated

class GraphState(TypedDict):
    messages: Annotated[list, operator.add]

class Workflow:
    def __init__(self): 
        self.llm = init_llm()
        self.workflow = StateGraph(GraphState)
        self.init_workflow()
        
        self.w = self.workflow.compile()

    def tool_node(self, state):
        """
        Este nó executa a ferramenta que o Supervisor decidiu chamar,
        usando nosso mapa de ferramentas manual.
        """
        print("---EXECUTANDO FERRAMENTA (MANUALMENTE)---")
        tool_calls = state['messages'][-1].tool_calls
        tool_messages = []
    
        for call in tool_calls:
            tool_name = call['name']
            tool_to_call = tool_map[tool_name]
            tool_args = call['args']
    
            output = tool_to_call.invoke(tool_args)
    
            tool_messages.append(
                ToolMessage(content=str(output), tool_call_id=call['id'])
            )
    
        return {"messages": tool_messages}
    
    def supervisor_node(self, state):
        print("---CHAMANDO SUPERVISOR---")
        response = self.llm.invoke(state['messages'])
        return {"messages": [response]}
    
    def should_continue(self, state):
        if not state['messages'][-1].tool_calls:
            return "end"
        else:
            return "continue"

    def init_workflow(self):
        self.workflow.add_node("supervisor", self.supervisor_node)
        self.workflow.add_node("tools", self.tool_node)
        
        self.workflow.set_entry_point("supervisor")
        
        self.workflow.add_conditional_edges(
            "supervisor",
            self.should_continue,
            {"continue": "tools", "end": END},
        )
        self.workflow.add_edge("tools", "supervisor")
    
    def ask(self, question: str) -> str:
        initial_state = {"messages": [HumanMessage(content=question)]}
        final_state = self.w.invoke(initial_state)
        
        final_answer = final_state['messages'][-1].content
        
        return final_answer
