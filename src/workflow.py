from src.llm import init_llm
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult, Generation
from typing import Any, List
from typing import Any, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation, LLMResult
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import PrivateAttr


class GraphState(TypedDict):
    messages: Annotated[list, operator.add]


class Workflow:
    def __init__(self, retriever):
        self.llm, self.tool_map = init_llm(retriever)
        self.workflow = StateGraph(GraphState)
        self.init_workflow()
        self.w = self.workflow.compile()

    def tool_node(self, state):
        print("[*] EXECUTANDO FERRAMENTA")
        tool_calls = state["messages"][-1].tool_calls
        tool_messages = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_to_call = self.tool_map[tool_name]
            tool_args = call["args"]

            output = tool_to_call.invoke(tool_args)
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        return {"messages": tool_messages}

    def supervisor_node(self, state):
        print("[*] CHAMANDO SUPERVISOR")
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def should_continue(self, state):
        return "continue" if state["messages"][-1].tool_calls else "end"

    def init_workflow(self):
        self.workflow.add_node("supervisor", self.supervisor_node)
        self.workflow.add_node("tools", self.tool_node)

        self.workflow.set_entry_point("supervisor")
        self.workflow.add_conditional_edges(
            "supervisor", self.should_continue, {"continue": "tools", "end": END}
        )
        self.workflow.add_edge("tools", "supervisor")

    def ask(self, question: str) -> str:
        initial_state = {"messages": [HumanMessage(content=question)]}
        final_state = self.w.invoke(initial_state)
        return final_state["messages"][-1].content

class WorkflowLLM(BaseLanguageModel):
    # Atributo privado (não gerenciado pelo Pydantic)
    _workflow: Any = PrivateAttr()

    def __init__(self, workflow, **kwargs):
        super().__init__(**kwargs)
        self._workflow = workflow

    # ======================
    # Métodos síncronos
    # ======================
    def predict(self, text: str, **kwargs) -> str:
        return self._workflow.ask(text)

    def predict_messages(self, messages, **kwargs) -> AIMessage:
        text = " ".join(m.content for m in messages if isinstance(m, HumanMessage))
        return AIMessage(content=self._workflow.ask(text))

    def generate_prompt(self, prompt, **kwargs) -> str:
        return self._workflow.ask(str(prompt))

    # ======================
    # Métodos assíncronos
    # ======================
    async def apredict(self, text: str, **kwargs) -> str:
        return self._workflow.ask(text)

    async def apredict_messages(self, messages, **kwargs) -> AIMessage:
        text = " ".join(m.content for m in messages if isinstance(m, HumanMessage))
        return AIMessage(content=self._workflow.ask(text))

    async def agenerate_prompt(self, prompt, **kwargs) -> str:
        return self._workflow.ask(str(prompt))

    # ======================
    # Invoke compatível com LangChain
    # ======================
    def invoke(self, prompt, **kwargs) -> str:
        if isinstance(prompt, str):
            return self._workflow.ask(prompt)
        elif isinstance(prompt, list):
            last = prompt[-1]
            return self._workflow.ask(getattr(last, "content", str(last)))
        else:
            return self._workflow.ask(str(prompt))

    # ======================
    # Ragas precisa desse hook
    # ======================
    def set_run_config(self, config: Any):
        self._run_config = config

    # ======================
    # Métodos exigidos pelo BaseLanguageModel
    # ======================
    @property
    def _llm_type(self) -> str:
        return "workflow-llm"

    def _generate(
        self, prompts: List[str], stop: Any = None, run_manager: Any = None
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            output = self._workflow.ask(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)
