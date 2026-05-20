from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END


class SubgraphState(MessagesState):
    summary: str


def researcher_node(llm):
    def node(state: SubgraphState) -> dict:
        # Visual indicator during execution
        print("   ↳ Subgraph: Researcher working...")

        # Extract the latest user-provided topic
        topic = state["messages"][-1].content

        # System instruction defining the Researcher's role
        sys = SystemMessage(
            content="Role: Researcher\nProvide key background points."
        )

        # Invoke the LLM with role + topic
        resp = llm.invoke([sys, HumanMessage(content=topic)])
        summary = resp.content.strip()

        # Store result in shared state and message history
        return {
            "summary": summary,  # Passed to the next node
            "messages": [
                HumanMessage(
                    content=summary,
                    name="sub_researcher"
                )
            ],
        }

    return node


def summarizer_node(llm):
    def node(state: SubgraphState) -> dict:
        # Visual indicator during execution
        print("   ↳ Subgraph: Summarizer refining output...")

        # System instruction defining the Summarizer's role
        sys = SystemMessage(
            content="Role: Summarizer\nCreate a concise, polished summary."
        )

        # Use the researcher's summary as input
        resp = llm.invoke(
            [sys, HumanMessage(content=state["summary"])]
        )

        # Append final refined output to message history
        return {
            "messages": [
                HumanMessage(
                    content=resp.content.strip(),
                    name="sub_summarizer",
                )
            ]
        }

    return node


def build_subgraph(llm):
    graph = StateGraph(SubgraphState)

    # Register subgraph nodes
    graph.add_node("researcher", researcher_node(llm))
    graph.add_node("summarizer", summarizer_node(llm))

    # Define linear execution flow
    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "summarizer")
    graph.add_edge("summarizer", END)

    # Compile into an executable subgraph
    return graph.compile()
