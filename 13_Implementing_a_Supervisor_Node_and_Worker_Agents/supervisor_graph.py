from __future__ import annotations
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command


class State(MessagesState):
    """Shared state across supervisor and workers."""
    next: str  # Next node decision made by supervisor


def make_supervisor_node(llm, members: List[str]):
    # System prompt defining strict routing rules for the supervisor
    system_prompt = (
        "You are a supervisor managing a team of workers.\n"
        f"Workers: {members}\n\n"
        "STRICT RULES:\n"
        "- You MUST choose exactly one of the following options:\n"
        f"  {members + ['FINISH']}\n"
        "- Start with researcher.\n"
        "- Then analyst.\n"
        "- Then writer.\n"
        "- Choose FINISH ONLY after writer has responded.\n\n"
        "Return ONLY ONE WORD from the allowed options."
    )

    def supervisor(state: State) -> Command:
        # Combine system prompt with conversation history
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = llm.invoke(messages)

        # Normalize supervisor decision
        choice = response.content.strip().upper()

        # Fallback safety if model returns invalid choice
        if choice not in {m.upper() for m in members} | {"FINISH"}:
            choice = "RESEARCHER"

        # End graph execution
        if choice == "FINISH":
            return Command(goto=END)

        # Route to selected worker
        return Command(goto=choice.lower())

    return supervisor


def make_worker_node(llm, worker_name: str, role_prompt: str):
    def worker(state: State) -> Command:
        # Worker receives role instructions + full message history
        messages: List[BaseMessage] = [
            SystemMessage(content=role_prompt),
            *state["messages"],
        ]

        response = llm.invoke(messages)

        # Append worker output and return control to supervisor
        return Command(
            update={
                "messages": [
                    HumanMessage(content=response.content, name=worker_name)
                ]
            },
            goto="supervisor",
        )

    return worker


def build_graph(llm):
    # Ordered list of worker roles
    workers = ["researcher", "analyst", "writer"]

    graph = StateGraph(State)

    # Add supervisor node
    graph.add_node("supervisor", make_supervisor_node(llm, workers))

    # Add worker nodes
    graph.add_node(
        "researcher",
        make_worker_node(
            llm,
            "researcher",
            "Role: Researcher\nGather background information and key points.",
        ),
    )

    graph.add_node(
        "analyst",
        make_worker_node(
            llm,
            "analyst",
            "Role: Analyst\nAnalyze the problem and outline the approach.",
        ),
    )

    graph.add_node(
        "writer",
        make_worker_node(
            llm,
            "writer",
            "Role: Writer\nProduce the final clear response.",
        ),
    )

    # Entry point of the graph
    graph.add_edge(START, "supervisor")

    return graph.compile()
