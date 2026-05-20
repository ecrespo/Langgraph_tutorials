from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, Dict, Any
import operator
import os

from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

from langchain_anthropic import ChatAnthropic


load_dotenv()
if not os.getenv("ANTHROPIC_API_KEY"):
    raise RuntimeError("ANTHROPIC_API_KEY not set")


def build_llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        temperature=0.3,
    )


class TaskState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    history: Annotated[Sequence[str], operator.add]

    task: str
    plan: str
    draft: str
    final: str

    feedback: Dict[str, Any]

def planner_node(state: TaskState) -> TaskState:
    llm = build_llm()

    prompt = f"""
You are a task planner.

Break the following task into 3 clear steps.
Do not execute the task.

Task:
{state["task"]}
""".strip()

    plan = llm.invoke(prompt).content

    return {
        "plan": plan,
        "messages": ["Planner: task decomposed"],
        "history": ["planner_node"],
    }


def executor_node(state: TaskState) -> TaskState:
    llm = build_llm()

    prompt = f"""
You are an executor.

Follow this plan and produce a concise draft (max 150 words).

Plan:
{state["plan"]}

Task:
{state["task"]}
""".strip()

    draft = llm.invoke(prompt).content

    return {
        "draft": draft,
        "messages": ["Executor: draft created"],
        "history": ["executor_node"],
    }


def human_review_node(state: TaskState) -> TaskState:
    payload = {
        "message": "Review the draft and optionally change direction",
        "draft": state["draft"],
        "options": [
            "approve",
            "revise with feedback",
            "change focus",
        ],
    }

    decision = interrupt(payload)

    return {
        "feedback": decision,
        "messages": [f"Human review: {decision.get('action')}"],
        "history": ["human_review_node"],
    }


def revise_node(state: TaskState) -> TaskState:
    llm = build_llm()
    feedback = state["feedback"]

    if feedback.get("action") == "approve":
        return {
            "final": state["draft"],
            "messages": ["Executor: approved without changes"],
            "history": ["revise_node"],
        }

    prompt = f"""
Revise the draft based on human feedback.

Draft:
{state["draft"]}

Feedback:
{feedback.get("notes", "No notes")}
""".strip()

    revised = llm.invoke(prompt).content

    return {
        "final": revised,
        "messages": ["Executor: draft revised"],
        "history": ["revise_node"],
    }

def create_graph():
    graph = StateGraph(TaskState)

    graph.add_node("plan", planner_node)
    graph.add_node("execute", executor_node)
    graph.add_node("review", human_review_node)
    graph.add_node("revise", revise_node)

    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "review")
    graph.add_edge("review", "revise")
    graph.add_edge("revise", END)

    graph.set_entry_point("plan")

    memory = InMemorySaver()
    return graph.compile(checkpointer=memory)


def run_demo():
    app = create_graph()

    initial_state: TaskState = {
        "messages": [],
        "history": [],
        "task": "Explain how API rate limiting works for beginners",
        "plan": "",
        "draft": "",
        "final": "",
        "feedback": {},
    }

    result = app.invoke(
        initial_state,
        config={"configurable": {"thread_id": "task-chain-demo"}},
    )

    interrupt_obj = result["__interrupt__"][0]
    payload = interrupt_obj.value

    print("\n--- HUMAN REVIEW ---")
    print(payload["draft"])

    print("\nChoose action:")
    print("1) approve")
    print("2) revise")
    print("3) change focus")

    choice = input("> ").strip()

    if choice == "1":
        decision = {"action": "approve"}
    elif choice == "2":
        notes = input("Revision notes: ")
        decision = {"action": "revise", "notes": notes}
    else:
        notes = input("New focus: ")
        decision = {"action": "change focus", "notes": notes}

    final = app.invoke(
        Command(resume=decision),
        config={"configurable": {"thread_id": "task-chain-demo"}},
    )

    print("\n--- FINAL OUTPUT ---\n")
    print(final["final"])

    print("\n--- EXECUTION TRACE ---")
    print(" -> ".join(final["history"]))


if __name__ == "__main__":
    run_demo()
