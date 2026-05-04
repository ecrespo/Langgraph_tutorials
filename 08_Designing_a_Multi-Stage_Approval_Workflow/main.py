"""
Imagine a developer using an AI agent to generate a small utility function for a personal project.
Before pushing the code to Git, it must pass through a multi-stage approval workflow that mirrors
real development checkpoints:

* code review
* testing
* manager
* And UI/UX

At each stage, the developer acts as the reviewer, approving, editing, or requesting changes.
The workflow only progresses after each checkpoint is completed, ensuring deliberate, controlled execution.
"""

from __future__ import annotations

from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import operator
import textwrap
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment. Add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
)

# State schema
class GateState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    history: Annotated[Sequence[str], operator.add]
    counter: int

    task: str
    code: str

    stages: List[str]    
    stage_index: int
    last_decision: Dict[str, Any]
    approval_log: List[Dict[str, Any]]
    next_action: str  # "revise" | "advance" | "publish" | "reject"

# Nodes
def input_node(state: GateState) -> GateState:
    print("\n=== INPUT NODE ===")
    print(f"Task: {state['task']}")
    return {
        "messages": [f"User: {state['task']}"],
        "history": ["input_node"],
        "counter": state["counter"] + 1,
    }

def generate_code_node(state: GateState) -> GateState:
    """
    LLM generates the draft function so there's something meaningful to review.
    Keep output small: ONE function, no extra commentary.
    """
    print("\n=== GENERATE CODE NODE (LLM) ===")

    prompt = f"""
    You are helping with a personal project. Generate Python code only.

    Task:
    {state["task"]}

    Rules:
    - Output ONLY valid Python code (no markdown fences, no explanation).
    - Keep it short and readable.
    - Include docstring and basic error handling.
    - Do not import external libraries.
    """

    response = llm.invoke(prompt)
    code = (response.content or "").strip()

    return {
        "code": code,
        "messages": ["Assistant: Draft code generated (LLM)."],
        "history": ["generate_code_node"],
        "counter": state["counter"] + 1,
    }

def review_node(state: GateState) -> GateState:
    stage = state["stages"][state["stage_index"]]
    print(f"\n=== REVIEW NODE (INTERRUPT) | Stage: {stage.upper()} ===")

    focus_map = {
        "code_review": "Correctness, readability, edge cases",
        "testing": "Do we need basic tests? Are cases covered?",
        "manager": "Scope is reasonable + maintainable",
        "ui_ux": "User-facing messages are clear/friendly",
    }

    payload = {
        "stage": stage,
        "message": f"Checkpoint: {stage}",
        "review_focus": focus_map.get(stage, ""),
        "artifact": state["code"],
        "options": ["approved", "edited", "changes_requested", "rejected"],
    }

    decision = interrupt(payload)

    status = (decision.get("status") or "rejected").strip().lower()
    return {
        "last_decision": decision,
        "messages": [f"Decision @ {stage}: {status}"],
        "history": [f"review_node:{stage}"],
        "counter": state["counter"] + 1,
    }

def apply_decision_node(state: GateState) -> GateState:
    print("\n=== APPLY DECISION NODE ===")
    stage = state["stages"][state["stage_index"]]
    decision = state.get("last_decision", {})
    status = (decision.get("status") or "rejected").strip().lower()

    notes = decision.get("notes", "")
    log_entry = {"stage": stage, "status": status, "reviewer": "self", "notes": notes}

    edited_code = (decision.get("edited_code") or "").strip()
    updates: Dict[str, Any] = {}

    if status in ("edited", "changes_requested") and edited_code:
        updates["code"] = edited_code
        updates["messages"] = [f"Assistant: Code updated based on {stage} feedback."]

    if status == "approved":
        next_action = "publish" if state["stage_index"] >= len(state["stages"]) - 1 else "advance"
    elif status == "edited":
        next_action = "revise"
    elif status == "changes_requested":
        next_action = "advance"
    else:
        next_action = "reject"

    print(f"Next action: {next_action}")

    return {
        **updates,
        "approval_log": state.get("approval_log", []) + [log_entry],
        "next_action": next_action,
        "history": ["apply_decision_node"],
        "counter": state["counter"] + 1,
    }

def revise_node(state: GateState) -> GateState:
    stage = state["stages"][state["stage_index"]]
    print(f"\n=== REVISE NODE | Stage: {stage.upper()} ===")
    return {
        "history": [f"revise_node:{stage}"],
        "counter": state["counter"] + 1,
    }

def advance_stage_node(state: GateState) -> GateState:
    print("\n=== ADVANCE STAGE NODE ===")
    new_index = min(state["stage_index"] + 1, len(state["stages"]) - 1)
    print(f"Stage index: {state['stage_index']} -> {new_index}")
    return {
        "stage_index": new_index,
        "history": ["advance_stage_node"],
        "counter": state["counter"] + 1,
    }

def publish_node(state: GateState) -> GateState:
    print("\n=== PUBLISH NODE ===")
    print("\nREADY TO PUSH (FINAL CODE):\n")
    print(textwrap.fill(state["code"], width=100))
    return {
        "messages": ["Assistant: All checkpoints passed. Ready to push to Git."],
        "history": ["publish_node"],
        "counter": state["counter"] + 1,
    }

def reject_node(state: GateState) -> GateState:
    print("\n=== REJECT NODE ===")
    print("Rejected. Not ready to push.")
    return {
        "messages": ["Assistant: Rejected at a checkpoint. Not ready to push."],
        "history": ["reject_node"],
        "counter": state["counter"] + 1,
    }
    
# Router 
def route_from_decision(state: GateState) -> str:
    return state.get("next_action", "reject")

# Graph builder (major highlight)
def create_graph():
    workflow = StateGraph(GateState)

    workflow.add_node("input", input_node)
    workflow.add_node("generate", generate_code_node)
    workflow.add_node("review", review_node)
    workflow.add_node("apply", apply_decision_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("advance", advance_stage_node)
    workflow.add_node("publish", publish_node)
    workflow.add_node("reject", reject_node)

    workflow.add_edge("input", "generate")
    workflow.add_edge("generate", "review")
    workflow.add_edge("review", "apply")

    #Multi-stage approval workflow:
    workflow.add_conditional_edges(
        "apply",
        route_from_decision,
        {"revise": "revise", "advance": "advance", "publish": "publish", "reject": "reject"},
    )
    workflow.add_edge("revise", "review")
    workflow.add_edge("advance", "review")

    workflow.add_edge("publish", END)
    workflow.add_edge("reject", END)

    workflow.set_entry_point("input")
    return workflow.compile(checkpointer=InMemorySaver())




def run_demo():
    app = create_graph()
    print("=" * 78)
    print("Pre-Push Code Gate (Multi-Stage Approval Workflow)".center(78))
    print("=" * 78)

    task = input("\nEnter task (e.g., 'Write sanitize_username(username: str) -> str'):\n> ").strip()
    if not task:
        print("No task provided.")
        return

    config = {"configurable": {"thread_id": "pre-push-thread-1"}}

    state: GateState = {
        "messages": [],
        "history": [],
        "counter": 0,
        "task": task,
        "code": "",
        "stages": ["code_review", "testing", "manager", "ui_ux"],
        "stage_index": 0,
        "last_decision": {},
        "approval_log": [],
        "next_action": "",
    }

    next_input = state

    while True:
        result = app.invoke(next_input, config=config)
        interrupts = result.get("__interrupt__")

        if not interrupts:
            print("\n--- Completed ---")
            for m in result.get("messages", []):
                print(m)

            print("\nApproval log:")
            for entry in result.get("approval_log", []):
                print(f"- {entry['stage']}: {entry['status']}")

            print("\nHistory:")
            print(" -> ".join(result.get("history", [])))
            break

        payload = interrupts[0].value
        stage = payload["stage"]

        print("\n" + "=" * 78)
        print(f"CHECKPOINT: {stage.upper()}".center(78))
        print("=" * 78)
        print(payload["review_focus"])
        print("\n--- CURRENT CODE ---\n")
        print(payload["artifact"])

        print("\nChoose:")
        print("1) Approve")
        print("2) Edit (paste updated code)")
        print("3) Request changes")
        print("4) Reject")

        choice = input("Enter 1/2/3/4: ").strip()
        notes = input("Notes (optional): ").strip()

        if choice == "1":
            decision = {"status": "approved", "notes": notes}
        elif choice == "2":
            edited = input("\nPaste edited code:\n> ").strip()
            decision = {"status": "edited", "notes": notes, "edited_code": edited}
        elif choice == "3":
            decision = {"status": "changes_requested", "notes": notes}
        else:
            decision = {"status": "rejected", "notes": notes}

        next_input = Command(resume=decision)


if __name__ == "__main__":
    run_demo()
