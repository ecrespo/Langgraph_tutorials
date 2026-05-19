from __future__ import annotations
import os
import json
import uuid
import operator
from typing import Any, Dict, List, TypedDict, Annotated, Sequence, Callable, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Please set it in your environment or .env file.")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

# State
class PlannerState(TypedDict):
    messages: Annotated[Sequence[str], operator.add]
    history: Annotated[Sequence[str], operator.add]
    counter: int

    task: str
    plan: List[Dict[str, Any]]
    step_index: int

    artifacts: Dict[str, Any]
    step_log: List[Dict[str, Any]]
    final_output: str


def pretty_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# Nodes
def input_node(state: PlannerState) -> Dict[str, Any]:
    print("\n=== INPUT NODE ===")
    print(f"Task: {state['task']}")
    return {
        "messages": [f"User task: {state['task']}"],
        "history": ["input_node"],
        "counter": state["counter"] + 1,
    }


ALLOWED_ACTIONS = {"analyze_task", "organize_plan", "write_final_output"}

def _fallback_plan() -> List[Dict[str, Any]]:
    return [
        {"id": 1, "action": "analyze_task", "args": {}},
        {"id": 2, "action": "organize_plan", "args": {}},
        {"id": 3, "action": "write_final_output", "args": {}},
    ]

def _validate_plan(plan: Any) -> List[Dict[str, Any]]:
    """
    Deterministic validation / normalization:
    - must be a list
    - each step must have {id:int, action:str in ALLOWED_ACTIONS, args:dict}
    - re-number ids if needed
    """
    if not isinstance(plan, list) or not plan:
        raise ValueError("Plan must be a non-empty list.")

    normalized: List[Dict[str, Any]] = []
    for idx, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Step {idx} must be an object/dict.")
        action = step.get("action")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Allowed: {sorted(ALLOWED_ACTIONS)}")
        args = step.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("args must be an object/dict.")

        normalized.append(
            {
                "id": int(step.get("id", idx)),
                "action": action,
                "args": args,
            }
        )

    # Ensure ordered ids 1..n (nice for teaching)
    for i, s in enumerate(normalized, start=1):
        s["id"] = i

    return normalized

def plan_node(state: PlannerState) -> Dict[str, Any]:
    print("\n=== PLAN NODE (LLM) ===")

    model = get_llm()
    prompt = f"""
You are a planning assistant.

Create a SMALL ordered plan to solve the following task:

Task:
{state["task"]}

Output ONLY valid JSON matching this EXACT schema (3 steps only):
[
  {{"id": 1, "action": "analyze_task", "args": {{}}}},
  {{"id": 2, "action": "organize_plan", "args": {{}}}},
  {{"id": 3, "action": "write_final_output", "args": {{}}}}
]

Allowed actions (use only these exact strings):
- analyze_task
- organize_plan
- write_final_output

Rules:
- Exactly 3 steps
- Ordered list only
- Output JSON only (no markdown, no commentary)
""".strip()

    resp = model.invoke(prompt)
    raw = (resp.content or "").strip()

    plan: List[Dict[str, Any]]
    try:
        plan = _validate_plan(json.loads(raw))
    except Exception as e:
        print("Failed to parse/validate LLM plan. Using fallback plan.")
        print("Reason:", str(e))
        plan = _fallback_plan()

    print("Plan created:")
    print(pretty_json(plan))

    return {
        "plan": plan,
        "step_index": 0,
        "history": ["plan_node:llm"],
        "counter": state["counter"] + 1,
    }


def plan_review_node(state: PlannerState) -> Dict[str, Any]:
    print("\n=== PLAN REVIEW NODE (INTERRUPT) ===")

    decision = interrupt(
        {
            "message": "Review the plan BEFORE execution.",
            "plan": state["plan"],
            "options": ["approved", "edited", "rejected"],
            "edit_instructions": (
                "If edited, provide an 'edited_plan' as a JSON list of steps using allowed actions."
            ),
        }
    )

    status = (decision.get("status") or "rejected").lower()
    updates: Dict[str, Any] = {
        "history": [f"plan_review_node:{status}"],
        "counter": state["counter"] + 1,
    }

    if status == "approved":
        return updates

    if status == "edited":
        edited = decision.get("edited_plan")
        try:
            updates["plan"] = _validate_plan(edited)
            updates["step_index"] = 0
            return updates
        except Exception as e:
            updates["final_output"] = f"Edited plan invalid. Execution stopped. Reason: {e}"
            return updates

    # rejected (default)
    updates["final_output"] = "Plan rejected. Execution stopped."
    return updates

def _executor_analyze_task(state: PlannerState, artifacts: Dict[str, Any]) -> None:
    """
    Either:
    - Deterministic extraction (very basic), OR
    - LLM bullet requirements if available
    """
    model = get_llm()
    prompt = f"""
Extract the key requirements and constraints from this task:

{state["task"]}

Return a short bullet list.
""".strip()
    resp = model.invoke(prompt)
    artifacts["requirements"] = (resp.content or "").strip()

def _executor_organize_plan(state: PlannerState, artifacts: Dict[str, Any]) -> None:
    model = get_llm()
    prompt = f"""
Based on these requirements:

{artifacts.get("requirements", "")}

Create a structured breakdown of the plan in bullets (brief).
""".strip()
    resp = model.invoke(prompt)
    artifacts["structured_steps"] = (resp.content or "").strip()

def _executor_write_final_output(state: PlannerState, artifacts: Dict[str, Any]) -> None:

    model = get_llm()
    prompt = f"""
Task:
{state["task"]}

Requirements:
{artifacts.get("requirements", "")}

Structured steps:
{artifacts.get("structured_steps", "")}

Write the final response.
""".strip()
    resp = model.invoke(prompt)
    artifacts["final"] = (resp.content or "").strip()

ACTION_HANDLERS: Dict[str, Callable[[PlannerState, Dict[str, Any]], None]] = {
    "analyze_task": _executor_analyze_task,
    "organize_plan": _executor_organize_plan,
    "write_final_output": _executor_write_final_output,
}

def execute_step_node(state: PlannerState) -> Dict[str, Any]:
    print("\n=== EXECUTE STEP NODE ===")

    i = state["step_index"]
    plan = state["plan"]

    if i >= len(plan):
        return {
            "history": ["execute_step_node:noop"],
            "counter": state["counter"] + 1,
        }

    step = plan[i]
    action = step["action"]

    artifacts = dict(state.get("artifacts", {}))

    # Deterministic guard
    if action not in ACTION_HANDLERS:
        artifacts["error"] = f"Unknown action: {action}"
    else:
        # Execute the step
        ACTION_HANDLERS[action](state, artifacts)

    log_entry = {"step_index": i, "action": action}
    print(f"Executed step {i+1}/{len(plan)}: {action}")

    return {
        "artifacts": artifacts,
        "step_log": state.get("step_log", []) + [log_entry],
        "step_index": i + 1,
        "history": [f"execute_step_node:{action}"],
        "counter": state["counter"] + 1,
    }

def finalize_node(state: PlannerState) -> Dict[str, Any]:
    print("\n=== FINALIZE NODE ===")

    output = state.get("final_output") or state.get("artifacts", {}).get("final") or ""
    print("\nFINAL OUTPUT:\n")
    print(output)

    return {
        "final_output": output,
        "history": ["finalize_node"],
        "counter": state["counter"] + 1,
    }

def route_after_plan_review(state: PlannerState) -> str:
    return "finalize" if state.get("final_output") else "execute"


def route_execute_or_finalize(state: PlannerState) -> str:
    return "finalize" if state["step_index"] >= len(state["plan"]) else "execute"

def build_graph():
    g = StateGraph(PlannerState)

    g.add_node("input", input_node)
    g.add_node("plan", plan_node)
    g.add_node("plan_review", plan_review_node)
    g.add_node("execute", execute_step_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "input")
    g.add_edge("input", "plan")
    g.add_edge("plan", "plan_review")

    g.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {"execute": "execute", "finalize": "finalize"},
    )

    g.add_conditional_edges(
        "execute",
        route_execute_or_finalize,
        {"execute": "execute", "finalize": "finalize"},
    )

    g.add_edge("finalize", END)

    return g.compile(checkpointer=InMemorySaver())

def _read_multiline_json(prompt: str) -> Optional[Any]:
    """
    Reads multiline JSON from stdin.
    End input with a single line containing: ENDJSON
    """
    print(prompt)
    print("Paste JSON, then type ENDJSON on a new line.")
    lines: List[str] = []
    while True:
        line = input()
        if line.strip() == "ENDJSON":
            break
        lines.append(line)
    raw = "\n".join(lines).strip()
    if not raw:
        return None
    return json.loads(raw)

def run_demo():
    app = build_graph()

    print("=" * 72)
    print("Planner → Executor: Task Planning and Structured Execution".center(72))
    print("=" * 72)

    task = input("\nEnter task:\n> ").strip()
    if not task:
        task = "Create a beginner-friendly plan to learn Python in 2 weeks."

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state: PlannerState = {
        "messages": [],
        "history": [],
        "counter": 0,
        "task": task,
        "plan": [],
        "step_index": 0,
        "artifacts": {},
        "step_log": [],
        "final_output": "",
    }

    # First run
    result = app.invoke(state, config=config)

    # Handle interrupts interactively
    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value

        print("\n=== EXECUTION PAUSED FOR REVIEW ===")
        print(interrupt_payload["message"])
        print("\nPlan:\n")
        print(pretty_json(interrupt_payload["plan"]))

        print("\nChoose:")
        print("1) Approve")
        print("2) Edit plan (paste JSON)")
        print("3) Reject")

        choice = input("Enter 1/2/3: ").strip()

        decision: Dict[str, Any]
        if choice == "1":
            decision = {"status": "approved"}
        elif choice == "2":
            try:
                edited = _read_multiline_json(
                    "\nEdit the plan JSON (must be a list of 3 steps with allowed actions)."
                )
                decision = {"status": "edited", "edited_plan": edited}
            except Exception as e:
                print("Invalid JSON input:", e)
                decision = {"status": "rejected"}
        else:
            decision = {"status": "rejected"}

        # Resume execution
        result = app.invoke(Command(resume=decision), config=config)

    print("\n=== WORKFLOW COMPLETED ===")
    # Optional: print step log
    if "step_log" in result:
        print("\nStep log:")
        print(pretty_json(result["step_log"]))


if __name__ == "__main__":
    run_demo()
