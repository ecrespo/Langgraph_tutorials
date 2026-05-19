"""
This graph models a simple bug-analysis workflow: set a bug prompt, identify its cause,
propose a fix, and produce a final output string. LangGraph checkpoints state at each step
under a single thread_id, so we can inspect the full execution history (checkpoint_id, next node,
and state diffs) and resume execution from any chosen checkpoint by passing its checkpoint_id.
"""
import uuid
from typing_extensions import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Define State
class State(TypedDict):
    prompt: NotRequired[str]
    bug: NotRequired[str]
    fix: NotRequired[str]
    output: NotRequired[str]

# Nodes
def set_prompt(state: State):
    return {"prompt": "Bug: Python function returns None unexpectedly"}  

def identify_bug(state: State):
    return {"bug": "Missing return statement in a branch"}

def propose_fix(state: State):
    return {"fix": "Add an explicit return value in every branch"}

def finalize(state: State):
    return {
        "output": f"{state['prompt']} | Cause: {state['bug']} | Fix: {state['fix']}"
    }

def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("set_prompt", set_prompt)
    workflow.add_node("identify_bug", identify_bug)
    workflow.add_node("propose_fix", propose_fix)
    workflow.add_node("finalize", finalize)

    workflow.add_edge(START, "set_prompt")
    workflow.add_edge("set_prompt", "identify_bug")
    workflow.add_edge("identify_bug", "propose_fix")
    workflow.add_edge("propose_fix", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile(checkpointer=InMemorySaver())

# Helper: Execution log visualization
def state_diff(prev_vals: dict, curr_vals: dict):
    prev_vals = prev_vals or {}
    curr_vals = curr_vals or {}
    changes = {}
    for k in sorted(set(prev_vals) | set(curr_vals)):
        if prev_vals.get(k) != curr_vals.get(k):
            changes[k] = {"from": prev_vals.get(k), "to": curr_vals.get(k)}
    return changes

def show_timeline(graph, config):
    states = list(graph.get_state_history(config))
    states = list(reversed(states))  # chronological

    print("\n=== EXECUTION TIMELINE ===")
    prev = None
    for i, s in enumerate(states):
        cid = s.config["configurable"]["checkpoint_id"]
        nxt = s.next
        vals = dict(s.values or {})
        diff = state_diff(prev.values if prev else {}, vals)
        print(f"\nStep {i}")
        print(" checkpoint_id:", cid)
        print(" next:", nxt)
        print(" state_diff:", diff)
        prev = s

    return states

def checkpoint_exists(timeline, checkpoint_id: str) -> bool:
    return any(
        s.config["configurable"]["checkpoint_id"] == checkpoint_id
        for s in timeline
    )

def main():
    graph = build_graph()

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\nRUN INITIAL EXECUTION")
    out = graph.invoke({}, config)
    print("Final output:", out["output"])

    timeline = show_timeline(graph, config)

    print("\n" + "-" * 100)
    print("Resume execution from any checkpoint shown above.")
    checkpoint_id = input("checkpoint_id: ").strip()

    if not checkpoint_exists(timeline, checkpoint_id):
        print("\nInvalid checkpoint_id.")
        return

    print("\nRESUMING FROM CHECKPOINT")
    resume_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id
        }
    }

    out2 = graph.invoke(None, resume_config)

    if "output" in out2:
        print("Final output:", out2["output"])
    else:
        print("Final state:", out2)

    print("\nTimeline after resuming (new checkpoints appended)")
    show_timeline(graph, config)

if __name__ == "__main__":
    main()


