"""
You are building an AI agent that reviews buggy code and proposes fixes.
The workflow already runs correctly and produces a valid solution.
Now, you want to explore alternative review perspectives—concise, detailed, and performance-focused—without re-running everything
Using execution snapshots, you pause the workflow at a chosen checkpoint.
From the same frozen past, you branch into multiple futures.
Each branch changes only review_style, allowing precise, side-by-side comparison of outcomes.
"""

import os
import uuid
import hashlib
from typing_extensions import TypedDict, NotRequired
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI


# --- State ---
class State(TypedDict):
    prompt: NotRequired[str]
    review_style: NotRequired[str]
    llm_prompt: NotRequired[str]
    fix: NotRequired[str]
    output: NotRequired[str]


# --- Nodes ---
def set_prompt(state: State):
    return {
        "prompt": (
            "Bug: `parse_port(env)` returns None when PORT is missing.\n"
            "Code:\n"
            "  def parse_port(env):\n"
            "      if 'PORT' in env:\n"
            "          return int(env['PORT'])\n"
            "      # missing return\n"
            "Expected: return 8080 if missing."
        )
    }

def build_llm_prompt(state: State):
    style = state.get("review_style", "concise")

    style_rules = {
        "concise": (
            "Output format:\n"
            "1) Only a Python code block.\n"
            "No explanation text."
        ),
        "detailed": (
            "Output format:\n"
            "1) Exactly 3 bullet points explaining the fix.\n"
            "2) Then a Python code block."
        ),
        "performance-focused": (
            "Output format:\n"
            "1) One sentence explaining performance/efficiency impact.\n"
            "2) Then a Python code block.\n"
            "Keep it tight."
        ),
    }

    rules = style_rules.get(style, style_rules["concise"])

    return {
        "llm_prompt": (
            "You are a code review assistant.\n"
            f"{state['prompt']}\n\n"
            f"Style: {style}\n"
            f"{rules}\n\n"
            "Must fix the missing return by providing a default port of 8080."
        )
    }

def llm_propose_fix(state: State):
    msg = model.invoke(state["llm_prompt"])
    return {"fix": msg.content}


def finalize(state: State):
    return {"output": f"Style: {state.get('review_style')}\n\n{state.get('fix')}"}


# --- Graph ---
def build_graph():
    g = StateGraph(State)
    g.add_node("set_prompt", set_prompt)
    g.add_node("build_llm_prompt", build_llm_prompt)
    g.add_node("llm_propose_fix", llm_propose_fix)
    g.add_node("finalize", finalize)

    g.add_edge(START, "set_prompt")
    g.add_edge("set_prompt", "build_llm_prompt")
    g.add_edge("build_llm_prompt", "llm_propose_fix")
    g.add_edge("llm_propose_fix", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=InMemorySaver())


# --- Helpers Functions ---
def show_checkpoints(graph, config):
    states = list(reversed(list(graph.get_state_history(config))))
    print("\n=== CHECKPOINTS ===")
    for i, s in enumerate(states):
        cid = s.config["configurable"]["checkpoint_id"]
        print(f"[{i}] next={s.next} | checkpoint_id={cid}")
    return states


def h10(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:10]


def explain_snapshot(s):
    vals = dict(s.values or {})
    prompt = vals.get("prompt", "")
    llm_prompt = vals.get("llm_prompt", "")
    fix_exists = "fix" in vals

    print("\n=== SNAPSHOT YOU CHOSE ===")
    print(f"Resume will start at next node: {s.next}")
    print("What is identical across branches (the 'past'):")
    print(f"  prompt_hash     : {h10(prompt)}")
    print(f"  llm_prompt_hash : {h10(llm_prompt) if llm_prompt else '(not built yet)'}")
    print(f"  fix already made?: {fix_exists}")

    if s.next == ("llm_propose_fix",):
        print("\nInterpretation: We are paused RIGHT BEFORE the LLM decision.")
    elif s.next == ("finalize",):
        print("\nInterpretation: LLM already ran earlier; branching may not change the fix much.")
    elif s.next == ():
        print("\nInterpretation: This is the END checkpoint; nothing will run on resume.")
    else:
        print("\nInterpretation: You paused earlier; more of the pipeline will re-run on resume.")


def print_branch_results(results):
    print("\n=== BRANCH RESULTS (compare full fixes) ===")
    for r in results:
        print("\n" + "=" * 80)
        print(f"BRANCH {r['branch']} | review_style={r['review_style']}")
        print(f"Resumed from: {r['resume_next']}")
        print("-" * 80)
        print("Fix output:\n")
        print(r["fix"].strip() or "(no fix produced)")
    print("\n" + "=" * 80)


def main():
    load_dotenv()
    global model

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set.")

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    graph = build_graph()

    thread_id = str(uuid.uuid4())
    base_config = {"configurable": {"thread_id": thread_id}}

    # 1) Baseline run to create checkpoint history
    print("\nRUN #1 (baseline) - creating checkpoint history")
    graph.invoke({"review_style": "concise"}, base_config)

    # 2) Choose checkpoint
    states = show_checkpoints(graph, base_config)
    idx = int(input("\nPick checkpoint index to branch from: ").strip())
    selected_state = states[idx]
    snapshot_config = selected_state.config

    explain_snapshot(selected_state)

    # 3) Branch from chosen snapshot
    branches = [
        ("A", "concise"),
        ("B", "detailed"),
        ("C", "performance-focused"),
    ]

    results = []
    for name, style in branches:
        branch_config = graph.update_state(snapshot_config, values={"review_style": style})
        out = graph.invoke(None, branch_config)
        results.append(
            {
                "branch": name,
                "review_style": style,
                "resume_next": str(selected_state.next),
                "fix": out.get("fix", out.get("output", "")),
            }
        )

    # 4) Show results in a compare-friendly way
    print_branch_results(results)

    print(
        "- All branches start from the SAME snapshot (same past).\n"
        "- Only `review_style` changes.\n"
        "- If you branched before `llm_propose_fix`, the LLM sees a different style in the prompt → different fix.\n"
    )


if __name__ == "__main__":
    main()
