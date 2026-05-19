from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, MessagesState, START, END


class State(MessagesState):
    # Shared state across nodes
    draft: str
    feedback: str


def generator_node(llm):
    def _node(state: State) -> dict:
        print("→ Generator node: creating a first draft...")

        # Get latest user task
        task = state["messages"][-1].content

        # System prompt for generator
        sys = SystemMessage(
            content=(
                "Role: Generator\n"
                "Create a concise first draft based on the user's task.\n"
                "Output: a short, structured draft.\n"
            )
        )

        # Generate draft
        resp = llm.invoke([sys, HumanMessage(content=task)])
        draft = resp.content.strip()

        # Append generator output to messages
        return {
            "draft": draft,
            "messages": [HumanMessage(content=draft, name="generator")]
        }

    return _node


def reviewer_node(llm):
    def _node(state: State) -> dict:
        print("→ Reviewer node: providing feedback on the draft...")

        # System prompt for reviewer
        sys = SystemMessage(
            content=(
                "Role: Reviewer\n"
                "Review the draft and provide actionable feedback.\n"
                "Focus on clarity, completeness, and correctness.\n"
                "Output: bullet points.\n"
            )
        )

        # Get draft from state
        draft = state.get("draft", "")

        # Generate feedback
        resp = llm.invoke([sys, HumanMessage(content=draft)])
        feedback = resp.content.strip()

        # Append reviewer feedback to messages
        return {
            "feedback": feedback,
            "messages": [HumanMessage(content=feedback, name="reviewer")]
        }

    return _node


def refiner_node(llm):
    def _node(state: State) -> dict:
        print("→ Refiner node: producing the improved final version...")

        # System prompt for refiner
        sys = SystemMessage(
            content=(
                "Role: Refiner\n"
                "Revise the draft using the reviewer feedback.\n"
                "Output: final polished version.\n"
            )
        )

        # Read draft and feedback from state
        draft = state.get("draft", "")
        feedback = state.get("feedback", "")

        # Build refinement prompt
        prompt = (
            "Draft:\n"
            f"{draft}\n\n"
            "Feedback:\n"
            f"{feedback}\n\n"
            "Return the revised final answer:"
        )

        # Generate refined output
        resp = llm.invoke([sys, HumanMessage(content=prompt)])
        final_answer = resp.content.strip()

        # Append final output to messages
        return {
            "messages": [HumanMessage(content=final_answer, name="refiner")]
        }

    return _node


def build_graph(llm):
    # Create stateful graph
    graph = StateGraph(State)

    # Register graph nodes
    graph.add_node("generator", generator_node(llm))
    graph.add_node("reviewer", reviewer_node(llm))
    graph.add_node("refiner", refiner_node(llm))

    # Define linear execution flow
    graph.add_edge(START, "generator")
    graph.add_edge("generator", "reviewer")
    graph.add_edge("reviewer", "refiner")
    graph.add_edge("refiner", END)

    return graph.compile()
