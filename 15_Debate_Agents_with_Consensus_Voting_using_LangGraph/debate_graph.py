from typing import List, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END


class State(MessagesState):
    """Shared state for debate and voting."""
    votes: List[str]  # Stores YES/NO votes from agents


def debate_agent(llm, name: str, stance: str):
    def node(state: State) -> dict:
        # Agent presents its argument
        print(f"→ {name.upper()} presenting argument...")

        # Debate topic from last user message
        topic = state["messages"][-1].content

        # Role and stance definition for the agent
        system_prompt = SystemMessage(
            content=(
                f"Role: Debate Agent\n"
                f"Stance: {stance}\n\n"
                "Provide a concise argument.\n"
                "End clearly with either YES or NO."
            )
        )

        response = llm.invoke([system_prompt, HumanMessage(content=topic)])
        content = response.content.strip()

        # Extract vote from agent response
        vote = "YES" if "YES" in content.upper() else "NO"

        # Append vote and message to shared state
        return {
            "votes": [vote],
            "messages": [HumanMessage(content=content, name=name)]
        }

    return node


def judge_node():
    def node(state: State) -> dict:
        # Aggregate votes from all agents
        print("→ JUDGE aggregating votes...")

        votes = state.get("votes", [])
        yes_votes = votes.count("YES")
        no_votes = votes.count("NO")

        # Majority decision logic
        if yes_votes > no_votes:
            decision = "YES"
        elif no_votes > yes_votes:
            decision = "NO"
        else:
            decision = "TIE"

        summary = (
            f"Votes collected: {votes}\n"
            f"Final decision (majority voting): {decision}"
        )

        # Append judge decision to message history
        return {
            "messages": [HumanMessage(content=summary, name="judge")]
        }

    return node


def build_graph(llm):
    graph = StateGraph(State)

    # Add debate agents
    graph.add_node(
        "agent_a",
        debate_agent(llm, "agent_a", "Strongly support the proposal"),
    )
    graph.add_node(
        "agent_b",
        debate_agent(llm, "agent_b", "Oppose the proposal"),
    )
    graph.add_node(
        "agent_c",
        debate_agent(llm, "agent_c", "Neutral analytical perspective"),
    )

    # Add judge node
    graph.add_node("judge", judge_node())

    # Define debate flow
    graph.add_edge(START, "agent_a")
    graph.add_edge("agent_a", "agent_b")
    graph.add_edge("agent_b", "agent_c")
    graph.add_edge("agent_c", "judge")
    graph.add_edge("judge", END)

    return graph.compile()
