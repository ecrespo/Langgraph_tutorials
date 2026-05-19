from langchain_core.messages import HumanMessage
from llm_provider import get_llm
from debate_graph import build_graph

print(" Debate Agents with Consensus Voting (LangGraph + Gemini)")

def print_trace(messages):
    # Display ordered debate messages
    print("\n--- Debate Trace ---\n")
    for i, m in enumerate(messages, start=1):
        name = getattr(m, "name", None)
        role = name.upper() if name else "USER/SYSTEM"

        print(f"{i:02d}. [{role}]")
        print(m.content)
        print()


def main():
   
 # Initialize LLM and debate graph
    llm = get_llm()
    app = build_graph(llm)

    # Read debate topic
    topic = input("Enter a debate topic (or exit()): ").strip()
    if topic.lower() in {"exit()", "exit", "quit"}:
        print("\nExiting.\n")
        return

    # Validate input
    if not topic:
        print("Please enter a valid topic.\n")
        return

    print("\n--- Debate Started ---\n")

    # Invoke debate graph
    final_state = app.invoke(
        {"messages": [HumanMessage(content=topic)]}
    )

    # Print debate results
    print_trace(final_state["messages"])
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
