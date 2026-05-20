from langchain_core.messages import HumanMessage
from llm_provider import get_llm
from message_passing_graph import build_graph


def banner():
    # Print header
    print("\n" + "=" * 60)
    print(" Message Passing Across Agent Nodes (LangGraph + Claude)")
    print("=" * 60)
    print("Flow: generator → reviewer → refiner\n")


def print_trace(messages):
    # Print messages passed between nodes
    print("\n--- Message Trace ---\n")
    for i, m in enumerate(messages, start=1):
        name = getattr(m, "name", None) or "user/system"
        print(f"{i:02d}. [{name.upper()}]")
        print(m.content)
        print()


def main():
    banner()

    # Initialize LLM and LangGraph app
    llm = get_llm()
    app = build_graph(llm)

    # Read user task
    task = input("Enter a task (or exit()): ").strip()
    if task.lower() in {"exit()", "exit", "quit"}:
        print("\nExiting.\n")
        return
    if not task:
        print("Please provide a valid task.\n")
        return

    print("\n--- Graph Execution Started ---\n")

    # Invoke the graph with the initial user message
    final_state = app.invoke({"messages": [HumanMessage(content=task)]})

    # Display full message history
    print_trace(final_state["messages"])

    # Display additional state stored by the graph
    print("--- State Artifacts ---\n")
    print("DRAFT (stored in state['draft']):\n")
    print(final_state.get("draft", ""))
    print("\nFEEDBACK (stored in state['feedback']):\n")
    print(final_state.get("feedback", ""))
    print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
