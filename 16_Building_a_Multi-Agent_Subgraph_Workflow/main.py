from langchain_core.messages import HumanMessage
from llm_provider import get_llm
import parent_graph


def print_trace(messages):
    print("\n--- Execution Trace ---\n")

    for i, m in enumerate(messages, start=1):
        name = getattr(m, "name", None)
        role = name.upper() if name else "USER/SYSTEM"

        print(f"{i:02d}. [{role}]")
        print(m.content)
        print()


def main():
    # Initialize shared LLM instance
    llm = get_llm()

    # Build graph using the function that ACTUALLY exists
    app = parent_graph.build_subgraph(llm)

    # Accept user input
    task = input("Enter a task (or exit()): ").strip()

    if task.lower() in {"exit()", "exit", "quit"}:
        print("\nExiting.\n")
        return

    print("\n--- Workflow Started ---\n")

    # Invoke the graph
    final_state = app.invoke(
        {"messages": [HumanMessage(content=task)]}
    )

    print_trace(final_state["messages"])
    print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
