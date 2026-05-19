from langchain_core.messages import HumanMessage
from llm_provider import get_llm
from supervisor_graph import build_graph


def main():
    # Print application header
    print("\nSupervisor–Worker Architecture (LangGraph)\n")
    print("Enter a task. Type 'exit()' to quit.\n")

    # Initialize LLM and graph
    llm = get_llm()
    app = build_graph(llm)

    # Interactive task loop
    while True:
        user_request = input("Task: ").strip()

        # Exit condition
        if user_request.lower() in {"exit()", "exit", "quit"}:
            print("\nExiting.\n")
            return

        # Handle empty input
        if not user_request:
            print("Please enter a task.\n")
            continue

        # Invoke graph with user message
        final_state = app.invoke(
            {"messages": [HumanMessage(content=user_request)]}
        )

        print("\n========== FINAL OUTPUT ==========\n")

        # Print message trace
        for msg in final_state["messages"]:
            role = msg.name if msg.name else "user/system"
            print(f"[{role.upper()}]\n{msg.content}\n")


if __name__ == "__main__":
    main()
