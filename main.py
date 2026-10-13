"""CLI entry point for backend testing without Streamlit."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from graph.orchestrator import run_query


def main():
    print("Financial Knowledge Assistant (CLI Mode)")
    print("Type 'quit' to exit, 'ingest' to run data ingestion.\n")

    conversation_history = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "ingest":
            from rag.ingestion import ingest_all
            ingest_all()
            continue

        result = run_query(query, conversation_history=conversation_history)

        if result.get("needs_clarification"):
            print(f"\nAssistant: {result['clarification_question']}\n")
            continue

        if not result.get("final_answer"):
            answer = input("\nNo internal sources found. Search the internet? (yes/no): ").strip().lower()
            if answer in ("yes", "y"):
                result = run_query(query, conversation_history=conversation_history, web_approved=True)
            else:
                print("Assistant: No answer available from internal sources.\n")
                continue

        print(f"\nAssistant: {result.get('final_answer', '')}")

        if result.get("detected_sources"):
            print(f"  Sources: {', '.join(result['detected_sources'])}")

        if result.get("follow_up_question"):
            print(f"\n  Follow-up: {result['follow_up_question']}")

        conversation_history.extend(result.get("conversation_history", []))
        print()


if __name__ == "__main__":
    main()
