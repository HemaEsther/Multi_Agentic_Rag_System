from agents.base_rag_agent import run_rag_agent


def run(query: str, conversation_history: list[dict] | None = None) -> dict:
    return run_rag_agent(
        query=query,
        source_type="competitors",
        source_label="Competitor Annual Reports",
        conversation_history=conversation_history,
    )
