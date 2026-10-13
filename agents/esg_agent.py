from agents.base_rag_agent import run_rag_agent


def run(query: str, conversation_history: list[dict] | None = None) -> dict:
    return run_rag_agent(
        query=query,
        source_type="esg",
        source_label="ESG / GHG Protocol",
        conversation_history=conversation_history,
    )
