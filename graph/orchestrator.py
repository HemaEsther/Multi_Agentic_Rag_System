from typing import Annotated, Optional

from google import genai
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict

from agents import competitor_agent, esg_agent, ifrs_agent, web_agent
from agents.source_classifier import check_ambiguity, classify_sources
from config.settings import GEMINI_API_KEY, LLM_MODEL

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _merge_list(a: list, b: list) -> list:
    return a + b


class AgentState(TypedDict):
    query: str
    conversation_history: Annotated[list[dict], _merge_list]
    detected_sources: list[str]
    needs_clarification: bool
    clarification_question: str
    esg_result: Optional[dict]
    ifrs_result: Optional[dict]
    competitor_result: Optional[dict]
    web_result: Optional[dict]
    web_approved: bool
    final_answer: str
    follow_up_question: str
    all_visual_content: list[dict]
    all_sources: list[dict]


def node_check_ambiguity(state: AgentState) -> dict:
    result = check_ambiguity(state["query"])
    return {
        "needs_clarification": result["is_ambiguous"],
        "clarification_question": result["clarification_question"],
    }


def node_classify_sources(state: AgentState) -> dict:
    sources = classify_sources(state["query"])
    return {"detected_sources": sources}


def node_run_esg(state: AgentState) -> dict:
    result = esg_agent.run(state["query"], state.get("conversation_history"))
    return {"esg_result": result}


def node_run_ifrs(state: AgentState) -> dict:
    result = ifrs_agent.run(state["query"], state.get("conversation_history"))
    return {"ifrs_result": result}


def node_run_competitor(state: AgentState) -> dict:
    result = competitor_agent.run(state["query"], state.get("conversation_history"))
    return {"competitor_result": result}


def node_request_web_approval(state: AgentState) -> dict:
    approved = interrupt({
        "type": "web_approval",
        "message": "I couldn't find an answer in the internal documents. Do you want me to search the internet for this?",
    })
    return {"web_approved": bool(approved)}


def node_run_web(state: AgentState) -> dict:
    result = web_agent.run(state["query"])
    return {"web_result": result}


def node_synthesize(state: AgentState) -> dict:
    results = []
    all_visual = []
    all_sources = []

    for key in ("esg_result", "ifrs_result", "competitor_result", "web_result"):
        r = state.get(key)
        if r and r.get("answer") and "No relevant information" not in r["answer"]:
            label = {
                "esg_result": "ESG/GHG Protocol",
                "ifrs_result": "IFRS Standards",
                "competitor_result": "Competitor Reports",
                "web_result": "Web Search",
            }[key]
            results.append(f"**[{label}]**\n{r['answer']}")
            all_visual.extend(r.get("visual_content", []))
            all_sources.extend(r.get("sources", []))

    if not results:
        combined = "I could not find relevant information to answer your question from any available source."
        follow_up = "Could you rephrase your question or provide more context?"
    else:
        combined_text = "\n\n---\n\n".join(results)
        prompt = f"""You are synthesizing answers from multiple specialized agents for a financial knowledge assistant.

Agent answers:
{combined_text}

Original question: {state["query"]}

Create a coherent, unified final answer that:
1. Integrates insights from all sources
2. Avoids repetition
3. Clearly attributes information to its source
4. Is professional and precise

Then on a new line starting with "FOLLOW_UP:", write one relevant follow-up question that would help the user explore this topic deeper.

Format:
<your synthesized answer>

FOLLOW_UP: <your follow-up question>
"""
        chat = _get_client().chats.create(model=LLM_MODEL)
        response = chat.send_message(prompt)
        raw = response.text

        if "FOLLOW_UP:" in raw:
            parts = raw.split("FOLLOW_UP:", 1)
            combined = parts[0].strip()
            follow_up = parts[1].strip()
        else:
            combined = raw.strip()
            follow_up = "Would you like to explore any specific aspect of this topic further?"

    return {
        "final_answer": combined,
        "follow_up_question": follow_up,
        "all_visual_content": all_visual,
        "all_sources": all_sources,
        "conversation_history": [
            {"role": "user", "content": state["query"]},
            {"role": "assistant", "content": combined},
        ],
    }


def _route_after_ambiguity(state: AgentState) -> str:
    if state.get("needs_clarification"):
        return "clarification_needed"
    return "classify_sources"


def _route_after_classify(state: AgentState) -> list[str]:
    sources = state.get("detected_sources", [])
    if not sources:
        return ["request_web_approval"]
    nodes = []
    if "esg" in sources:
        nodes.append("run_esg")
    if "ifrs" in sources:
        nodes.append("run_ifrs")
    if "competitors" in sources:
        nodes.append("run_competitor")
    return nodes if nodes else ["request_web_approval"]


def _route_after_web_approval(state: AgentState) -> str:
    if state.get("web_approved"):
        return "run_web"
    return "synthesize"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("check_ambiguity", node_check_ambiguity)
    builder.add_node("classify_sources", node_classify_sources)
    builder.add_node("run_esg", node_run_esg)
    builder.add_node("run_ifrs", node_run_ifrs)
    builder.add_node("run_competitor", node_run_competitor)
    builder.add_node("request_web_approval", node_request_web_approval)
    builder.add_node("run_web", node_run_web)
    builder.add_node("synthesize", node_synthesize)

    builder.add_edge(START, "check_ambiguity")

    builder.add_conditional_edges(
        "check_ambiguity",
        _route_after_ambiguity,
        {
            "clarification_needed": END,
            "classify_sources": "classify_sources",
        },
    )

    builder.add_conditional_edges(
        "classify_sources",
        _route_after_classify,
        {
            "run_esg": "run_esg",
            "run_ifrs": "run_ifrs",
            "run_competitor": "run_competitor",
            "request_web_approval": "request_web_approval",
        },
    )

    builder.add_conditional_edges(
        "request_web_approval",
        _route_after_web_approval,
        {
            "run_web": "run_web",
            "synthesize": "synthesize",
        },
    )

    builder.add_edge("run_esg", "synthesize")
    builder.add_edge("run_ifrs", "synthesize")
    builder.add_edge("run_competitor", "synthesize")
    builder.add_edge("run_web", "synthesize")
    builder.add_edge("synthesize", END)

    return builder


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        builder = build_graph()
        _compiled_graph = builder.compile(interrupt_before=["request_web_approval"])
    return _compiled_graph


def run_query(
    query: str,
    conversation_history: list[dict] | None = None,
    web_approved: bool = False,
    thread_id: str = "default",
) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "detected_sources": [],
        "needs_clarification": False,
        "clarification_question": "",
        "esg_result": None,
        "ifrs_result": None,
        "competitor_result": None,
        "web_result": None,
        "web_approved": web_approved,
        "final_answer": "",
        "follow_up_question": "",
        "all_visual_content": [],
        "all_sources": [],
    }

    result = graph.invoke(initial_state, config=config)
    return result
