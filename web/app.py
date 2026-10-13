import base64
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.orchestrator import run_query

st.set_page_config(
    page_title="Financial Knowledge Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.source-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75em;
    font-weight: 600;
    margin: 2px;
}
.badge-esg { background-color: #1a7f37; color: white; }
.badge-ifrs { background-color: #0550ae; color: white; }
.badge-competitors { background-color: #9a3412; color: white; }
.badge-web { background-color: #6e40c9; color: white; }
.follow-up-box {
    background-color: #f0f7ff;
    border-left: 4px solid #0550ae;
    padding: 10px 16px;
    border-radius: 4px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "pending_web_query" not in st.session_state:
        st.session_state.pending_web_query = None
    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = False
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def source_badge(source_type: str) -> str:
    labels = {
        "esg": ("ESG", "esg"),
        "ifrs": ("IFRS", "ifrs"),
        "competitors": ("Competitors", "competitors"),
        "web": ("Web", "web"),
    }
    label, cls = labels.get(source_type, (source_type.upper(), "esg"))
    return f'<span class="source-badge badge-{cls}">{label}</span>'


def render_message(msg: dict):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            result = msg.get("result")
            if result:
                if result.get("needs_clarification"):
                    st.warning(f"Could you clarify: **{result['clarification_question']}**")
                    return

                if result.get("web_approval_needed"):
                    st.info(result["message"])
                    return

                detected = result.get("detected_sources", [])
                if detected:
                    badges = " ".join(source_badge(s) for s in detected)
                    st.markdown(f"**Sources used:** {badges}", unsafe_allow_html=True)

                st.markdown(result.get("final_answer", ""))

                visual = result.get("all_visual_content", [])
                if visual:
                    with st.expander(f"Visual content ({len(visual)} items)", expanded=False):
                        for item in visual:
                            if item["type"] == "table":
                                st.markdown(f"**Table from {item['file_name']}, page {item['page']}**")
                                st.markdown(item["content"])
                                st.divider()
                            elif item["type"] == "image" and item.get("image_data"):
                                st.markdown(f"**Image from {item['file_name']}, page {item['page']}**")
                                img_bytes = base64.b64decode(item["image_data"])
                                st.image(img_bytes, caption=item.get("caption", ""))
                                st.divider()

                if result.get("follow_up_question"):
                    st.markdown(
                        f'<div class="follow-up-box">💡 <b>Follow-up:</b> {result["follow_up_question"]}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])


def process_query(query: str, web_approved: bool = False):
    with st.spinner("Thinking..."):
        result = run_query(
            query=query,
            conversation_history=st.session_state.conversation_history,
            web_approved=web_approved,
            thread_id=st.session_state.thread_id,
        )

    if result.get("needs_clarification"):
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.get("clarification_question", ""),
            "result": result,
        })
        st.session_state.last_result = None
        st.session_state.pending_web_query = None
        return

    if not result.get("final_answer") and result.get("detected_sources") == []:
        st.session_state.pending_web_query = query
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I couldn't find an answer in internal documents. Do you want me to search the internet?",
            "result": {
                "web_approval_needed": True,
                "message": "I couldn't find an answer in internal documents. Do you want me to search the internet?",
            },
        })
        return

    st.session_state.conversation_history.extend(
        result.get("conversation_history", [])
    )
    st.session_state.last_result = result
    st.session_state.pending_web_query = None
    st.session_state.pending_feedback = True

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("final_answer", ""),
        "result": result,
    })


def main():
    init_session()

    with st.sidebar:
        st.title("📊 Financial Knowledge Assistant")
        st.markdown("---")
        st.markdown("**Data Sources**")
        st.markdown("- ESG / GHG Protocol documents")
        st.markdown("- IFRS-18 Standards")
        st.markdown("- Pfizer Annual Report")
        st.markdown("---")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.last_result = None
            st.session_state.pending_web_query = None
            st.session_state.pending_feedback = False
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()

    st.title("Financial Knowledge Assistant")
    st.caption("Multi-Agentic RAG system for ESG, IFRS, and Competitor Intelligence")

    for msg in st.session_state.messages:
        render_message(msg)

    if st.session_state.pending_web_query:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, search the internet", type="primary", use_container_width=True):
                query = st.session_state.pending_web_query
                st.session_state.pending_web_query = None
                process_query(query, web_approved=True)
                st.rerun()
        with col2:
            if st.button("No, skip web search", use_container_width=True):
                st.session_state.pending_web_query = None
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Understood. No web search performed.",
                    "result": None,
                })
                st.rerun()

    if st.session_state.pending_feedback and st.session_state.last_result:
        st.markdown("---")
        st.markdown("**Was this answer helpful?**")
        col1, col2, col3 = st.columns([1, 1, 6])
        with col1:
            if st.button("👍 Yes", use_container_width=True):
                st.session_state.pending_feedback = False
                st.success("Thank you for your feedback!")
                st.rerun()
        with col2:
            if st.button("👎 No", use_container_width=True):
                st.session_state.pending_feedback = False
                st.info("Thank you. You can ask a follow-up question or try rephrasing.")
                st.rerun()

    if prompt := st.chat_input("Ask a question about ESG, IFRS, or competitor insights..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
