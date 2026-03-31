"""Production RAG — Streamlit frontend."""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_BASE = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Production RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults = {
        "messages": [],
        "tenant_id": "default",
        "enable_crag": True,
        "enable_self_rag": True,
        "diagnostics": [],
        "citations": [],
        "retrieval_details": [],
        "jwt_token": os.environ.get("RAG_JWT_TOKEN", ""),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


def _auth_headers() -> dict[str, str]:
    tok = st.session_state.jwt_token
    return {"Authorization": f"Bearer {tok}"} if tok else {}


with st.sidebar:
    st.header("⚙️ Configuration")

    st.session_state.tenant_id = st.text_input(
        "Tenant ID",
        value=st.session_state.tenant_id,
        help="Weaviate tenant (research organisation / corpus partition)",
    )

    st.session_state.jwt_token = st.text_input(
        "JWT Token",
        value=st.session_state.jwt_token,
        type="password",
        help="Bearer token for API authentication",
    )

    st.divider()
    st.subheader("Adaptive RAG")
    st.session_state.enable_crag = st.toggle(
        "CRAG (Corrective Retrieval)",
        value=st.session_state.enable_crag,
        help="Re-retrieves if initial quality is below threshold",
    )
    st.session_state.enable_self_rag = st.toggle(
        "Self-RAG (Critique & Refine)",
        value=st.session_state.enable_self_rag,
        help="Critiques and regenerates if answer has unsupported claims",
    )

    st.divider()
    st.subheader("� Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")
    pdf_strategy = st.selectbox(
        "Chunking strategy (PDF)",
        ["recursive", "fixed", "semantic", "hierarchical", "late", "docling"],
        index=0,
        key="pdf_strategy",
    )
    if st.button("Upload & Ingest PDF", use_container_width=True, disabled=uploaded_pdf is None):
        with st.spinner("Saving and ingesting PDF..."):
            try:
                import os
                uploads_dir = "/uploads"
                os.makedirs(uploads_dir, exist_ok=True)
                safe_name = uploaded_pdf.name.replace(" ", "_")
                upload_path = os.path.join(uploads_dir, safe_name)
                with open(upload_path, "wb") as f:
                    f.write(uploaded_pdf.getvalue())
                headers = _auth_headers()
                r = httpx.post(
                    f"{API_BASE}/ingest",
                    json={
                        "pdf_paths": [upload_path],
                        "tenant_id": st.session_state.tenant_id,
                        "chunking_strategy": pdf_strategy,
                        "enable_raptor": False,
                    },
                    headers=headers,
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                st.success(f"Job queued: `{data.get('job_id', 'unknown')}`")
            except Exception as exc:
                st.error(f"Ingest failed: {exc}")

    st.divider()
    st.subheader("📥 Ingest ArXiv Papers")
    arxiv_ids_input = st.text_area(
        "ArXiv IDs (one per line)",
        placeholder="2312.10997\n2310.01558",
        height=100,
    )
    ingest_strategy = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed", "semantic", "hierarchical", "late"],
        index=0,
    )
    if st.button("Ingest ArXiv", use_container_width=True):
        ids = [x.strip() for x in arxiv_ids_input.splitlines() if x.strip()]
        if ids:
            with st.spinner(f"Ingesting {len(ids)} paper(s)..."):
                try:
                    headers = _auth_headers()
                    r = httpx.post(
                        f"{API_BASE}/ingest",
                        json={
                            "arxiv_ids": ids,
                            "tenant_id": st.session_state.tenant_id,
                            "chunking_strategy": ingest_strategy,
                            "enable_raptor": True,
                        },
                        headers=headers,
                        timeout=30,
                    )
                    r.raise_for_status()
                    data = r.json()
                    st.success(f"Job submitted: `{data.get('job_id', 'unknown')}`")
                except Exception as exc:
                    st.error(f"Ingest failed: {exc}")
        else:
            st.warning("Enter at least one ArXiv ID")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.diagnostics = []
        st.session_state.citations = []
        st.session_state.retrieval_details = []
        st.rerun()


def _highlight_snippet(text: str, query: str, max_len: int = 300) -> str:
    """Return a truncated snippet of text (no true highlighting in Streamlit)."""
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _stream_sse(query: str) -> str:
    """Call the /chat endpoint and render SSE events in real-time. Returns the full answer."""
    url = f"{API_BASE}/chat"
    body = {
        "query": query,
        "tenant_id": st.session_state.tenant_id,
        "enable_crag": st.session_state.enable_crag,
        "enable_self_rag": st.session_state.enable_self_rag,
    }
    headers = {**_auth_headers(), "Accept": "text/event-stream"}

    answer_placeholder = st.empty()
    answer_tokens: list[str] = []
    citations: list[dict] = []
    retrieval_details: list[dict] = []

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", url, json=body, headers=headers) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if not raw or raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "context_chunk":
                        retrieval_details.append(event.get("data", {}))

                    elif etype == "token":
                        answer_tokens.append(event.get("data", {}).get("token", ""))
                        answer_placeholder.markdown("".join(answer_tokens) + "▌")

                    elif etype == "citation":
                        citations.append(event.get("data", {}))

                    elif etype == "diagnostics":
                        diag = event.get("data", {})
                        st.session_state.diagnostics = [diag]

                    elif etype == "error":
                        st.error(f"Pipeline error: {event.get('data', {}).get('message')}")
                        return

        final_answer = "".join(answer_tokens)
        answer_placeholder.markdown(final_answer)
        st.session_state.citations = citations
        st.session_state.retrieval_details = retrieval_details
        return final_answer

    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.read().decode()
        except Exception:
            detail = str(exc)
        st.error(f"API error {exc.response.status_code}: {detail}")
        return ""
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return ""


st.title("🔬 Production RAG — Scientific Paper Q&A")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask a question about the ingested papers…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        final_answer = _stream_sse(user_input)

    if final_answer:
        st.session_state.messages.append({"role": "assistant", "content": final_answer})


if st.session_state.retrieval_details or st.session_state.citations or st.session_state.diagnostics:
    with st.expander("📊 Retrieval diagnostics", expanded=False):
        col_left, col_right = st.columns([3, 2])

        with col_left:
            if st.session_state.retrieval_details:
                st.subheader("Retrieved context chunks")
                for i, chunk in enumerate(st.session_state.retrieval_details[:10], 1):
                    score = chunk.get("score")
                    score_str = f"Score={score:.4f}" if score is not None else ""
                    with st.container(border=True):
                        chunk_type = chunk.get("chunk_type", "text")
                        type_badge = " 📊 TABLE" if chunk_type == "table" else (" 🖼️ FIGURE" if chunk_type == "figure" else "")
                        st.caption(f"**Chunk {i}**{type_badge} — {score_str}")
                        st.text(chunk.get("snippet", "")[:250])
                        source_parts = []
                        if chunk.get("title"):
                            source_parts.append(chunk["title"])
                        if chunk.get("arxiv_id"):
                            source_parts.append(f"arXiv:{chunk['arxiv_id']}")
                        if chunk.get("page"):
                            source_parts.append(f"p.{chunk['page']}")
                        if source_parts:
                            st.caption("Source: " + " | ".join(source_parts))

        with col_right:
            if st.session_state.diagnostics:
                st.subheader("Pipeline stats")
                diag = st.session_state.diagnostics[0]
                st.metric("CRAG hops", diag.get("adaptive_hops", 0))
                st.metric(
                    "Retrieval quality score",
                    f"{diag.get('quality_score', 0.0):.3f}",
                )
                st.metric("Chunks retrieved", diag.get("candidate_count", 0))
                st.metric("Chunks after rerank", diag.get("reranked_count", 0))

    if st.session_state.citations:
        with st.expander("📎 Citations", expanded=False):
            for cit in st.session_state.citations:
                with st.container(border=True):
                    st.markdown(f"**[{cit.get('citation_id', '')}]**")
                    st.caption(
                        f"Title: {cit.get('title', 'N/A')} | "
                        f"ArXiv: {cit.get('arxiv_id', 'N/A')} | "
                        f"Page: {cit.get('page', 'N/A')}"
                    )
                    snippet = cit.get("snippet", "")
                    if snippet:
                        st.text(_highlight_snippet(snippet, ""))
