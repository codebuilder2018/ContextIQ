"""Streamlit UI for the ContextIQ Agentic RAG System.

This module provides the presentation layer for the RAG application.
It initializes the RAG pipeline once, accepts natural-language questions,
executes the LangGraph workflow, and presents answers, source documents,
response metrics, and recent search history.

The underlying RAG functionality is intentionally kept unchanged.
UI improvements are implemented through styling and presentation only.
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Add the project root to Python's import path so the src package
# can be imported when this file is launched directly by Streamlit.
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ContextIQ | Agentic RAG",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# UI STYLING
# ============================================================================

st.markdown(
    """
    <style>
        /* ------------------------------------------------------------------
           Global layout
           ------------------------------------------------------------------ */
        .block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        /* ------------------------------------------------------------------
           Header
           ------------------------------------------------------------------ */
        .app-header {
            padding: 1.4rem 1.5rem 1.2rem 1.5rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 18px;
            margin-bottom: 1.25rem;
            background: linear-gradient(
                135deg,
                rgba(76, 175, 80, 0.08),
                rgba(33, 150, 243, 0.05)
            );
        }

        .app-title {
            font-size: 2rem;
            font-weight: 750;
            letter-spacing: -0.03em;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #6b7280;
            font-size: 0.98rem;
            margin-bottom: 0;
        }

        /* ------------------------------------------------------------------
           Status
           ------------------------------------------------------------------ */
        .status-card {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            border: 1px solid rgba(76, 175, 80, 0.22);
            background: rgba(76, 175, 80, 0.07);
            margin-bottom: 1rem;
        }

        .status-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #22c55e;
            display: inline-block;
        }

        .status-text {
            font-size: 0.9rem;
            font-weight: 600;
        }

        /* ------------------------------------------------------------------
           Search form
           ------------------------------------------------------------------ */
        div[data-testid="stForm"] {
            padding: 1.1rem;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            background: rgba(128, 128, 128, 0.025);
        }

        .stTextInput > div > div > input {
            border-radius: 10px;
        }

        .stButton > button,
        button[kind="primaryFormSubmit"] {
            width: 100%;
            min-height: 2.65rem;
            border-radius: 10px;
            font-weight: 650;
            border: 0;
            transition: all 0.18s ease;
        }

        .stButton > button:hover,
        button[kind="primaryFormSubmit"]:hover {
            transform: translateY(-1px);
        }

        /* ------------------------------------------------------------------
           Answer card
           ------------------------------------------------------------------ */
        .answer-heading {
            font-size: 1.15rem;
            font-weight: 700;
            margin: 1.35rem 0 0.65rem 0;
        }

        .answer-meta {
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }

        /* ------------------------------------------------------------------
           History cards
           ------------------------------------------------------------------ */
        .history-card {
            border: 1px solid rgba(128, 128, 128, 0.16);
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.7rem;
            background: rgba(128, 128, 128, 0.025);
        }

        .history-question {
            font-weight: 650;
            margin-bottom: 0.3rem;
        }

        .history-answer {
            color: #6b7280;
            font-size: 0.9rem;
        }

        .history-meta {
            color: #9ca3af;
            font-size: 0.76rem;
            margin-top: 0.4rem;
        }

        /* ------------------------------------------------------------------
           Section spacing
           ------------------------------------------------------------------ */
        .section-divider {
            margin: 1.4rem 0 1rem 0;
            border-top: 1px solid rgba(128, 128, 128, 0.15);
        }

        /* ------------------------------------------------------------------
           Mobile responsiveness
           ------------------------------------------------------------------ */
        @media (max-width: 640px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .app-title {
                font-size: 1.65rem;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state() -> None:
    """Initialize Streamlit session-state variables.

    The RAG system is stored in session state so the application can reuse
    the initialized graph during the current browser session. Search
    history is also maintained here for the existing recent-search view.
    """
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if "history" not in st.session_state:
        st.session_state.history = []


# ============================================================================
# RAG INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_rag():
    """Initialize and cache the complete RAG system.

    The initialization pipeline creates:
        1. The configured LLM.
        2. The document processor.
        3. The vector store.
        4. The document embeddings and FAISS index.
        5. The LangGraph workflow.

    Returns:
        A tuple containing the GraphBuilder instance and the number of
        processed document chunks.

    Returns:
        tuple: ``(graph_builder, num_chunks)`` on success, or
            ``(None, 0)`` when initialization fails.
    """
    try:
        # Initialize the configured LLM.
        llm = Config.get_llm()

        # Initialize document processing using application configuration.
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        # Initialize the vector store.
        vector_store = VectorStore()

        # Use the document sources configured by the application.
        urls = Config.DEFAULT_URLS

        # Load and split source documents into RAG-ready chunks.
        documents = doc_processor.process_urls(urls)

        # Create the FAISS vector index from processed documents.
        vector_store.create_vectorstore(documents)

        # Build the LangGraph RAG workflow.
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm,
        )
        graph_builder.build()

        return graph_builder, len(documents)

    except Exception as exc:
        st.error(f"Failed to initialize: {exc}")
        return None, 0


# ============================================================================
# PRESENTATION HELPERS
# ============================================================================

def render_header() -> None:
    """Render the application header."""
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🔍 ContextIQ</div>
            <p class="app-subtitle">
                Agentic RAG Knowledge Assistant — ask questions across
                your indexed knowledge with grounded AI responses.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(num_chunks: int) -> None:
    """Render the current RAG system status.

    Args:
        num_chunks: Number of document chunks indexed by the RAG system.
    """
    st.markdown(
        f"""
        <div class="status-card">
            <span class="status-dot"></span>
            <span class="status-text">
                System ready · {num_chunks:,} document chunks indexed
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(documents) -> None:
    """Render retrieved source documents in an expandable section.

    Args:
        documents: Retrieved LangChain Document objects returned by the
            RAG workflow.
    """
    with st.expander("📄 Source Documents", expanded=False):
        if not documents:
            st.caption("No source documents were returned.")
            return

        for index, doc in enumerate(documents, start=1):
            content = getattr(doc, "page_content", "") or ""
            preview = content[:500]
            if len(content) > 500:
                preview += "..."

            metadata = getattr(doc, "metadata", {}) or {}
            source = (
                metadata.get("title")
                or metadata.get("source")
                or f"Document {index}"
            )

            st.markdown(f"**{index}. {source}**")
            st.text_area(
                f"Document {index}",
                preview,
                height=120,
                disabled=True,
                label_visibility="collapsed",
                key=f"source_document_{index}",
            )


def render_history() -> None:
    """Render the existing recent-search history.

    The original behavior of showing only the last three searches is
    preserved.
    """
    if not st.session_state.history:
        return

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📜 Recent Searches")

    for index, item in enumerate(
        reversed(st.session_state.history[-3:]),
        start=1,
    ):
        question = item["question"]
        answer = item["answer"]
        elapsed = item["time"]

        if len(answer) > 220:
            answer = answer[:220] + "..."

        st.markdown(
            f"""
            <div class="history-card">
                <div class="history-question">Q: {question}</div>
                <div class="history-answer">A: {answer}</div>
                <div class="history-meta">
                    Response time: {elapsed:.2f}s
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Run the Streamlit application."""
    init_session_state()

    # Application header.
    render_header()

    # Initialize the RAG system once per cached resource/session lifecycle.
    if not st.session_state.initialized:
        with st.spinner("Initializing ContextIQ..."):
            rag_system, num_chunks = initialize_rag()

            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                render_status(num_chunks)
    else:
        # Preserve the same ready-state behavior when Streamlit reruns.
        st.markdown(
            """
            <div class="status-card">
                <span class="status-dot"></span>
                <span class="status-text">System ready</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Search interface.
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?",
        )
        submit = st.form_submit_button(
            "🔍 Search",
            type="primary",
        )

    # Execute the existing RAG workflow when a question is submitted.
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching indexed knowledge..."):
                start_time = time.time()

                # Execute the LangGraph RAG workflow.
                result = st.session_state.rag_system.run(question)

                elapsed_time = time.time() - start_time

                # Preserve the existing search-history behavior.
                st.session_state.history.append(
                    {
                        "question": question,
                        "answer": result["answer"],
                        "time": elapsed_time,
                    }
                )

                # Render answer.
                st.markdown(
                    '<div class="answer-heading">💡 Answer</div>',
                    unsafe_allow_html=True,
                )
                st.success(result["answer"])

                # Render retrieved source documents.
                render_sources(result["retrieved_docs"])

                # Render response-time information.
                st.markdown(
                    f'<div class="answer-meta">⏱ Response time: '
                    f'{elapsed_time:.2f} seconds</div>',
                    unsafe_allow_html=True,
                )

        elif not st.session_state.initialized:
            st.warning("The RAG system is not initialized yet. Please retry.")

    # Render recent search history.
    render_history()


if __name__ == "__main__":
    main()