"""
Streamlit Chat UI for the RAG Pipeline.
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path

from core.parser import parse_pdf
from core.chunker import chunk_markdown
from core.vectorstore import VectorStore
from core.rag_chain import RAGChain
from config.settings import GEMINI_API_KEY


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper RAG",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1f 100%);
    }

    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #c4b5fd;
        margin-bottom: 2px;
    }

    .sidebar-sub {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.82rem;
        margin-top: 0;
        margin-bottom: 16px;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(76, 175, 80, 0.2);
        color: #81c784;
        border: 1px solid rgba(76, 175, 80, 0.3);
    }
    .status-empty {
        background: rgba(255, 152, 0, 0.2);
        color: #ffb74d;
        border: 1px solid rgba(255, 152, 0, 0.3);
    }

    .stack-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .stack-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stack-value {
        color: #e0e0ff;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    /* Hide sidebar toggle button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── API Key Verification ───────────────────────────────────────────────────
if not GEMINI_API_KEY:
    st.error("""
    ### 🔑 GEMINI_API_KEY is not configured!
    
    To use this app:
    - **On Streamlit Cloud:**
      1. Click **Manage app** in the bottom-right corner.
      2. Click the three dots `⋮` ➜ **Settings** ➜ **Secrets**.
      3. Add your Gemini API key:
         ```toml
         GEMINI_API_KEY = "your_actual_gemini_api_key_here"
         ```
      4. Save and click **Reboot app**.
    - **Locally:** Add `GEMINI_API_KEY=your_key` to your `.env` file.
    """)
    st.stop()

# ─── Session State Init ──────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = VectorStore()

if "chain" not in st.session_state:
    st.session_state.chain = RAGChain(st.session_state.store)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested" not in st.session_state:
    st.session_state.ingested = st.session_state.store.count > 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Paper RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Scientific Document Q&A</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Status
    doc_count = st.session_state.store.count
    if doc_count > 0:
        st.markdown(
            f'<span class="status-badge status-ready">Ready — {doc_count} chunks</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-empty">No documents ingested</span>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # PDF Ingestion
    st.subheader("Ingest PDF")

    project_root = Path(__file__).resolve().parent

    # Upload a new PDF
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        save_path = project_root / uploaded_file.name
        if not save_path.exists():
            save_path.write_bytes(uploaded_file.getvalue())
            st.success(f"Saved: {uploaded_file.name}")
        # Set this as the active PDF
        if "active_pdf" not in st.session_state or st.session_state.active_pdf != str(save_path):
            st.session_state.active_pdf = str(save_path)
            st.rerun()

    st.markdown("")

    # List all PDFs in the project directory (including newly uploaded ones)
    pdf_files = sorted(project_root.glob("*.pdf"))
    pdf_names = [p.name for p in pdf_files]

    if pdf_names:
        # If an active PDF was just uploaded, pre-select it
        default_idx = 0
        if "active_pdf" in st.session_state:
            active_name = Path(st.session_state.active_pdf).name
            if active_name in pdf_names:
                default_idx = pdf_names.index(active_name)

        selected_pdf = st.selectbox("Select PDF", pdf_names, index=default_idx, label_visibility="collapsed")
        pdf_path = str(project_root / selected_pdf)
        st.session_state.active_pdf = pdf_path

        # Auto-ingest if not already in the vector store
        if not st.session_state.store.is_ingested(pdf_path):
            st.info("Auto-processing document...")

            with st.spinner("Parsing PDF (uses cache if available)..."):
                try:
                    markdown = parse_pdf(pdf_path)
                except Exception as e:
                    st.error(f"Parse error: {e}")
                    st.stop()

            with st.spinner("Chunking text..."):
                chunks = chunk_markdown(markdown)

            with st.spinner("Embedding & storing in database..."):
                n = st.session_state.store.ingest(chunks, pdf_path)
                st.session_state.ingested = True
                st.session_state.chain = RAGChain(st.session_state.store)

            st.success(f"Ready! Created {n} chunks.")
            import time
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.ingested = True
    else:
        st.info("Upload a PDF to get started.")

    st.markdown("---")

    selected_model = st.selectbox(
        "Gemini Model",
        options=["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-3.1-pro-preview"],
        index=0,
        help="Select Gemini model. Note that Pro preview models may require a billing-linked API key."
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chain.clear_history()
        st.rerun()

    if st.button("Clear Vector Store", use_container_width=True):
        st.session_state.store.clear()
        st.session_state.ingested = False
        st.session_state.messages = []
        st.session_state.chain = RAGChain(st.session_state.store)
        st.rerun()

    st.markdown("---")

    st.markdown(f"""
    <div class="stack-card">
        <div class="stack-label">Cloud Stack</div>
        <div class="stack-value">
            {selected_model} (LLM)<br>
            gemini-embedding-2 &middot; ChromaDB
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Chat Area ──────────────────────────────────────────────────────────

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"Retrieved {len(msg['chunks'])} chunks"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    page = chunk.get("metadata", {}).get("page", "?")
                    heading = chunk.get("metadata", {}).get("heading", "")
                    sim = 1.0 - chunk.get("distance", 0)
                    st.markdown(f"**Chunk {i}** — Page {page} | {heading} | Sim: {sim:.2f}")
                    st.code(chunk.get("text", "")[:500], language=None)

# Chat input
if prompt := st.chat_input("Ask about the paper..."):
    if not st.session_state.ingested:
        st.warning("Please ingest a PDF first using the sidebar.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        from core.retriever import retrieve as do_retrieve
        _, raw_hits = do_retrieve(st.session_state.store, prompt)

        response_placeholder = st.empty()
        full_response = ""

        try:
            for token in st.session_state.chain.ask_stream(prompt, model=selected_model):
                full_response += token
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
        except Exception as e:
            st.error(f"❌ Error during generation: {e}")
            full_response = f"*(Error: {e})*"
            response_placeholder.markdown(full_response)

        if raw_hits:
            with st.expander(f"Retrieved {len(raw_hits)} chunks"):
                for i, chunk in enumerate(raw_hits, 1):
                    page = chunk.get("metadata", {}).get("page", "?")
                    heading = chunk.get("metadata", {}).get("heading", "")
                    sim = 1.0 - chunk.get("distance", 0)
                    st.markdown(f"**Chunk {i}** — Page {page} | {heading} | Sim: {sim:.2f}")
                    st.code(chunk.get("text", "")[:500], language=None)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "chunks": raw_hits,
    })
