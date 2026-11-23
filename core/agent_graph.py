"""
LangGraph State Machine for Agentic Chemistry RAG.
Replaces linear chains with an agentic workflow featuring:
- Router Node: Dispatches between ChromaDB (paper data) and PubChem (general chemistry).
- Retriever Node: Performs Parent-Document retrieval.
- Critique/Grader Node: Evaluates chunk relevance.
- Query Rewriter Node: Re-retrieval loop if initial context is irrelevant.
- Generator Node: Produces chemical responses with LaTeX formatting via llama3.1:8b.
"""

import json
import re
from typing import TypedDict, Optional, Generator, Any
from langgraph.graph import StateGraph, END

from config.settings import (
    TOP_K,
    SIMILARITY_THRESHOLD,
    MAX_REWRITE_RETRIES,
    CHEM_SYSTEM_PROMPT,
    LLM_MODEL,
)
from core.vectorstore import VectorStore
from core.pubchem_tool import query_chemistry_knowledge
from core.llm import query_llm


class AgentState(TypedDict):
    """LangGraph state representation for the Chemistry RAG Agent."""
    user_query: str
    chat_history: list[dict]
    route: str  # "vectorstore" or "web_search"
    current_query: str
    retrieval_count: int
    retrieved_parents: list[dict]
    doc_context: str
    is_relevant: bool
    final_answer: str


class ChemAgentGraph:
    """Wrapper class managing the LangGraph state machine and execution."""

    def __init__(self, store: VectorStore):
        self.store = store
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs and compiles the LangGraph StateGraph."""
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("router", self._router_node)
        builder.add_node("retriever", self._retriever_node)
        builder.add_node("pubchem", self._pubchem_node)
        builder.add_node("grader", self._grader_node)
        builder.add_node("rewriter", self._rewriter_node)
        builder.add_node("generator", self._generator_node)

        # Set entry point
        builder.set_entry_point("router")

        # Routing conditional edge
        builder.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "vectorstore": "retriever",
                "web_search": "pubchem",
            },
        )

        # PubChem leads directly to generation
        builder.add_edge("pubchem", "generator")

        # Retriever leads to grader critique
        builder.add_edge("retriever", "grader")

        # Grader conditional edge (Critique/Loop)
        builder.add_conditional_edges(
            "grader",
            self._grade_decision,
            {
                "generator": "generator",
                "rewriter": "rewriter",
            },
        )

        # Rewriter loops back to retriever
        builder.add_edge("rewriter", "retriever")

        # Generator leads to END
        builder.add_edge("generator", END)

        return builder.compile()

    # ─── Node Implementations ────────────────────────────────────────────────

    def _router_node(self, state: AgentState) -> dict:
        """
        The Router Node:
        Decides whether to route the query to ChromaDB (paper-specific)
        or PubChem/Web Search (general chemistry knowledge).
        """
        user_query = state["user_query"]
        print(f"[Agent] Router analyzing query: '{user_query}'", flush=True)

        router_prompt = (
            "You are a specialized router for a chemistry research assistant.\n"
            "Analyze the user question and classify it into ONE category:\n"
            "- 'vectorstore': The question is about a specific scientific paper, experimental results, "
            "synthesized molecules described in the document, figures, tables, yields, or author findings.\n"
            "- 'web_search': The question asks for general chemical properties, standard PubChem compound lookup, "
            "CAS registry numbers, SMILES of common chemicals, boiling points, molecular formulas, or general textbook chemistry.\n\n"
            f"User Question: {user_query}\n\n"
            "Respond ONLY with a JSON object: {\"route\": \"vectorstore\"} or {\"route\": \"web_search\"}"
        )

        try:
            resp = query_llm(router_prompt, system_prompt="You are an accurate classifier that outputs strict JSON.")
            # Extract json
            match = re.search(r'\{.*"route":\s*"([^"]+)".*\}', resp, re.DOTALL)
            if match:
                selected_route = match.group(1).strip()
                if selected_route in ("vectorstore", "web_search"):
                    print(f"[Agent] Route decided by LLM: {selected_route}", flush=True)
                    return {
                        "route": selected_route,
                        "current_query": user_query,
                        "retrieval_count": 0,
                    }
        except Exception as e:
            print(f"[Agent] Router LLM failed: {e}. Using heuristic.", flush=True)

        # Heuristic fallback: if query mentions paper/figure/table/experiment/yield -> vectorstore
        paper_signals = ["paper", "figure", "table", "yield", "author", "experiment", "reaction", "result", "sample", "study"]
        if any(w in user_query.lower() for w in paper_signals) or self.store.count > 0:
            route = "vectorstore"
        else:
            route = "web_search"

        print(f"[Agent] Route decided: {route}", flush=True)
        return {
            "route": route,
            "current_query": user_query,
            "retrieval_count": 0,
        }

    def _retriever_node(self, state: AgentState) -> dict:
        """
        The Retriever Node:
        Executes Parent-Document retrieval against ChromaDB using current_query.
        """
        query = state["current_query"]
        count = state.get("retrieval_count", 0) + 1
        print(f"[Agent] Retrieval #{count} executing with query: '{query}'", flush=True)

        parents = self.store.search_parent_docs(
            query=query,
            top_k=3,
            threshold=SIMILARITY_THRESHOLD,
        )

        if not parents:
            context = "No relevant document sections found."
        else:
            formatted = []
            for i, p in enumerate(parents[:3], 1):
                header = f"### [Document Section {i} | Page {p.get('page', '?')}"
                if p.get("heading"):
                    header += f" | {p['heading']}"
                header += f" | Cosine Sim: {p.get('similarity', 0.0):.2f}]"
                # Cap parent text if excessively long
                sec_text = p['text'][:2500]
                formatted.append(f"{header}\n{sec_text}")
            context = "\n\n".join(formatted)

        print(f"[Agent] Retrieved {len(parents)} parent sections.", flush=True)
        return {
            "retrieved_parents": parents,
            "doc_context": context,
            "retrieval_count": count,
        }

    def _pubchem_node(self, state: AgentState) -> dict:
        """
        The PubChem Node:
        Queries PubChem PUG REST API for verified chemical property data.
        """
        query = state["current_query"]
        print(f"[Agent] Querying PubChem database for: '{query}'", flush=True)
        context = query_chemistry_knowledge(query)
        return {
            "doc_context": context,
            "is_relevant": True,
            "retrieved_parents": [],
        }

    def _grader_node(self, state: AgentState) -> dict:
        """
        The Critique / Loop Node:
        Grades whether the retrieved parent chunks are relevant to answering the query.
        """
        parents = state.get("retrieved_parents", [])
        if not parents:
            print("[Agent] Grader: No chunks retrieved -> Marked irrelevant.", flush=True)
            return {"is_relevant": False}

        # Check similarity of top parent
        top_sim = parents[0].get("similarity", 0.0)
        if top_sim < SIMILARITY_THRESHOLD:
            print(f"[Agent] Grader: Top similarity {top_sim:.2f} below threshold -> Marked irrelevant.", flush=True)
            return {"is_relevant": False}

        user_query = state["user_query"]
        sample_context = state["doc_context"][:1500]

        grader_prompt = (
            "You are a scientific grader assessing retrieval relevance for a chemistry RAG system.\n"
            f"User Question: {user_query}\n\n"
            f"Retrieved Excerpt:\n{sample_context}\n\n"
            "Does the retrieved excerpt contain information that helps answer the user's question? "
            "Answer with ONLY 'YES' or 'NO'."
        )

        try:
            resp = query_llm(grader_prompt, system_prompt="Answer only YES or NO.")
            is_rel = "YES" in resp.upper()
            print(f"[Agent] Grader assessment: {'RELEVANT (YES)' if is_rel else 'IRRELEVANT (NO)'}", flush=True)
            return {"is_relevant": is_rel}
        except Exception as e:
            print(f"[Agent] Grader LLM failed: {e}. Relying on similarity.", flush=True)
            return {"is_relevant": top_sim >= 0.45}

    def _rewriter_node(self, state: AgentState) -> dict:
        """
        The Query Rewriter Node:
        Rewrites the query using alternative chemical terminology, IUPAC names,
        or related mechanisms for the re-retrieval loop.
        """
        orig_query = state["user_query"]
        curr_query = state["current_query"]
        print(f"[Agent] Rewriting query after critique. Previous: '{curr_query}'", flush=True)

        prompt = (
            "You are a chemistry search optimization specialist.\n"
            f"Original user question: {orig_query}\n"
            f"Previous unsuccessful query: {curr_query}\n\n"
            "Formulate ONE improved, concise search query using alternative chemical nomenclature, "
            "synonyms, molecular formulas, or reaction mechanisms to find the required data in a scientific paper.\n"
            "Output ONLY the single rewritten search query without quotes or explanations."
        )

        try:
            new_query = query_llm(prompt, system_prompt="You produce only a single concise search query string.").strip()
            new_query = new_query.split("\n")[0].strip('"\'')
            if not new_query:
                new_query = orig_query
        except Exception as e:
            print(f"[Agent] Rewriter failed: {e}. Reverting to original query.", flush=True)
            new_query = orig_query

        print(f"[Agent] New search query: '{new_query}'", flush=True)
        return {"current_query": new_query}

    def _format_sources_block(self, state: AgentState) -> str:
        """Formats retrieved parent chunks with paper name, page, heading, and snippet into collapsible markdown."""
        route = state.get("route", "vectorstore")
        if route == "web_search":
            return (
                "\n\n---\n"
                "<details class=\"sources-card\">\n"
                "<summary><b>🌐 Verified Source: NCBI PubChem Chemical Knowledgebase</b></summary>\n"
                "<div class=\"sources-list\">\n"
                "<div class=\"source-item\">\n"
                "Data retrieved and verified live from the National Center for Biotechnology Information (NCBI) PubChem Chemical Registry.\n"
                "</div>\n"
                "</div>\n"
                "</details>"
            )

        parents = state.get("retrieved_parents", [])
        if not parents:
            return ""

        lines = [
            "\n\n---\n",
            f"<details class=\"sources-card\"><summary><b>📚 Retrieved Sources & Citations ({len(parents)} sections from {len(set(p.get('paper_name', '') for p in parents))} papers)</b></summary>\n<div class=\"sources-list\">"
        ]
        for i, p in enumerate(parents, 1):
            paper = p.get("paper_name", "Research Paper")
            page = p.get("page", "?")
            heading = p.get("heading", "")
            sim = p.get("similarity", 0.0)
            snippet = p.get("best_child_match", p.get("text", "")[:250]).replace("\n", " ").strip()
            lines.append(f"<div class=\"source-item\">")
            header_parts = [f"<b>[{i}] 📄 {paper}</b>", f"<span class=\"page-badge\">Page {page}</span>"]
            if heading:
                header_parts.append(f"<span class=\"section-badge\">{heading}</span>")
            header_parts.append(f"<span class=\"sim-badge\">{int(sim * 100)}% match</span>")
            lines.append(f"<div class=\"source-header\">{' &bull; '.join(header_parts)}</div>")
            lines.append(f"<blockquote class=\"source-snippet\">\"{snippet}\"</blockquote>")
            lines.append("</div>")
        lines.append("</div>\n</details>")
        return "\n".join(lines)

    def _generator_node(self, state: AgentState) -> dict:
        """
        The Generation Node:
        Synthesizes the final answer using llama3.1:8b, formatting chemical equations in LaTeX,
        and attaches verified paper citations.
        """
        prompt = self._build_generation_prompt(state)
        answer = query_llm(
            prompt=prompt,
            system_prompt=CHEM_SYSTEM_PROMPT,
        )
        sources_block = self._format_sources_block(state)
        return {"final_answer": answer + sources_block}

    # ─── Conditional Routing Logic ───────────────────────────────────────────

    def _route_decision(self, state: AgentState) -> str:
        """Route to vectorstore or web_search."""
        return state.get("route", "vectorstore")

    def _grade_decision(self, state: AgentState) -> str:
        """
        If chunks are relevant or max retries reached, generate answer.
        Otherwise loop to rewriter.
        """
        is_relevant = state.get("is_relevant", False)
        retrieval_count = state.get("retrieval_count", 0)

        if is_relevant or retrieval_count >= MAX_REWRITE_RETRIES:
            if not is_relevant:
                print(f"[Agent] Max retrieval retries ({MAX_REWRITE_RETRIES}) reached. Proceeding to generation.", flush=True)
            return "generator"
        else:
            print("[Agent] Critique failed: Triggering re-retrieval loop.", flush=True)
            return "rewriter"

    # ─── Public API ──────────────────────────────────────────────────────────

    def _build_generation_prompt(self, state: AgentState) -> str:
        """Formats the context and history into the generation prompt."""
        history_lines = []
        for msg in state.get("chat_history", []):
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        history_str = "\n".join(history_lines) if history_lines else "No previous conversation."

        return (
            f"### Conversation History:\n{history_str}\n\n"
            f"### Verified Chemical & Scientific Context:\n{state.get('doc_context', 'No context provided.')}\n\n"
            f"### Current Question:\n{state.get('user_query', '')}\n\n"
            "Answer the question thoroughly and accurately based on the context above. "
            "Format all chemical formulas, molecular structures, and reaction mechanisms in strict LaTeX notation ($...$ or $$...$$)."
        )

    def ask(self, user_query: str, chat_history: list[dict] | None = None) -> str:
        """Execute full agent graph synchronously and return final answer."""
        initial_state: AgentState = {
            "user_query": user_query,
            "chat_history": chat_history or [],
            "route": "vectorstore",
            "current_query": user_query,
            "retrieval_count": 0,
            "retrieved_parents": [],
            "doc_context": "",
            "is_relevant": False,
            "final_answer": "",
        }

        final_state = self.graph.invoke(initial_state)
        return final_state.get("final_answer", "")

    def ask_stream(
        self,
        user_query: str,
        chat_history: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """
        Execute router, retrieval, and critique nodes, then stream tokens
        from the generation node.
        """
        # Step 1: Run graph up to generation preparation
        state: AgentState = {
            "user_query": user_query,
            "chat_history": chat_history or [],
            "route": "vectorstore",
            "current_query": user_query,
            "retrieval_count": 0,
            "retrieved_parents": [],
            "doc_context": "",
            "is_relevant": False,
            "final_answer": "",
        }

        # Step 1: Route
        route_out = self._router_node(state)
        state.update(route_out)

        if state["route"] == "web_search":
            pub_out = self._pubchem_node(state)
            state.update(pub_out)
        else:
            # Re-retrieval loop with grader
            while True:
                ret_out = self._retriever_node(state)
                state.update(ret_out)

                grade_out = self._grader_node(state)
                state.update(grade_out)

                if state["is_relevant"] or state["retrieval_count"] >= MAX_REWRITE_RETRIES:
                    break

                # Rewrite query and loop
                rewrite_out = self._rewriter_node(state)
                state.update(rewrite_out)

        # Step 2: Stream final generation tokens
        gen_prompt = self._build_generation_prompt(state)
        for token in query_llm(
            prompt=gen_prompt,
            system_prompt=CHEM_SYSTEM_PROMPT,
            stream=True,
        ):
            yield token

        # Yield verified paper sources block
        sources_block = self._format_sources_block(state)
        if sources_block:
            yield sources_block
