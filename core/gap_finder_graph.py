"""
Knowledge Gap Finder: Multi-Agent LangGraph pipeline (Step 1 of the research system).

Agent 1 (Question-Asking Agent): reads the retrieved paper set and generates
probing research questions about it.

Agent 2 (Answering Agent): runs the review phases and answers Agent 1's questions
from the evidence; questions it cannot answer well become knowledge gaps.

    Phase 1  Query Generation        topic -> search queries
    Phase 2  Literature Search       OpenAlex + uploaded PDFs (ChromaDB)
    Agent 1  Question Asking         papers -> research questions
    Phase 3  Abstract Summarization  abstracts -> Lit Gap Table rows
    Phase 4  Pattern Detection       table -> patterns, consensus, contradictions
    Phase 5  Gap Identification      population / variable / method / contradiction gaps
    Phase 6  Output                  Markdown Gap Report + candidate research questions
"""

import json
import re
from collections import Counter
from typing import TypedDict, Generator, Any

import requests
from langgraph.graph import StateGraph, END

from config.settings import (
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    GAP_MAX_PAPERS,
    GAP_NUM_QUERIES,
    OPENALEX_PER_QUERY,
    GAP_LOCAL_SECTIONS,
    GAP_SYSTEM_PROMPT,
    GAP_LOCAL_MIN_SIM,
)
from core.llm import get_available_llm_model
from core.openalex_tool import search_openalex
from core.vectorstore import VectorStore

GAP_TYPES = ("population", "variable", "method", "contradiction")
TABLE_FIELDS = ("population_or_system", "variables", "method", "key_finding", "limitation")
_NUM_CTX = 8192  # llama3's full context; Ollama's default is smaller


class GapState(TypedDict, total=False):
    topic: str
    max_papers: int
    include_local: bool
    search_queries: list[str]
    papers: list[dict]
    agent1_questions: list[str]
    gap_table: list[dict]
    patterns: dict
    answers: list[dict]
    gaps: list[dict]
    research_questions: list[str]
    report_md: str


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def _ask_llm(prompt: str, json_mode: bool = True, timeout: int = 300) -> str:
    """
    Call Ollama's native /api/chat so we can set num_ctx (full 8k context)
    and format=json (forces syntactically valid JSON from small local models).
    """
    payload: dict[str, Any] = {
        "model": get_available_llm_model(),
        "messages": [
            {"role": "system", "content": GAP_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": LLM_TEMPERATURE, "num_ctx": _NUM_CTX},
    }
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama API error ({r.status_code}): {r.text[:200]}")
    return r.json().get("message", {}).get("content", "")


def _parse_json(text: str, default: Any) -> Any:
    """Parse JSON from an LLM response, tolerating surrounding prose."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text or "", re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return default


def _llm_json(prompt: str, default: Any) -> Any:
    try:
        return _parse_json(_ask_llm(prompt, json_mode=True), default)
    except Exception as e:
        print(f"[GapFinder] LLM call failed: {e}", flush=True)
        return default


_FILLER = {"", "none", "n/a", "na", "not stated", "unclear", "unknown", "-", "—", "no", "nothing"}


def _is_filler(text: str) -> bool:
    return text.strip().strip(".").lower() in _FILLER


def _as_str_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, dict):
            # Models sometimes return objects, e.g. {"population": "...", "question": "..."}.
            # Prefer a question/text-like key, else the longest string value.
            strs = {k: x for k, x in v.items() if isinstance(x, str)}
            preferred = [x for k, x in strs.items()
                         if any(w in k.lower() for w in ("question", "text", "query", "description"))]
            v = (preferred or sorted(strs.values(), key=len, reverse=True) or [""])[0]
        if isinstance(v, str) and not _is_filler(v):
            out.append(v.strip())
    return out[:limit]


def _first_list(obj: Any) -> list:
    """Return obj if it's a list, else the first list value inside a dict."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                return v
    return []


def _cell(value: Any) -> str:
    """Make a value safe for a Markdown table cell."""
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    text = str(value or "").replace("|", "/").replace("\n", " ").strip()
    return text or "—"


# ─── Graph ───────────────────────────────────────────────────────────────────

class GapFinderGraph:
    """LangGraph state machine for the Knowledge Gap Finder."""

    PHASES = [
        ("query_gen", "Phase 1 · Query Generation"),
        ("search", "Phase 2 · Literature Search"),
        ("question_agent", "Agent 1 · Question Asking"),
        ("summarize", "Phase 3 · Abstract Summaries & Lit Gap Table"),
        ("patterns", "Phase 4 · Pattern Detection"),
        ("gaps", "Phase 5 · Gap Identification"),
        ("report", "Phase 6 · Gap Report"),
    ]

    def __init__(self, store: VectorStore):
        self.store = store
        self._nodes = {
            "query_gen": self._query_gen_node,
            "search": self._search_node,
            "question_agent": self._question_agent_node,
            "summarize": self._summarize_node,
            "patterns": self._pattern_node,
            "gaps": self._gap_node,
            "report": self._report_node,
        }
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GapState)
        for name, fn in self._nodes.items():
            builder.add_node(name, fn)
        builder.set_entry_point("query_gen")
        order = [name for name, _ in self.PHASES]
        for a, b in zip(order, order[1:]):
            builder.add_edge(a, b)
        builder.add_edge(order[-1], END)
        return builder.compile()

    # ─── Phase 1: Query Generation ───────────────────────────────────────────

    def _query_gen_node(self, state: GapState) -> dict:
        topic = state["topic"]
        prompt = (
            f"Research topic: {topic}\n\n"
            f"Write {GAP_NUM_QUERIES} distinct academic search queries to find papers on this topic "
            "for a literature review. Cover synonyms, key sub-areas, and methods. "
            "Each query: 3-8 words, no boolean operators.\n"
            'Respond as JSON: {"queries": ["...", "..."]}'
        )
        queries = _as_str_list(_first_list(_llm_json(prompt, {})), GAP_NUM_QUERIES)
        # Always search the raw topic first; dedupe case-insensitively
        seen, final = set(), []
        for q in [topic] + queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                final.append(q)
        return {"search_queries": final[: GAP_NUM_QUERIES + 1]}

    # ─── Phase 2: Literature Search ──────────────────────────────────────────

    def _search_node(self, state: GapState) -> dict:
        max_papers = state.get("max_papers") or GAP_MAX_PAPERS
        by_id: dict[str, dict] = {}
        rank: dict[str, float] = {}
        for qi, q in enumerate(state["search_queries"]):
            for pos, p in enumerate(search_openalex(q, OPENALEX_PER_QUERY)):
                if not p["id"]:
                    continue
                # Reciprocal-rank fusion across queries; papers found by several queries rise
                rank[p["id"]] = rank.get(p["id"], 0.0) + 1.0 / (pos + 1 + qi)
                by_id.setdefault(p["id"], p)

        ranked = sorted(
            by_id.values(),
            key=lambda p: (rank[p["id"]], p["cited_by_count"]),
            reverse=True,
        )[:max_papers]

        papers = []
        for i, p in enumerate(ranked, 1):
            papers.append({**p, "ref": f"P{i}"})

        if state.get("include_local", True) and self.store.count > 0:
            try:
                sections = self.store.search_parent_docs(
                    state["topic"], top_k=GAP_LOCAL_SECTIONS * 2, threshold=GAP_LOCAL_MIN_SIM
                )
            except Exception as e:
                print(f"[GapFinder] Local search failed: {e}", flush=True)
                sections = []
            # Drop near-duplicate sections (same text prefix), keep the best-matching ones
            seen_text, unique = set(), []
            for s in sections:
                key = s.get("text", "")[:300]
                if key not in seen_text:
                    seen_text.add(key)
                    unique.append(s)
            for j, s in enumerate(unique[:GAP_LOCAL_SECTIONS], 1):
                papers.append({
                    "ref": f"L{j}",
                    "id": s.get("parent_id", ""),
                    "title": f"{s.get('paper_name', 'Uploaded paper')} (p. {s.get('page', '?')}"
                             + (f", {s['heading']}" if s.get("heading") else "") + ")",
                    "year": None,
                    "authors": [],
                    "venue": "Uploaded PDF",
                    "doi": "",
                    "url": "",
                    "cited_by_count": 0,
                    "abstract": s.get("text", ""),
                    "source": "local",
                })

        print(f"[GapFinder] Search found {len(papers)} papers/sections.", flush=True)
        return {"papers": papers}

    # ─── Agent 1: Question Asking ────────────────────────────────────────────

    def _question_agent_node(self, state: GapState) -> dict:
        titles = "\n".join(f"{p['ref']}: {p['title']}" for p in state["papers"])
        prompt = (
            f"Research topic: {state['topic']}\n\nPapers found:\n{titles}\n\n"
            "You are the Question-Asking Agent. Write 6-8 probing questions that a systematic "
            "review of these papers should answer to expose knowledge gaps. Cover: which "
            "populations/materials/systems are studied and which are missing; which variables "
            "and outcomes are measured; which methods dominate and which are absent; and where "
            "findings may disagree.\n"
            'Respond as JSON: {"questions": ["...", "..."]}'
        )
        questions = _as_str_list(_first_list(_llm_json(prompt, {})), 8)
        if not questions:
            questions = [
                f"Which populations, materials or systems are studied in research on {state['topic']}, and which are missing?",
                "Which independent and dependent variables are measured most often, and which are rarely examined?",
                "Which research methods dominate, and which methodological approaches are absent?",
                "Do any studies report contradictory or inconsistent findings?",
                "What limitations do the authors themselves report?",
            ]
        return {"agent1_questions": questions}

    # ─── Phase 3: Abstract Reading & Summarization (Lit Gap Table) ───────────

    def summarize_paper(self, paper: dict) -> dict:
        prompt = (
            f"Paper {paper['ref']}: {paper['title']}\n"
            f"Text:\n{paper['abstract'][:2000]}\n\n"
            "Extract the following from the text only. Use 'not stated' when absent. "
            "Keep each value under 20 words.\n"
            "- population_or_system: who/what was studied (population, material, organism, system)\n"
            "- variables: key independent and dependent variables or outcomes measured\n"
            "- method: study design / experimental or computational technique\n"
            "- key_finding: the main result\n"
            "- limitation: a stated or obvious limitation\n"
            'Respond as JSON with exactly these keys: {"population_or_system": "", "variables": "", '
            '"method": "", "key_finding": "", "limitation": ""}'
        )
        data = _llm_json(prompt, {})
        if not isinstance(data, dict):
            data = {}
        row = {"ref": paper["ref"]}
        for f in TABLE_FIELDS:
            row[f] = _cell(data.get(f)) if data.get(f) else "unclear"
        return row

    def _summarize_node(self, state: GapState) -> dict:
        return {"gap_table": [self.summarize_paper(p) for p in state["papers"]]}

    # ─── Phase 4: Pattern Detection ──────────────────────────────────────────

    def _table_text(self, state: GapState) -> str:
        years = {p["ref"]: p.get("year") for p in state["papers"]}
        lines = []
        for r in state["gap_table"]:
            lines.append(
                f"{r['ref']} ({years.get(r['ref']) or 'n/a'}) | system: {r['population_or_system']} | "
                f"variables: {r['variables']} | method: {r['method']} | finding: {r['key_finding']} | "
                f"limitation: {r['limitation']}"
            )
        return "\n".join(lines)

    def _pattern_node(self, state: GapState) -> dict:
        prompt = (
            f"Research topic: {state['topic']}\n\nLit Gap Table (one row per paper):\n"
            f"{self._table_text(state)}\n\n"
            "Identify patterns across the papers. Cite paper refs (e.g. P3) for every point.\n"
            'Respond as JSON: {"dominant_methods": ["..."], "common_populations": ["..."], '
            '"common_variables": ["..."], "consensus_findings": ["..."], '
            '"contradictions": ["..."], "recurring_limitations": ["..."]}'
        )
        llm = _llm_json(prompt, {})
        if not isinstance(llm, dict):
            llm = {}
        keys = ("dominant_methods", "common_populations", "common_variables",
                "consensus_findings", "contradictions", "recurring_limitations")
        patterns = {k: _as_str_list(llm.get(k), 6) for k in keys}

        # Deterministic stats (no LLM)
        ext = [p for p in state["papers"] if p["source"] == "openalex"]
        years = [p["year"] for p in ext if p.get("year")]
        cites = [p["cited_by_count"] for p in ext]
        method_counts = Counter(
            r["method"].lower() for r in state["gap_table"]
            if r["method"] not in ("unclear", "not stated", "—")
        )
        patterns["stats"] = {
            "n_papers": len(ext),
            "n_local": len(state["papers"]) - len(ext),
            "year_range": f"{min(years)}–{max(years)}" if years else "n/a",
            "papers_by_year": dict(sorted(Counter(years).items())),
            "citations_range": f"{min(cites)}–{max(cites)}" if cites else "n/a",
            "repeated_methods": [m for m, c in method_counts.most_common(5) if c > 1],
        }
        return {"patterns": patterns}

    # ─── Phase 5: Gap Identification (Agent 2 answers Agent 1) ───────────────

    def _gap_node(self, state: GapState) -> dict:
        table = self._table_text(state)
        pat = {k: v for k, v in state["patterns"].items() if k != "stats"}
        questions = "\n".join(f"Q{i}. {q}" for i, q in enumerate(state["agent1_questions"], 1))

        # Step 1: Agent 2 answers each of Agent 1's questions from the evidence
        answer_prompt = (
            f"Research topic: {state['topic']}\n\nLit Gap Table:\n{table}\n\n"
            f"Detected patterns:\n{json.dumps(pat, ensure_ascii=False)}\n\n"
            f"Questions from the Question-Asking Agent:\n{questions}\n\n"
            "You are the Answering Agent. Answer each question using ONLY the table, citing refs. "
            "Rate coverage: 'strong' (several papers address it), 'thin' (one paper or indirect), "
            "or 'none' (no paper addresses it).\n"
            'Respond as JSON: {"answers": [{"question": "...", "answer": "...", '
            '"coverage": "strong|thin|none", "refs": ["P1"]}]}'
        )
        answers = []
        for a in _first_list(_llm_json(answer_prompt, {})):
            if isinstance(a, dict) and a.get("question"):
                cov = str(a.get("coverage", "")).lower()
                answers.append({
                    "question": str(a["question"]),
                    "answer": str(a.get("answer", "")),
                    "coverage": cov if cov in ("strong", "thin", "none") else "thin",
                    "refs": _as_str_list(a.get("refs"), 10),
                })

        # Step 2: Turn weak coverage, patterns and contradictions into typed gaps
        weak = [a for a in answers if a["coverage"] != "strong"]
        gap_prompt = (
            f"Research topic: {state['topic']}\n\nLit Gap Table:\n{table}\n\n"
            f"Detected patterns:\n{json.dumps(pat, ensure_ascii=False)}\n\n"
            f"Questions the literature answers poorly:\n"
            f"{json.dumps(weak, ensure_ascii=False) if weak else 'none identified'}\n\n"
            "Identify specific knowledge gaps. Each gap has a type:\n"
            "- population: groups, materials, or systems that are understudied or missing\n"
            "- variable: variables, outcomes, or interactions not measured\n"
            "- method: designs or techniques missing or over-relied upon (e.g. no long-term, "
            "no in-situ, no large-scale, no computational validation)\n"
            "- contradiction: ONLY when two or more papers report opposing or inconsistent "
            "results on the same question. Name both sides, e.g. 'P2 reports X while P5 reports Y'. "
            "A limitation or open problem in a single paper is NOT a contradiction, and two "
            "different approaches that both succeed are NOT a contradiction.\n"
            "Write each description as one full sentence saying what is missing and why it "
            "matters (not a short label). Give 2-3 gaps per type where the evidence supports it; "
            "skip a type if the evidence does not support it. Cite the refs that show the gap.\n"
            'Respond as JSON: {"gaps": [{"type": "population|variable|method|contradiction", '
            '"description": "...", "evidence": ["P1", "P4"]}]}'
        )
        valid_refs = {p["ref"] for p in state["papers"]}
        gaps = []
        for g in _first_list(_llm_json(gap_prompt, {})):
            if not isinstance(g, dict) or not g.get("description"):
                continue
            gtype = str(g.get("type", "")).lower().strip()
            gtype = next((t for t in GAP_TYPES if t in gtype), None)
            if not gtype:
                continue
            evidence = [r for r in _as_str_list(g.get("evidence"), 10) if r in valid_refs]
            if gtype == "contradiction" and len(set(evidence)) < 2:
                # A contradiction needs at least two disagreeing papers; otherwise it's a limitation
                continue
            gaps.append({"type": gtype, "description": str(g["description"]).strip(), "evidence": evidence})

        # Step 3: candidate research questions from the gaps
        rqs: list[str] = []
        if gaps:
            rq_prompt = (
                f"Research topic: {state['topic']}\n\nIdentified knowledge gaps:\n"
                + "\n".join(f"- [{g['type']}] {g['description']}" for g in gaps)
                + "\n\nPropose 3-5 specific, testable research questions that would address "
                "these gaps. Each must be ONE complete question sentence ending in '?' that names "
                "the population/system, the variables, and the method.\n"
                'Respond as JSON: {"research_questions": ["How does ... affect ... in ...?", "..."]}'
            )
            # Keep only real questions, not bare topic phrases
            rqs = [q for q in _as_str_list(_first_list(_llm_json(rq_prompt, {})), 8)
                   if len(q.split()) >= 6][:5]

        return {"answers": answers, "gaps": gaps, "research_questions": rqs}

    # ─── Phase 6: Output (Gap Report) ────────────────────────────────────────

    def _report_node(self, state: GapState) -> dict:
        papers = state["papers"]
        by_ref = {p["ref"]: p for p in papers}
        pat = state["patterns"]
        stats = pat.get("stats", {})

        def ref_link(ref: str) -> str:
            p = by_ref.get(ref)
            return f"[{ref}]({p['url']})" if p and p.get("url") else ref

        md: list[str] = [f"# Knowledge Gap Report: {state['topic']}", ""]
        md.append(
            f"Reviewed **{stats.get('n_papers', 0)} papers** from OpenAlex "
            f"({stats.get('year_range', 'n/a')}, citations {stats.get('citations_range', 'n/a')})"
            + (f" and **{stats['n_local']} sections** from your uploaded PDFs" if stats.get("n_local") else "")
            + f". Found **{len(state['gaps'])} gaps**."
        )
        md += ["", "**Search queries:** " + "; ".join(f"`{q}`" for q in state["search_queries"]), ""]

        # Gaps first: they're the point of the report
        md += ["## Knowledge Gaps", ""]
        titles = {
            "population": "Population / System Gaps",
            "variable": "Variable Gaps",
            "method": "Method Gaps",
            "contradiction": "Contradictions",
        }
        for t in GAP_TYPES:
            items = [g for g in state["gaps"] if g["type"] == t]
            md.append(f"### {titles[t]}")
            if not items:
                md.append("_None identified from this paper set._")
            for g in items:
                ev = ", ".join(ref_link(r) for r in g["evidence"]) or "no specific paper cited"
                md.append(f"- {g['description']} _(evidence: {ev})_")
            md.append("")

        md += ["## Candidate Research Questions", ""]
        if state["research_questions"]:
            md += [f"{i}. {q}" for i, q in enumerate(state["research_questions"], 1)]
        else:
            md.append("_No research questions generated._")
        md.append("")

        md += ["## Patterns", ""]
        labels = [
            ("dominant_methods", "Dominant methods"),
            ("common_populations", "Common populations / systems"),
            ("common_variables", "Common variables"),
            ("consensus_findings", "Consensus findings"),
            ("contradictions", "Possible contradictions"),
            ("recurring_limitations", "Recurring limitations"),
        ]
        for key, label in labels:
            if pat.get(key):
                md.append(f"**{label}**")
                md += [f"- {x}" for x in pat[key]]
                md.append("")
        if stats.get("papers_by_year"):
            md.append("**Papers by year:** " + ", ".join(f"{y}: {c}" for y, c in stats["papers_by_year"].items()))
            md.append("")

        md += ["## Agent 1 Questions & Agent 2 Answers", ""]
        if state["answers"]:
            icon = {"strong": "🟢 strong", "thin": "🟡 thin", "none": "🔴 none"}
            for a in state["answers"]:
                refs = ", ".join(ref_link(r) for r in a["refs"] if r in by_ref)
                answer = a["answer"] if not _is_filler(a["answer"]) else "No paper in this set addresses this."
                md.append(f"- **{a['question']}** ({icon[a['coverage']]} coverage)  ")
                md.append(f"  {answer}" + (f" _({refs})_" if refs else ""))
        else:
            md += [f"- {q}" for q in state["agent1_questions"]]
        md.append("")

        md += ["## Lit Gap Table", "",
               "| Ref | Year | Population / System | Variables | Method | Key Finding | Limitation |",
               "|---|---|---|---|---|---|---|"]
        for r in state["gap_table"]:
            p = by_ref[r["ref"]]
            md.append(
                f"| {ref_link(r['ref'])} | {p.get('year') or '—'} | {r['population_or_system']} | "
                f"{r['variables']} | {r['method']} | {r['key_finding']} | {r['limitation']} |"
            )
        md.append("")

        md += ["## Sources", ""]
        for p in papers:
            authors = ", ".join(p["authors"]) if p["authors"] else ""
            meta = " · ".join(x for x in [authors, str(p["year"] or ""), p["venue"],
                                          f"{p['cited_by_count']} citations" if p["source"] == "openalex" else ""] if x)
            title = f"[{_cell(p['title'])}]({p['url']})" if p.get("url") else _cell(p["title"])
            md.append(f"- **{p['ref']}** {title}" + (f" — {meta}" if meta else ""))

        return {"report_md": "\n".join(md)}

    # ─── Public API ──────────────────────────────────────────────────────────

    def _initial_state(self, topic: str, max_papers: int | None, include_local: bool) -> GapState:
        return {
            "topic": topic.strip(),
            "max_papers": max(3, min(max_papers or GAP_MAX_PAPERS, 30)),
            "include_local": include_local,
            "search_queries": [], "papers": [], "agent1_questions": [], "gap_table": [],
            "patterns": {}, "answers": [], "gaps": [], "research_questions": [], "report_md": "",
        }

    def run(self, topic: str, max_papers: int | None = None, include_local: bool = True) -> str:
        """Run the full graph and return the Markdown report."""
        final = self.graph.invoke(self._initial_state(topic, max_papers, include_local))
        return final.get("report_md", "")

    def run_stream(
        self, topic: str, max_papers: int | None = None, include_local: bool = True
    ) -> Generator[dict, None, None]:
        """
        Run the phases in order, yielding progress events:
          {"type": "progress", "phase": id, "label": str, "status": "start"|"done", "detail": str}
          {"type": "report", "markdown": str, "gaps": [...], "papers": int}
        Phase 3 also yields per-paper progress since it is the slow step.
        """
        state = self._initial_state(topic, max_papers, include_local)
        for phase_id, label in self.PHASES:
            yield {"type": "progress", "phase": phase_id, "label": label, "status": "start", "detail": ""}

            if phase_id == "summarize":
                table = []
                total = len(state["papers"])
                for i, p in enumerate(state["papers"], 1):
                    table.append(self.summarize_paper(p))
                    yield {"type": "progress", "phase": phase_id, "label": label, "status": "step",
                           "detail": f"Read {i}/{total}: {p['title'][:80]}"}
                state["gap_table"] = table
            else:
                state.update(self._nodes[phase_id](state))

            if phase_id == "search" and not state["papers"]:
                yield {"type": "error", "message": "No papers with abstracts found for this topic. "
                                                   "Try a broader or differently worded topic."}
                return

            yield {"type": "progress", "phase": phase_id, "label": label, "status": "done",
                   "detail": self._phase_detail(phase_id, state)}

        yield {"type": "report", "markdown": state["report_md"],
               "gaps": state["gaps"], "papers": len(state["papers"])}

    @staticmethod
    def _phase_detail(phase_id: str, state: GapState) -> str:
        if phase_id == "query_gen":
            return f"{len(state['search_queries'])} queries: " + "; ".join(state["search_queries"])
        if phase_id == "search":
            n_local = sum(1 for p in state["papers"] if p["source"] == "local")
            return f"{len(state['papers']) - n_local} papers from OpenAlex, {n_local} uploaded-PDF sections"
        if phase_id == "question_agent":
            return f"{len(state['agent1_questions'])} questions"
        if phase_id == "summarize":
            return f"{len(state['gap_table'])} rows in Lit Gap Table"
        if phase_id == "patterns":
            n = sum(len(v) for k, v in state["patterns"].items() if k != "stats")
            return f"{n} patterns detected"
        if phase_id == "gaps":
            counts = Counter(g["type"] for g in state["gaps"])
            return ", ".join(f"{counts.get(t, 0)} {t}" for t in GAP_TYPES)
        return "Report ready"
