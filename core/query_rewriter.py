"""
Query Rewriter — uses LLM to transform user questions into
search-optimized queries for better vector retrieval.
"""

from core.llm import query_llm

REWRITE_SYSTEM_PROMPT = """\
You are a search query optimizer for a scientific document retrieval system.
Your job is to rewrite the user's natural language question into 3 distinct, 
search-optimized queries that will retrieve the most relevant text chunks 
from a research paper.

Rules:
- Output EXACTLY 3 queries, one per line, numbered 1. 2. 3.
- Each query should target the question from a DIFFERENT angle.
- Include specific keywords, synonyms, and related terms.
- If the question mentions a figure, table, or equation number, include both 
  the number AND descriptive terms (e.g., "caption", "description", "shows").
- Keep each query concise (under 30 words).
- Do NOT include any explanation, just the 3 queries."""

REWRITE_USER_TEMPLATE = """\
Rewrite this question into 3 search-optimized queries:

Question: {question}"""


def rewrite_query(user_question: str, model: str | None = None) -> list[str]:
    """
    Use the LLM to rewrite a user question into multiple search queries.

    Args:
        user_question: The raw user question.
        model: Optional model override.

    Returns:
        List of 2-3 rewritten search queries.
    """
    prompt = REWRITE_USER_TEMPLATE.format(question=user_question)

    try:
        response = query_llm(
            prompt,
            system_prompt=REWRITE_SYSTEM_PROMPT,
            model=model,
        )

        # Parse numbered lines (1. ... 2. ... 3. ...)
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading number + period/parenthesis
            cleaned = line.lstrip("0123456789.)- ").strip()
            if cleaned:
                queries.append(cleaned)

        # Always include the original question as a fallback
        if not queries:
            return [user_question]

        return queries[:3]

    except Exception as e:
        print(f"[!] Query rewrite failed: {e}. Using original query.", flush=True)
        return [user_question]
