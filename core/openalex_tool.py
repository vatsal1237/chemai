"""
OpenAlex literature search tool.
Queries the free OpenAlex Works API (no API key required) and returns
normalized paper records with reconstructed abstracts.
"""

import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_HEADERS = {"User-Agent": "ChemRAG-GapFinder/1.0"}


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    """Rebuild plain abstract text from OpenAlex's abstract_inverted_index."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(word for _, word in positions)


def _normalize(work: dict) -> dict:
    authorships = work.get("authorships") or []
    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in authorships[:3]
        if (a.get("author") or {}).get("display_name")
    ]
    if len(authorships) > 3:
        authors.append("et al.")

    source = ((work.get("primary_location") or {}).get("source") or {})
    doi = work.get("doi") or ""

    return {
        "id": (work.get("id") or "").rsplit("/", 1)[-1],  # e.g. W2741809807
        "title": work.get("display_name") or work.get("title") or "Untitled",
        "year": work.get("publication_year"),
        "authors": authors,
        "venue": source.get("display_name") or "",
        "doi": doi,
        "url": doi or work.get("id") or "",
        "cited_by_count": work.get("cited_by_count") or 0,
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "source": "openalex",
    }


def search_openalex(query: str, per_page: int = 10) -> list[dict]:
    """
    Search OpenAlex for works matching `query` that have an abstract.
    Results come back in OpenAlex relevance order. Returns [] on any error.
    """
    params = {
        "search": query,
        "filter": "has_abstract:true,type:article|review|preprint",
        "per_page": max(1, min(per_page, 50)),
        "select": "id,doi,display_name,publication_year,authorships,"
                  "primary_location,cited_by_count,abstract_inverted_index",
    }
    try:
        r = requests.get(OPENALEX_WORKS_URL, params=params, headers=_HEADERS, timeout=20)
        if r.status_code != 200:
            print(f"[OpenAlex] HTTP {r.status_code} for query '{query}'", flush=True)
            return []
        works = r.json().get("results", [])
    except Exception as e:
        print(f"[OpenAlex] Search failed for '{query}': {e}", flush=True)
        return []

    papers = [_normalize(w) for w in works]
    return [p for p in papers if len(p["abstract"]) > 80]
