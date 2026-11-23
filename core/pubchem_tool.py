"""
PubChem & Chemical Knowledge Tool.
Queries the National Center for Biotechnology Information (NCBI) PubChem REST API
(PUG-REST) to retrieve compound properties, molecular formulas in LaTeX, SMILES,
IUPAC nomenclature, and descriptions for general chemistry queries.
"""

import re
import urllib.parse
from typing import Optional
import requests


def search_pubchem_compound(compound_name: str) -> Optional[dict]:
    """
    Query PubChem PUG REST API for chemical properties by compound name or formula.

    Returns:
        Dictionary of chemical properties or None if not found.
    """
    clean_name = compound_name.strip()
    encoded = urllib.parse.quote(clean_name)
    prop_url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/"
        f"property/MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,InChIKey,Title/JSON"
    )

    desc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/description/JSON"

    headers = {"User-Agent": "ChemRAG-Agent/2.0"}

    result = {}
    try:
        r_prop = requests.get(prop_url, headers=headers, timeout=10)
        if r_prop.status_code == 200:
            data = r_prop.json()
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                result["properties"] = props[0]
    except Exception as e:
        print(f"[!] PubChem properties lookup failed for '{compound_name}': {e}", flush=True)

    try:
        r_desc = requests.get(desc_url, headers=headers, timeout=10)
        if r_desc.status_code == 200:
            data = r_desc.json()
            summaries = data.get("InformationList", {}).get("Information", [])
            descriptions = []
            for s in summaries:
                desc = s.get("Description")
                if desc:
                    descriptions.append(desc)
            if descriptions:
                result["description"] = "\n\n".join(descriptions[:2])
    except Exception as e:
        print(f"[!] PubChem description lookup failed for '{compound_name}': {e}", flush=True)

    return result if result else None


def format_pubchem_result(compound_name: str, data: dict) -> str:
    """Format PubChem compound data into rich Markdown with LaTeX notation."""
    props = data.get("properties", {})
    desc = data.get("description", "No extended description available.")

    formula = props.get("MolecularFormula", "")
    # Format formula with LaTeX subscripts: C60 -> \mathrm{C}_{60}
    latex_formula = re.sub(r'([A-Za-z]+)(\d+)', r'\\mathrm{\1}_{\2}', formula) if formula else formula

    lines = [
        f"### PubChem Verified Data: {props.get('Title', compound_name.title())}",
        f"- **Molecular Formula:** ${latex_formula}$" if latex_formula else "",
        f"- **Molecular Weight:** {props.get('MolecularWeight', 'N/A')} g/mol",
        f"- **IUPAC Name:** `{props.get('IUPACName', 'N/A')}`",
        f"- **Canonical SMILES:** `{props.get('CanonicalSMILES', 'N/A')}`",
        f"- **InChIKey:** `{props.get('InChIKey', 'N/A')}`",
        f"- **PubChem CID:** {props.get('CID', 'N/A')}",
        "",
        "**Summary & Chemical Description:**",
        desc,
    ]
    return "\n".join([l for l in lines if l is not None and l != ""])


def query_chemistry_knowledge(query: str) -> str:
    """
    Search general chemistry knowledge via PubChem compound database.
    Extracts potential chemical names from the query.
    """
    # Remove common filler words
    stop_words = {
        "what", "is", "the", "of", "and", "in", "for", "formula", "structure",
        "molecular", "weight", "mass", "property", "properties", "tell", "me",
        "about", "how", "does", "react", "cas", "number", "give", "can", "you",
        "describe", "explain", "iupac", "name", "smiles"
    }
    words = re.findall(r'[A-Za-z0-9_\-]+', query)
    candidates = [w for w in words if w.lower() not in stop_words and len(w) > 2]

    # Try full query first, then candidates
    search_terms = []
    if len(candidates) <= 3:
        search_terms.append(" ".join(candidates))
    search_terms.extend(candidates)

    for term in search_terms:
        data = search_pubchem_compound(term)
        if data:
            return format_pubchem_result(term, data)

    return f"No direct PubChem compound records found for search terms from: '{query}'. Provide chemical general explanation using foundational chemistry principles."

