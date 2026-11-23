"""
Run the Knowledge Gap Finder from the command line.

Usage:
    python scripts/gap_report.py "perovskite solar cell stability" [--max-papers 15] [--no-local] [--out report.md]
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL", "llama3:latest")

from core.vectorstore import VectorStore  # noqa: E402
from core.gap_finder_graph import GapFinderGraph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Knowledge Gap Finder")
    ap.add_argument("topic")
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--no-local", action="store_true", help="skip uploaded PDFs")
    ap.add_argument("--out", type=Path, default=None, help="write the Markdown report here")
    args = ap.parse_args()

    graph = GapFinderGraph(store=VectorStore())
    t0 = time.time()
    report = ""
    for ev in graph.run_stream(args.topic, max_papers=args.max_papers, include_local=not args.no_local):
        elapsed = f"[{time.time() - t0:6.1f}s]"
        if ev["type"] == "progress":
            if ev["status"] == "start":
                print(f"{elapsed} ▶ {ev['label']}")
            else:
                print(f"{elapsed}   {'✓' if ev['status'] == 'done' else '·'} {ev['detail']}")
        elif ev["type"] == "error":
            print(f"{elapsed} ✗ {ev['message']}")
            return 1
        elif ev["type"] == "report":
            report = ev["markdown"]

    if args.out:
        args.out.write_text(report)
        print(f"\nReport written to {args.out}")
    else:
        print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
