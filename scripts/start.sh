#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh – Bootstrap the Agentic Chemistry RAG stack
# (Ollama + LangGraph FastAPI Agent Server + Open WebUI)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "🚀 Starting all containers via Docker Compose …"
docker compose up -d

# ── Wait for Ollama to become healthy ──────────────────────────────────────
echo "⏳ Waiting for Ollama container to report healthy …"
MAX_RETRIES=30
RETRY_INTERVAL=5
for i in $(seq 1 $MAX_RETRIES); do
    if docker inspect --format='{{.State.Health.Status}}' ollama 2>/dev/null | grep -q "healthy"; then
        echo "✅ Ollama is healthy!"
        break
    fi
    if [ "$i" -eq "$MAX_RETRIES" ]; then
        echo "❌ Ollama did not become healthy after $((MAX_RETRIES * RETRY_INTERVAL))s. Check logs: docker compose logs ollama"
        exit 1
    fi
    echo "   … attempt $i/$MAX_RETRIES (retrying in ${RETRY_INTERVAL}s)"
    sleep "$RETRY_INTERVAL"
done

# ── Pre-load / Pull required local open-weight models ─────────────────────
echo "📦 Pulling llama3.1:8b for Generation & Reasoning …"
docker exec ollama ollama pull llama3.1:8b || echo "[!] Notice: llama3.1:8b pull deferred"

echo "📦 Pulling nomic-embed-text for Parent-Document Vector Embeddings …"
docker exec ollama ollama pull nomic-embed-text || echo "[!] Notice: nomic-embed-text pull deferred"

echo "📦 Pulling qwen2.5vl for Local Vision PDF Parsing …"
docker exec ollama ollama pull qwen2.5vl || docker exec ollama ollama pull llama3.2-vision || echo "[!] Notice: vision model pull deferred"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════════════════"
echo "  🧪  Agentic Chemistry RAG System is ONLINE!"
echo ""
echo "  🔗  Open WebUI (Chat):      http://localhost:3000"
echo "  🔗  FastAPI Agent Server:   http://localhost:8000"
echo "  🔗  API Docs (Swagger):     http://localhost:8000/docs"
echo "  🔗  Ollama Engine:          http://localhost:11434"
echo ""
echo "  Agent:  chem-rag-agent (LangGraph + Parent-Doc Retrieval + PubChem)"
echo "═════════════════════════════════════════════════════════════════════"
