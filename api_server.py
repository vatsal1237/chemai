"""
FastAPI Server for Agentic Chemistry RAG (Open WebUI & OpenAI Compatible).
Orchestrates LangGraph state machine, local vision PDF ingestion, and
provides a sequential concurrency queue to protect local GPU/CPU resources.
"""

import os
import time
import json
import uuid
import asyncio
import threading
from typing import Optional, AsyncGenerator, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from config.settings import (
    DATA_DIR,
    CHEM_AGENT_MODEL_NAME,
    OLLAMA_BASE_URL,
    LLM_MODEL,
    EMBEDDING_MODEL,
    VISION_MODEL,
)
from core.vectorstore import VectorStore
from core.parser import parse_pdf
from core.chunker import chunk_parent_child
from core.agent_graph import ChemAgentGraph
from core.gap_finder_graph import GapFinderGraph

# ---------------------------------------------------------------------------
# App initialization & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agentic Chemistry RAG API",
    version="2.0.0",
    description="Local Agentic RAG system for scientific chemistry papers using LangGraph & Ollama.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global State & Sequential Concurrency Queue
# ---------------------------------------------------------------------------
store = VectorStore()
agent_graph = ChemAgentGraph(store=store)
gap_graph = GapFinderGraph(store=store)

# Sequential inference lock: protects local GPU/Metal VRAM from concurrent thrashing
# Multiple users can connect simultaneously; requests are queued and processed in order
inference_lock = asyncio.Lock()
active_queue_count = 0

MODEL_ID = CHEM_AGENT_MODEL_NAME


# ---------------------------------------------------------------------------
# Pydantic models (OpenAI Compatibility)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = MODEL_ID
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class GapReportRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    max_papers: Optional[int] = Field(default=None, ge=3, le=30)
    include_local: bool = True


class ModelData(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelData]


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
@app.get("/api/status")
async def health_status():
    return {
        "status": "online",
        "system": "Agentic Chemistry RAG",
        "agent": MODEL_ID,
        "ollama_base": OLLAMA_BASE_URL,
        "llm_model": LLM_MODEL,
        "vision_model": VISION_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "child_vectors": store.count,
        "parent_sections": len(store.parent_docstore),
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def web_ui():
    """Interactive Web Chat UI with real-time streaming, KaTeX chemical rendering, and PDF upload."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chem-RAG Agent | Scientific Research Assistant</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {
            --bg-color: #0b0f19;
            --panel-bg: #111827;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --border: #1f2937;
            --bubble-user: #312e81;
            --bubble-bot: #1f2937;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: var(--panel-bg);
            border-bottom: 1px solid var(--border);
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand h1 { font-size: 1.25rem; font-weight: 700; color: #e0e7ff; }
        .badge {
            background: rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .stats { font-size: 0.85rem; color: var(--text-muted); }
        .main-container {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        .sidebar {
            width: 320px;
            background-color: var(--panel-bg);
            border-right: 1px solid var(--border);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .upload-card {
            border: 2px dashed #374151;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background: rgba(17, 24, 39, 0.5);
            cursor: pointer;
            transition: all 0.2s;
        }
        .upload-card:hover { border-color: var(--accent); }
        .upload-card input { display: none; }
        .btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .btn:hover { background: var(--accent-hover); }
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--bg-color);
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 12px;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        .message.user {
            align-self: flex-end;
            background-color: var(--bubble-user);
            color: #ffffff;
            border-bottom-right-radius: 2px;
        }
        .message.assistant {
            align-self: flex-start;
            background-color: var(--bubble-bot);
            border: 1px solid var(--border);
            border-bottom-left-radius: 2px;
        }
        .input-bar {
            padding: 16px 24px;
            background-color: var(--panel-bg);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
        }
        .input-bar textarea {
            flex: 1;
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            color: white;
            padding: 12px 16px;
            font-size: 0.95rem;
            resize: none;
            height: 48px;
            outline: none;
            font-family: inherit;
        }
        .input-bar textarea:focus { border-color: var(--accent); }
        .status-msg { font-size: 0.8rem; color: #10b981; margin-top: 8px; }
        pre, code { background: #111827; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
        pre { padding: 12px; overflow-x: auto; margin: 8px 0; }
        table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }
        th, td { border: 1px solid #374151; padding: 8px 12px; text-align: left; }
        th { background: #1f2937; }

        /* Sources Card Styles */
        details.sources-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 14px;
            font-size: 0.88rem;
        }
        details.sources-card summary {
            cursor: pointer;
            color: #a5b4fc;
            font-weight: 600;
            user-select: none;
            outline: none;
        }
        details.sources-card summary:hover {
            color: #c7d2fe;
        }
        .sources-list {
            margin-top: 10px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .source-item {
            background: #1f2937;
            border-left: 3px solid #6366f1;
            border-radius: 4px;
            padding: 8px 12px;
        }
        .source-header {
            font-size: 0.82rem;
            color: #e0e7ff;
            margin-bottom: 4px;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 6px;
        }
        .page-badge {
            background: rgba(99, 102, 241, 0.25);
            color: #c7d2fe;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }
        .section-badge {
            background: #374151;
            color: #d1d5db;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.78rem;
        }
        .sim-badge {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: 600;
        }
        .source-snippet {
            margin: 4px 0 0 0;
            padding-left: 8px;
            border-left: 2px solid #4b5563;
            color: #9ca3af;
            font-size: 0.8rem;
            font-style: italic;
            line-height: 1.4;
        }

        /* Tabs */
        .tabs { display: flex; gap: 4px; background: #0b0f19; padding: 4px; border-radius: 8px; border: 1px solid var(--border); }
        .tab {
            background: transparent; color: var(--text-muted); border: none; padding: 6px 14px;
            border-radius: 6px; font-weight: 600; font-size: 0.85rem; cursor: pointer; font-family: inherit;
        }
        .tab:hover { color: var(--text-main); }
        .tab.active { background: var(--accent); color: white; }
        .hidden { display: none !important; }

        /* Gap Finder */
        .gap-area { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 18px; }
        .gap-intro h2 { font-size: 1.15rem; color: #e0e7ff; margin-bottom: 6px; }
        .gap-intro p { color: var(--text-muted); font-size: 0.9rem; line-height: 1.5; }
        .gap-form { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
        .gap-form input[type=text] {
            flex: 1; min-width: 240px; background: #1f2937; border: 1px solid #374151; border-radius: 8px;
            color: white; padding: 12px 16px; font-size: 0.95rem; outline: none; font-family: inherit;
        }
        .gap-form input[type=text]:focus { border-color: var(--accent); }
        .gap-form select, .gap-form label { color: var(--text-muted); font-size: 0.85rem; }
        .gap-form select { background: #1f2937; border: 1px solid #374151; border-radius: 6px; color: white; padding: 8px; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn.secondary { background: #374151; }
        .btn.secondary:hover { background: #4b5563; }
        .phase-list { list-style: none; background: var(--panel-bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; }
        .phase-list li { padding: 6px 0; font-size: 0.88rem; color: #6b7280; display: flex; gap: 10px; }
        .phase-list li .icon { width: 18px; flex-shrink: 0; text-align: center; }
        .phase-list li.running { color: #c7d2fe; }
        .phase-list li.done { color: #d1d5db; }
        .phase-list li .detail { display: block; color: var(--text-muted); font-size: 0.8rem; margin-top: 2px; }
        .phase-list li.running .icon::before { content: '◐'; display: inline-block; animation: spin 1s linear infinite; }
        .phase-list li.done .icon::before { content: '✓'; color: #10b981; }
        .phase-list li.pending .icon::before { content: '○'; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .gap-report {
            background: var(--bubble-bot); border: 1px solid var(--border); border-radius: 12px;
            padding: 22px 26px; line-height: 1.6; font-size: 0.93rem;
        }
        .gap-report h1 { font-size: 1.3rem; color: #e0e7ff; margin-bottom: 10px; }
        .gap-report h2 { font-size: 1.08rem; color: #c7d2fe; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #374151; }
        .gap-report h3 { font-size: 0.95rem; color: #e0e7ff; margin: 14px 0 6px; }
        .gap-report ul, .gap-report ol { padding-left: 22px; margin: 6px 0; }
        .gap-report li { margin: 4px 0; }
        .gap-report a { color: #a5b4fc; }
        .gap-report table { display: block; overflow-x: auto; font-size: 0.82rem; }
        .gap-actions { display: flex; gap: 10px; }
        .gap-error { color: #ef4444; font-size: 0.9rem; }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>🧪 Chem-RAG Agent</h1>
            <span class="badge">LangGraph + Parent-Doc</span>
        </div>
        <nav class="tabs" role="tablist">
            <button class="tab active" id="tabChat" onclick="switchTab('chat')">💬 Chat</button>
            <button class="tab" id="tabGap" onclick="switchTab('gap')">🔍 Gap Finder</button>
        </nav>
        <div class="stats" id="dbStats">Loading database metrics...</div>
    </header>

    <div class="main-container">
        <div class="sidebar">
            <div>
                <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px;">Upload Paper (PDF)</h3>
                <div class="upload-card" onclick="document.getElementById('fileInput').click()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#9ca3af" stroke-width="2" style="margin-bottom: 8px;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
                    <p style="font-size: 0.85rem; color: #d1d5db;">Click to browse PDF</p>
                    <p style="font-size: 0.75rem; color: #6b7280; margin-top: 4px;">Local Vision OCR + Parent Chunking</p>
                    <input type="file" id="fileInput" accept=".pdf" onchange="uploadPDF()">
                </div>
                <div id="uploadStatus" class="status-msg"></div>
            </div>

            <div>
                <h3 style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 8px;">Features</h3>
                <ul style="font-size: 0.85rem; color: #9ca3af; line-height: 1.8; list-style: none; padding-left: 0;">
                    <li>✓ <strong>Router:</strong> Auto-routes Paper vs PubChem</li>
                    <li>✓ <strong>Critique:</strong> Relevance grading & loop</li>
                    <li>✓ <strong>Parent-Doc:</strong> Full reaction mechanisms</li>
                    <li>✓ <strong>LaTeX Math:</strong> $\\mathrm{C}_{60}$, formulas & reactions</li>
                </ul>
            </div>
        </div>

        <div class="chat-area">
            <div class="messages" id="messages">
                <div class="message assistant">
                    Hello! I am your <strong>Agentic Chemistry Research Assistant</strong>. Ask me any question about your uploaded research papers (e.g. <em>Fullerenes: Chemistry and its Applications</em>) or general chemistry (molecular formulas, SMILES, PubChem compound data).
                </div>
            </div>

            <div class="input-bar">
                <textarea id="promptInput" placeholder="Ask a chemistry question (e.g. What is the structure and synthesis of fullerene C60?)..." onkeydown="handleKey(event)"></textarea>
                <button class="btn" id="sendBtn" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <div class="gap-area hidden" id="gapArea">
            <div class="gap-intro">
                <h2>Knowledge Gap Finder</h2>
                <p>Type a research topic. The agents search OpenAlex and your uploaded PDFs, read the abstracts into a Lit Gap Table, detect patterns, and report <strong>population</strong>, <strong>variable</strong>, <strong>method</strong> and <strong>contradiction</strong> gaps with candidate research questions. Runs on the local model, so expect 2&ndash;5 minutes.</p>
            </div>
            <div class="gap-form">
                <input type="text" id="gapTopic" placeholder="e.g. perovskite solar cell stability under humidity" maxlength="300" onkeydown="if (event.key === 'Enter') runGapReport()">
                <label>Papers
                    <select id="gapMax">
                        <option value="8">8 (fast)</option>
                        <option value="15" selected>15</option>
                        <option value="25">25 (slow)</option>
                    </select>
                </label>
                <label><input type="checkbox" id="gapLocal" checked> Include uploaded PDFs</label>
                <button class="btn" id="gapBtn" onclick="runGapReport()">Find Gaps</button>
            </div>
            <ul class="phase-list hidden" id="phaseList"></ul>
            <div class="gap-error hidden" id="gapError"></div>
            <div class="gap-actions hidden" id="gapActions">
                <button class="btn secondary" onclick="downloadReport()">⬇ Download .md</button>
                <button class="btn secondary" onclick="copyReport()" id="copyBtn">Copy Markdown</button>
            </div>
            <div class="gap-report hidden" id="gapReport"></div>
        </div>
    </div>

    <script>
        const messagesDiv = document.getElementById('messages');
        const promptInput = document.getElementById('promptInput');
        const sendBtn = document.getElementById('sendBtn');
        let chatHistory = [];

        async function updateStats() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('dbStats').innerHTML = 
                    `<strong>Vectors:</strong> ${data.child_vectors} child blocks &bull; <strong>Parents:</strong> ${data.parent_sections} sections &bull; <strong>Model:</strong> ${data.llm_model}`;
            } catch(e) {}
        }
        updateStats();

        function renderMath() {
            if (window.renderMathInElement) {
                renderMathInElement(document.getElementById('messages'), {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\\\(', right: '\\\\)', display: false},
                        {left: '\\\\[', right: '\\\\]', display: true}
                    ],
                    throwOnError: false
                });
            }
        }

        function handleKey(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        async function sendMessage() {
            const query = promptInput.value.trim();
            if (!query) return;

            promptInput.value = '';
            appendMessage('user', query);

            const botMsgDiv = appendMessage('assistant', '<span style="color:#9ca3af;">Agent analyzing question & routing context...</span>');
            sendBtn.disabled = true;

            try {
                const response = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        model: 'chem-rag-agent',
                        messages: [...chatHistory, {role: 'user', content: query}],
                        stream: true
                    })
                });

                if (!response.ok) throw new Error('Request failed with ' + response.status);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullAnswer = '';
                botMsgDiv.innerHTML = '';

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6).trim();
                            if (dataStr === '[DONE]') continue;
                            try {
                                const parsed = JSON.parse(dataStr);
                                const token = parsed.choices?.[0]?.delta?.content || '';
                                fullAnswer += token;
                                botMsgDiv.innerHTML = marked.parse(fullAnswer);
                                messagesDiv.scrollTop = messagesDiv.scrollHeight;
                            } catch (e) {}
                        }
                    }
                }

                chatHistory.push({role: 'user', content: query});
                chatHistory.push({role: 'assistant', content: fullAnswer});
                renderMath();

            } catch (err) {
                botMsgDiv.innerHTML = `<span style="color:#ef4444;">Error: ${err.message}</span>`;
            } finally {
                sendBtn.disabled = false;
                renderMath();
            }
        }

        function appendMessage(role, html) {
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = role === 'user' ? html : marked.parse(html);
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return div;
        }

        // ─── Tabs ─────────────────────────────────────────────────────────
        function switchTab(tab) {
            const isGap = tab === 'gap';
            document.querySelector('.chat-area').classList.toggle('hidden', isGap);
            document.getElementById('gapArea').classList.toggle('hidden', !isGap);
            document.getElementById('tabChat').classList.toggle('active', !isGap);
            document.getElementById('tabGap').classList.toggle('active', isGap);
            try { localStorage.setItem('chemrag-tab', tab); } catch (e) {}
            if (isGap) document.getElementById('gapTopic').focus();
        }
        try { if (localStorage.getItem('chemrag-tab') === 'gap') switchTab('gap'); } catch (e) {}

        // ─── Knowledge Gap Finder ─────────────────────────────────────────
        let gapMarkdown = '';
        let gapTopicUsed = '';

        function escapeHtml(s) {
            return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        function phaseItem(ev) {
            let li = document.getElementById('phase-' + ev.phase);
            if (!li) {
                li = document.createElement('li');
                li.id = 'phase-' + ev.phase;
                li.innerHTML = '<span class="icon"></span><span><span class="label"></span><span class="detail"></span></span>';
                document.getElementById('phaseList').appendChild(li);
            }
            li.querySelector('.label').textContent = ev.label;
            return li;
        }

        function handleGapEvent(ev) {
            const err = document.getElementById('gapError');
            if (ev.type === 'queued') {
                if (ev.position > 1) {
                    err.classList.remove('hidden');
                    err.style.color = '#a5b4fc';
                    err.textContent = 'Waiting for ' + (ev.position - 1) + ' other request(s) to finish on the local model...';
                }
            } else if (ev.type === 'progress') {
                err.classList.add('hidden');
                const li = phaseItem(ev);
                li.className = ev.status === 'done' ? 'done' : 'running';
                if (ev.detail) li.querySelector('.detail').textContent = ev.detail;
            } else if (ev.type === 'report') {
                gapMarkdown = ev.markdown;
                const box = document.getElementById('gapReport');
                // Escape raw HTML from paper titles / model output before rendering Markdown
                box.innerHTML = marked.parse(escapeHtml(gapMarkdown));
                box.querySelectorAll('a').forEach(a => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
                box.classList.remove('hidden');
                document.getElementById('gapActions').classList.remove('hidden');
                if (window.renderMathInElement) renderMathInElement(box, { throwOnError: false });
                box.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else if (ev.type === 'error') {
                err.classList.remove('hidden');
                err.style.color = '#ef4444';
                err.textContent = ev.message;
                document.querySelectorAll('#phaseList li.running').forEach(li => li.className = 'pending');
            }
        }

        async function runGapReport() {
            const topic = document.getElementById('gapTopic').value.trim();
            if (topic.length < 3) { document.getElementById('gapTopic').focus(); return; }

            const btn = document.getElementById('gapBtn');
            btn.disabled = true;
            btn.textContent = 'Running...';
            gapMarkdown = '';
            gapTopicUsed = topic;
            document.getElementById('phaseList').innerHTML = '';
            document.getElementById('phaseList').classList.remove('hidden');
            ['gapReport', 'gapActions', 'gapError'].forEach(id => document.getElementById(id).classList.add('hidden'));

            try {
                const res = await fetch('/v1/gap-report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        topic: topic,
                        max_papers: parseInt(document.getElementById('gapMax').value, 10),
                        include_local: document.getElementById('gapLocal').checked
                    })
                });
                if (!res.ok) {
                    let detail = 'Request failed with ' + res.status;
                    try { const j = await res.json(); if (j.detail) detail = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch (e) {}
                    throw new Error(detail);
                }

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    // SSE events are separated by a blank line; keep any partial event in the buffer
                    const parts = buffer.split('\\n\\n');
                    buffer = parts.pop();
                    for (const part of parts) {
                        if (!part.startsWith('data: ')) continue;  // skip keepalive comments
                        try { handleGapEvent(JSON.parse(part.slice(6))); } catch (e) {}
                    }
                }
                if (!gapMarkdown && document.getElementById('gapError').classList.contains('hidden')) {
                    handleGapEvent({ type: 'error', message: 'The connection closed before the report finished. Please try again.' });
                }
            } catch (err) {
                handleGapEvent({ type: 'error', message: 'Error: ' + err.message });
            } finally {
                btn.disabled = false;
                btn.textContent = 'Find Gaps';
            }
        }

        function downloadReport() {
            if (!gapMarkdown) return;
            const slug = gapTopicUsed.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60) || 'report';
            const blob = new Blob([gapMarkdown], { type: 'text/markdown' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'gap-report-' + slug + '.md';
            a.click();
            URL.revokeObjectURL(a.href);
        }

        async function copyReport() {
            if (!gapMarkdown) return;
            try {
                await navigator.clipboard.writeText(gapMarkdown);
                const b = document.getElementById('copyBtn');
                b.textContent = 'Copied ✓';
                setTimeout(() => b.textContent = 'Copy Markdown', 1500);
            } catch (e) {}
        }

        async function uploadPDF() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) return;

            const status = document.getElementById('uploadStatus');
            status.style.color = '#a5b4fc';
            status.innerText = `Ingesting ${file.name}... (Vision OCR + Parent Chunking)`;

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok) {
                    status.style.color = '#10b981';
                    status.innerText = `✓ Ingested: ${data.child_chunks_ingested} child vectors, ${data.parent_sections} parent sections!`;
                    updateStats();
                } else {
                    throw new Error(data.detail || 'Upload failed');
                }
            } catch (err) {
                status.style.color = '#ef4444';
                status.innerText = `✗ Failed: ${err.message}`;
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List models available to Open WebUI."""
    return ModelListResponse(
        data=[
            ModelData(id=MODEL_ID, created=int(time.time())),
            ModelData(id=LLM_MODEL, created=int(time.time())),
        ]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint running LangGraph agent.
    Safely enqueues requests via inference_lock for multi-user local stability.
    """
    global active_queue_count

    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in messages array.")

    user_query = user_messages[-1].content
    history = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]

    completion_id = _completion_id()
    created = int(time.time())

    # --- Streaming Response (Server-Sent Events) ---
    if request.stream:
        async def _stream_generator() -> AsyncGenerator[str, None]:
            global active_queue_count
            active_queue_count += 1
            print(f"[Queue] New streaming request entered queue (Queue depth: {active_queue_count})", flush=True)

            try:
                # Wait sequentially for inference lock
                async with inference_lock:
                    print(f"[Queue] Acquired inference lock for query: '{user_query[:50]}...'", flush=True)

                    # Stream tokens from LangGraph agent
                    for token in agent_graph.ask_stream(user_query=user_query, chat_history=history):
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_ID,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        # Cooperative yield to event loop
                        await asyncio.sleep(0.001)

                    # Final stop chunk
                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_ID,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            except Exception as exc:
                print(f"[!] Error during streaming inference: {exc}", flush=True)
                err_chunk = {"error": {"message": str(exc), "type": "agent_error"}}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                active_queue_count = max(0, active_queue_count - 1)
                print(f"[Queue] Released inference lock (Queue depth remaining: {active_queue_count})", flush=True)

        return StreamingResponse(
            _stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming Response ---
    active_queue_count += 1
    print(f"[Queue] Non-streaming request waiting in queue (Depth: {active_queue_count})", flush=True)
    try:
        async with inference_lock:
            print(f"[Queue] Acquired inference lock for non-streaming query.", flush=True)
            # Run LangGraph synchronous invocation in thread pool
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                None,
                agent_graph.ask,
                user_query,
                history,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(exc)}")
    finally:
        active_queue_count = max(0, active_queue_count - 1)
        print(f"[Queue] Released inference lock.", flush=True)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        choices=[Choice(message=ChoiceMessage(content=answer))],
    )


# ---------------------------------------------------------------------------
# Knowledge Gap Finder Endpoint (Multi-Agent Literature Review)
# ---------------------------------------------------------------------------
@app.post("/v1/gap-report")
async def gap_report(request: GapReportRequest):
    """
    Run the Knowledge Gap Finder for a research topic and stream Server-Sent Events:
      {"type": "queued"} -> {"type": "progress", ...}* -> {"type": "report", "markdown": ...}
    or {"type": "error", "message": ...}. Shares the inference lock with chat so the
    local model handles one job at a time.
    """
    topic = request.topic.strip()
    cancelled = threading.Event()

    async def _event_stream() -> AsyncGenerator[str, None]:
        global active_queue_count
        active_queue_count += 1
        yield f"data: {json.dumps({'type': 'queued', 'position': active_queue_count})}\n\n"
        try:
            async with inference_lock:
                print(f"[Queue] Gap report started for topic: '{topic[:60]}'", flush=True)
                loop = asyncio.get_running_loop()
                events: asyncio.Queue = asyncio.Queue()
                done = object()

                def _worker():
                    # Runs the blocking LLM pipeline off the event loop so the server stays responsive
                    try:
                        for ev in gap_graph.run_stream(
                            topic, max_papers=request.max_papers, include_local=request.include_local
                        ):
                            if cancelled.is_set():  # client disconnected: stop at the next step
                                print("[Queue] Gap report cancelled by client.", flush=True)
                                break
                            loop.call_soon_threadsafe(events.put_nowait, ev)
                    except Exception as exc:
                        print(f"[!] Gap report failed: {exc}", flush=True)
                        loop.call_soon_threadsafe(
                            events.put_nowait, {"type": "error", "message": f"Gap report failed: {exc}"}
                        )
                    finally:
                        loop.call_soon_threadsafe(events.put_nowait, done)

                worker = loop.run_in_executor(None, _worker)
                while True:
                    try:
                        ev = await asyncio.wait_for(events.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"  # keeps proxies/tunnels from closing idle streams
                        continue
                    if ev is done:
                        break
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                await worker
        finally:
            cancelled.set()
            active_queue_count = max(0, active_queue_count - 1)
            print(f"[Queue] Gap report finished (Queue depth remaining: {active_queue_count})", flush=True)

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Document Ingestion Endpoint (Local Vision + Parent-Document Chunking)
# ---------------------------------------------------------------------------
@app.post("/upload")
@app.post("/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest scientific PDF paper:
    1. Rasterizes pages and transcribes using local vision model (llama3.2-vision or qwen2.5-vl).
    2. Chunks using Parent-Document Retrieval (~100 token child chunks linked to full parent sections).
    3. Embeds child chunks with nomic-embed-text into ChromaDB and saves parent sections to docstore.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    save_path = DATA_DIR / file.filename
    try:
        content = await file.read()
        save_path.write_bytes(content)

        print(f"[Upload] Received '{file.filename}'. Starting local vision parsing...", flush=True)
        markdown = parse_pdf(str(save_path))

        if not markdown.strip():
            raise HTTPException(status_code=422, detail="No readable content could be extracted from the document.")

        source_id = Path(file.filename).stem
        parent_store, child_chunks = chunk_parent_child(markdown, source_id=source_id)

        if not child_chunks:
            raise HTTPException(status_code=422, detail="Failed to produce chunks from the parsed document.")

        ingested_count = store.ingest_parents_and_children(
            parent_store=parent_store,
            child_chunks=child_chunks,
            pdf_path=str(save_path),
        )

        return {
            "status": "success",
            "filename": file.filename,
            "parent_sections": len(parent_store),
            "child_chunks_ingested": ingested_count,
            "total_vectors": store.count,
            "message": f"Successfully ingested {file.filename} with Parent-Document retrieval.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
