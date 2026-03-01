from __future__ import annotations

import io
import json
import os
import queue
import threading
from typing import Any

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    request,
    session,
    stream_with_context,
)
from langchain_core.callbacks import BaseCallbackHandler
from werkzeug.datastructures import FileStorage

from main import ResearchResponse, build_research_runtime, run_research

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "local-dev-secret")

MAX_UPLOAD_FILES = 6
MAX_SINGLE_DOC_CHARS = 12000
MAX_TOTAL_DOC_CHARS = 60000
MAX_UPLOAD_BYTES = 6 * 1024 * 1024
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".log",
    ".xml",
    ".html",
    ".htm",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rs",
    ".sql",
    ".ini",
    ".cfg",
    ".toml",
}

RUNTIME = None
RUNTIME_ERROR = None
try:
    RUNTIME = build_research_runtime(verbose=False)
except Exception as exc:
    RUNTIME_ERROR = str(exc)


class TokenQueueHandler(BaseCallbackHandler):
    def __init__(self, token_queue: queue.Queue):
        self.token_queue = token_queue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self.token_queue.put(token)


def format_response(response: ResearchResponse) -> str:
    lines = [f"Topic: {response.topic}", "", response.summary]
    if response.sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend([f"- {source}" for source in response.sources])
    if response.tools_used:
        lines.append("")
        lines.append("Tools Used:")
        lines.extend([f"- {tool}" for tool in response.tools_used])
    return "\n".join(lines)


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _safe_filename(filename: str) -> str:
    name = os.path.basename(str(filename or "").strip())
    return name[:180]


def _trim_text(value: str, max_chars: int) -> str:
    value = str(value or "").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... ({len(value) - max_chars} more characters)"


def _decode_text_blob(blob: bytes) -> str | None:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return blob.decode(encoding)
        except Exception:
            continue
    return None


def _extract_text_from_upload(upload: FileStorage) -> tuple[str | None, str | None]:
    filename = _safe_filename(upload.filename)
    if not filename:
        return None, "missing filename"

    blob = upload.read()
    if not blob:
        return None, "empty file"
    if len(blob) > MAX_UPLOAD_BYTES:
        return None, f"file too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"

    extension = os.path.splitext(filename)[1].lower()
    if extension in TEXT_EXTENSIONS:
        decoded = _decode_text_blob(blob)
        if decoded is None:
            return None, "could not decode text content"
        return _trim_text(decoded, MAX_SINGLE_DOC_CHARS), None

    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            return None, "PDF parsing requires pypdf"

        try:
            reader = PdfReader(io.BytesIO(blob))
            parts: list[str] = []
            for page in reader.pages[:25]:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text.strip())
            if not parts:
                return None, "no extractable PDF text found"
            return _trim_text("\n\n".join(parts), MAX_SINGLE_DOC_CHARS), None
        except Exception as exc:
            return None, f"PDF read error: {exc}"

    if extension == ".docx":
        try:
            from docx import Document
        except Exception:
            return None, "DOCX parsing requires python-docx"

        try:
            document = Document(io.BytesIO(blob))
            parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            if not parts:
                return None, "no extractable DOCX text found"
            return _trim_text("\n\n".join(parts), MAX_SINGLE_DOC_CHARS), None
        except Exception as exc:
            return None, f"DOCX read error: {exc}"

    decoded = _decode_text_blob(blob)
    if decoded is not None:
        return _trim_text(decoded, MAX_SINGLE_DOC_CHARS), None

    return None, f"unsupported file type: {extension or 'unknown'}"


def _merge_uploaded_docs(
    existing_docs: list[dict[str, str]] | None,
    uploads: list[FileStorage],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    docs_map: dict[str, str] = {}
    for item in existing_docs or []:
        name = _safe_filename(item.get("name", ""))
        text = str(item.get("text", ""))
        if name and text:
            docs_map[name] = text

    added_names: list[str] = []
    errors: list[str] = []

    for upload in uploads:
        name = _safe_filename(upload.filename)
        if not name:
            continue

        if len(docs_map) >= MAX_UPLOAD_FILES and name not in docs_map:
            errors.append(f"{name}: maximum of {MAX_UPLOAD_FILES} documents reached")
            continue

        text, error = _extract_text_from_upload(upload)
        if error:
            errors.append(f"{name}: {error}")
            continue

        docs_map[name] = text or ""
        if name not in added_names:
            added_names.append(name)

    updated_docs = [{"name": name, "text": text} for name, text in docs_map.items()]
    return updated_docs, added_names, errors


def _build_document_context(uploaded_docs: list[dict[str, str]] | None) -> tuple[str, list[str]]:
    docs = uploaded_docs or []
    if not docs:
        return "", []

    blocks: list[str] = []
    sources: list[str] = []
    total_chars = 0

    for item in docs:
        name = _safe_filename(item.get("name", ""))
        text = _trim_text(item.get("text", ""), MAX_SINGLE_DOC_CHARS)
        if not name or not text:
            continue

        block = f"[Document: {name}]\n{text}"
        if total_chars + len(block) > MAX_TOTAL_DOC_CHARS:
            remaining = MAX_TOTAL_DOC_CHARS - total_chars
            if remaining <= 0:
                break
            block = _trim_text(block, max_chars=remaining)
        blocks.append(block)
        total_chars += len(block)
        sources.append(f"uploaded:{name}")
        if total_chars >= MAX_TOTAL_DOC_CHARS:
            break

    return "\n\n".join(blocks), sources


def _parse_chat_payload() -> tuple[str, list[FileStorage]]:
    if request.is_json:
        data = request.get_json(silent=True) or {}
        return str(data.get("message", "")).strip(), []

    message = str(request.form.get("message", "")).strip()
    uploads = request.files.getlist("files")
    return message, uploads


def _default_query_for_docs() -> str:
    return "Analyze the uploaded documents and provide concise key findings."


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Jengo Research Chat</title>
  <style>
    :root {
      --bg: #ececec;
      --panel: #ffffff;
      --text: #1f1f1f;
      --subtle: #6b6b6b;
      --line: #e4e4e4;
      --accent: #10a37f;
      --accent-2: #0e8f6f;
      --bubble-user: #dff5ec;
      --bubble-assistant: #ffffff;
      --error: #b42318;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Soehne", "Segoe UI", -apple-system, sans-serif;
      color: var(--text);
      background: linear-gradient(130deg, #f5f5f5 0%, #ececec 60%, #e7efeb 100%);
      min-height: 100vh;
      padding: 10px;
    }
    .app-shell {
      width: min(1280px, 100%);
      height: calc(100vh - 20px);
      margin: 0 auto;
      display: grid;
      grid-template-columns: 270px 1fr;
      gap: 10px;
    }
    .sidebar {
      background: #171717;
      color: #f2f2f2;
      border-radius: 14px;
      padding: 14px 12px;
      display: grid;
      grid-template-rows: auto auto auto auto 1fr auto;
      gap: 12px;
      box-shadow: 0 12px 34px rgba(0, 0, 0, 0.22);
    }
    .brand {
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.2px;
      padding: 6px 8px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.03));
    }
    .sidebar button {
      width: 100%;
      border: 1px solid rgba(255, 255, 255, 0.18);
      background: transparent;
      color: #f2f2f2;
      border-radius: 10px;
      padding: 10px 12px;
      text-align: left;
      cursor: pointer;
      font-weight: 600;
      transition: background 120ms ease;
    }
    .sidebar button:hover {
      background: rgba(255, 255, 255, 0.08);
    }
    .history-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.9px;
      opacity: 0.72;
      padding: 0 8px;
    }
    .history-list {
      margin: 0;
      padding: 0;
      list-style: none;
      overflow: auto;
      display: grid;
      gap: 6px;
    }
    .history-item {
      font-size: 13px;
      color: #ebebeb;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 9px;
      padding: 8px 9px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .history-empty {
      font-size: 13px;
      opacity: 0.62;
      padding: 8px;
    }
    .sidebar-foot {
      font-size: 12px;
      opacity: 0.72;
      padding: 8px;
      border-top: 1px solid rgba(255, 255, 255, 0.12);
    }
    .chat-shell {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 14px 34px rgba(40, 40, 40, 0.14);
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .topbar {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #fafafa;
    }
    .topbar-title {
      margin: 0;
      font-weight: 700;
      font-size: 15px;
    }
    .topbar-sub {
      margin-top: 2px;
      font-size: 13px;
      color: var(--subtle);
    }
    .topbar button,
    .send-btn,
    .attach-btn {
      border: 0;
      background: var(--accent);
      color: #fff;
      padding: 9px 14px;
      border-radius: 11px;
      cursor: pointer;
      font-weight: 600;
    }
    .topbar button:hover,
    .send-btn:hover,
    .attach-btn:hover {
      background: var(--accent-2);
    }
    .topbar button:disabled,
    .send-btn:disabled,
    .attach-btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .messages {
      padding: 18px 22px;
      overflow-y: auto;
      background:
        radial-gradient(circle at 5% 0%, #f7f7f7 0%, transparent 50%),
        radial-gradient(circle at 100% 100%, #e9f4ef 0%, transparent 32%),
        #f9f9f9;
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .msg-row {
      display: flex;
      gap: 10px;
      align-items: flex-end;
      animation: rise 120ms ease-out;
    }
    .msg-row.user {
      justify-content: flex-end;
    }
    .avatar {
      width: 28px;
      height: 28px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-size: 12px;
      font-weight: 700;
      flex: 0 0 auto;
      border: 1px solid var(--line);
    }
    .avatar.assistant {
      background: #ededed;
      color: #222;
    }
    .avatar.user {
      background: #0f8f70;
      color: #fff;
      border-color: #0f8f70;
    }
    .bubble {
      max-width: min(780px, 88%);
      padding: 11px 13px;
      border-radius: 14px;
      white-space: pre-wrap;
      line-height: 1.48;
      border: 1px solid var(--line);
      background: var(--bubble-assistant);
      box-shadow: 0 2px 9px rgba(0, 0, 0, 0.03);
    }
    .bubble.user {
      background: var(--bubble-user);
      border-color: #b9dfd1;
    }
    .bubble.system {
      background: #fff5f5;
      border-color: #f3cccc;
      color: var(--error);
    }
    .composer-wrap {
      border-top: 1px solid var(--line);
      background: #fafafa;
      padding: 12px 16px 16px;
      display: grid;
      gap: 8px;
    }
    .composer-box {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 6px 7px 6px 8px;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.8);
    }
    textarea {
      width: 100%;
      min-height: 24px;
      max-height: 180px;
      border: 0;
      outline: none;
      resize: none;
      padding: 8px 0;
      font: inherit;
      background: #fff;
      color: var(--text);
    }
    .attach-btn {
      min-width: 70px;
      padding: 9px 10px;
      font-size: 12px;
    }
    .send-btn {
      width: 38px;
      height: 38px;
      border-radius: 11px;
      padding: 0;
      font-size: 18px;
      line-height: 1;
    }
    .composer-hint {
      font-size: 12px;
      color: var(--subtle);
      padding-left: 2px;
    }
    .docs-list {
      margin-top: 3px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .doc-chip {
      font-size: 11px;
      color: #13493f;
      background: #dff3ec;
      border: 1px solid #b7ddd0;
      border-radius: 999px;
      padding: 3px 8px;
      white-space: nowrap;
      max-width: 220px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (max-width: 900px) {
      .app-shell {
        grid-template-columns: 1fr;
        height: calc(100vh - 20px);
      }
      .sidebar {
        display: none;
      }
      .bubble {
        max-width: 92%;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">Jengo Research</div>
      <button id="sidebarNewBtn" type="button">+ New chat</button>
      <div class="history-title">Loaded Documents</div>
      <ul id="docsListSidebar" class="history-list">
        <li class="history-empty">No documents uploaded</li>
      </ul>
      <div class="history-title">Recent Prompts</div>
      <ul id="historyList" class="history-list">
        <li class="history-empty">No prompts yet</li>
      </ul>
      <div class="sidebar-foot">Jengo supports local Ollama and Anthropic.</div>
    </aside>

    <main class="chat-shell">
      <header class="topbar">
        <div>
          <h1 class="topbar-title">Jengo</h1>
          <div class="topbar-sub">AI research agent with web, Wikipedia, and document analysis</div>
        </div>
        <button id="resetBtn" type="button">New Chat</button>
      </header>

      <section id="messages" class="messages"></section>

      <form id="chatForm" class="composer-wrap">
        <div class="composer-box">
          <button id="attachBtn" class="attach-btn" type="button">Attach</button>
          <input id="fileInput" type="file" multiple hidden />
          <textarea id="messageInput" placeholder="Message Jengo" rows="1"></textarea>
          <button id="sendBtn" class="send-btn" type="submit" title="Send">↑</button>
        </div>
        <div class="composer-hint" id="docsHint">No documents queued.</div>
        <div class="docs-list" id="docsList"></div>
        <div class="composer-hint">Enter sends, Shift+Enter adds a new line.</div>
      </form>
    </main>
  </div>

  <script>
    const messagesEl = document.getElementById("messages");
    const formEl = document.getElementById("chatForm");
    const inputEl = document.getElementById("messageInput");
    const sendBtn = document.getElementById("sendBtn");
    const resetBtn = document.getElementById("resetBtn");
    const sidebarNewBtn = document.getElementById("sidebarNewBtn");
    const historyList = document.getElementById("historyList");
    const docsListSidebar = document.getElementById("docsListSidebar");
    const attachBtn = document.getElementById("attachBtn");
    const fileInput = document.getElementById("fileInput");
    const docsHint = document.getElementById("docsHint");
    const docsList = document.getElementById("docsList");

    let promptHistory = [];
    let loadedDocs = [];

    function addMessage(text, role) {
      const row = document.createElement("div");
      row.className = `msg-row ${role}`;

      const avatar = document.createElement("div");
      avatar.className = `avatar ${role === "system" ? "assistant" : role}`;
      avatar.textContent = role === "user" ? "U" : "J";

      const bubble = document.createElement("div");
      bubble.className = `bubble ${role}`;
      bubble.textContent = text;

      if (role === "user") {
        row.appendChild(bubble);
        row.appendChild(avatar);
      } else {
        row.appendChild(avatar);
        row.appendChild(bubble);
      }

      messagesEl.appendChild(row);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return {row, bubble};
    }

    function renderPromptHistory() {
      historyList.innerHTML = "";
      if (!promptHistory.length) {
        const li = document.createElement("li");
        li.className = "history-empty";
        li.textContent = "No prompts yet";
        historyList.appendChild(li);
        return;
      }
      promptHistory.forEach((item) => {
        const li = document.createElement("li");
        li.className = "history-item";
        li.textContent = item;
        historyList.appendChild(li);
      });
    }

    function renderDocs() {
      docsList.innerHTML = "";
      docsListSidebar.innerHTML = "";

      if (!loadedDocs.length) {
        docsHint.textContent = "No documents queued.";
        const sidebarEmpty = document.createElement("li");
        sidebarEmpty.className = "history-empty";
        sidebarEmpty.textContent = "No documents uploaded";
        docsListSidebar.appendChild(sidebarEmpty);
        return;
      }

      docsHint.textContent = `${loadedDocs.length} document(s) loaded for analysis in this chat.`;
      loadedDocs.forEach((name) => {
        const chip = document.createElement("div");
        chip.className = "doc-chip";
        chip.textContent = name;
        docsList.appendChild(chip);

        const li = document.createElement("li");
        li.className = "history-item";
        li.textContent = name;
        docsListSidebar.appendChild(li);
      });
    }

    function pushPromptHistory(message) {
      const oneLine = message.replace(/\\s+/g, " ").trim();
      if (!oneLine) return;
      promptHistory.unshift(oneLine);
      promptHistory = promptHistory.slice(0, 14);
      renderPromptHistory();
    }

    function autoResizeTextarea() {
      inputEl.style.height = "auto";
      inputEl.style.height = `${Math.min(inputEl.scrollHeight, 180)}px`;
    }

    function setBusy(isBusy) {
      sendBtn.disabled = isBusy;
      resetBtn.disabled = isBusy;
      sidebarNewBtn.disabled = isBusy;
      inputEl.disabled = isBusy;
      attachBtn.disabled = isBusy;
      fileInput.disabled = isBusy;
      if (!isBusy) inputEl.focus();
    }

    function buildMultipartPayload(message) {
      const formData = new FormData();
      formData.append("message", message);
      for (const file of fileInput.files) {
        formData.append("files", file);
      }
      return formData;
    }

    async function streamMessage(message, targetBubble) {
      const payload = buildMultipartPayload(message);
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          "Accept": "text/event-stream"
        },
        body: payload
      });

      if (!res.ok) {
        let data = {};
        try {
          data = await res.json();
        } catch (_) {}
        throw new Error(data.error || "Request failed.");
      }

      if (!res.body) {
        throw new Error("This browser does not support streaming responses.");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamedText = "";

      function processEventBlock(block) {
        const lines = block.split(/\\r?\\n/);
        let eventName = "message";
        let rawData = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            rawData += line.slice(5).trim();
          }
        }

        if (!rawData) return;
        const payload = JSON.parse(rawData);

        if (eventName === "token") {
          streamedText += payload.token || "";
          targetBubble.textContent = streamedText || "...";
          messagesEl.scrollTop = messagesEl.scrollHeight;
          return;
        }

        if (eventName === "done") {
          targetBubble.textContent = payload.reply || streamedText || "(No response)";
          messagesEl.scrollTop = messagesEl.scrollHeight;
          loadedDocs = payload.docs || loadedDocs;
          renderDocs();
          fileInput.value = "";

          if (payload.upload_errors && payload.upload_errors.length) {
            addMessage(`Upload warnings:\\n- ${payload.upload_errors.join("\\n- ")}`, "system");
          }
          return;
        }

        if (eventName === "error") {
          throw new Error(payload.message || "Streaming failed.");
        }
      }

      while (true) {
        const {value, done} = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, {stream: true});
        let splitIndex = buffer.indexOf("\\n\\n");
        while (splitIndex !== -1) {
          const block = buffer.slice(0, splitIndex).trim();
          buffer = buffer.slice(splitIndex + 2);
          if (block) processEventBlock(block);
          splitIndex = buffer.indexOf("\\n\\n");
        }
      }
    }

    formEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = inputEl.value.trim();
      const hasFiles = fileInput.files && fileInput.files.length > 0;
      if (!message && !hasFiles) return;

      if (message) {
        pushPromptHistory(message);
        addMessage(message, "user");
      } else {
        addMessage("[Analyze uploaded documents]", "user");
      }

      inputEl.value = "";
      autoResizeTextarea();
      const assistantMessage = addMessage("Thinking...", "assistant");
      setBusy(true);

      try {
        await streamMessage(message, assistantMessage.bubble);
      } catch (err) {
        assistantMessage.row.remove();
        addMessage(err.message, "system");
      } finally {
        setBusy(false);
      }
    });

    inputEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        formEl.requestSubmit();
      }
    });

    inputEl.addEventListener("input", autoResizeTextarea);

    attachBtn.addEventListener("click", () => {
      fileInput.click();
    });

    fileInput.addEventListener("change", () => {
      const selectedNames = Array.from(fileInput.files || []).map((f) => f.name);
      if (!selectedNames.length) {
        docsHint.textContent = loadedDocs.length
          ? `${loadedDocs.length} document(s) loaded for analysis in this chat.`
          : "No documents queued.";
        return;
      }
      docsHint.textContent = `Queued for upload: ${selectedNames.join(", ")}`;
    });

    async function resetChat() {
      setBusy(true);
      try {
        const res = await fetch("/api/reset", {method: "POST"});
        if (!res.ok) throw new Error("Could not reset chat.");
        messagesEl.innerHTML = "";
        promptHistory = [];
        loadedDocs = [];
        fileInput.value = "";
        renderPromptHistory();
        renderDocs();
        addMessage("New chat started. Upload docs or ask a research question.", "assistant");
      } catch (err) {
        addMessage(err.message, "system");
      } finally {
        setBusy(false);
      }
    }

    resetBtn.addEventListener("click", resetChat);
    sidebarNewBtn.addEventListener("click", resetChat);

    renderPromptHistory();
    renderDocs();
    addMessage("Hi, I am Jengo. I can research topics and analyze uploaded documents. Which sector or area do you want help with?", "assistant");
    autoResizeTextarea();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML_PAGE)


@app.post("/api/chat")
def chat():
    if RUNTIME_ERROR:
        return jsonify({"error": RUNTIME_ERROR}), 500

    user_message, uploads = _parse_chat_payload()
    history = session.get("chat_history", [])
    uploaded_docs = session.get("uploaded_docs", [])

    uploaded_docs, _, upload_errors = _merge_uploaded_docs(uploaded_docs, uploads)
    session["uploaded_docs"] = uploaded_docs
    session.modified = True

    document_context, document_sources = _build_document_context(uploaded_docs)
    effective_query = user_message if user_message else (_default_query_for_docs() if document_context else "")
    if not effective_query:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        response = run_research(
            query=effective_query,
            chat_history=history,
            runtime=RUNTIME,
            document_context=document_context,
            document_sources=document_sources,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    assistant_reply = format_response(response)
    user_echo = user_message if user_message else "[Analyze uploaded documents]"

    history.append({"role": "user", "content": user_echo})
    history.append({"role": "assistant", "content": assistant_reply})
    session["chat_history"] = history
    session.modified = True

    return jsonify(
        {
            "reply": assistant_reply,
            "structured": response.model_dump(),
            "docs": [doc["name"] for doc in uploaded_docs],
            "upload_errors": upload_errors,
        }
    )


@app.post("/api/chat/stream")
def chat_stream():
    if RUNTIME_ERROR:
        return jsonify({"error": RUNTIME_ERROR}), 500

    user_message, uploads = _parse_chat_payload()
    history = session.get("chat_history", [])
    uploaded_docs = session.get("uploaded_docs", [])

    uploaded_docs, _, upload_errors = _merge_uploaded_docs(uploaded_docs, uploads)
    session["uploaded_docs"] = uploaded_docs
    session.modified = True

    document_context, document_sources = _build_document_context(uploaded_docs)
    effective_query = user_message if user_message else (_default_query_for_docs() if document_context else "")
    if not effective_query:
        return jsonify({"error": "Message cannot be empty."}), 400

    token_queue: queue.Queue = queue.Queue()
    result_holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            response = run_research(
                query=effective_query,
                chat_history=history,
                runtime=RUNTIME,
                callbacks=[TokenQueueHandler(token_queue)],
                document_context=document_context,
                document_sources=document_sources,
            )
            result_holder["response"] = response
        except Exception as exc:
            result_holder["error"] = str(exc)
        finally:
            token_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    @stream_with_context
    def event_stream():
        while True:
            token = token_queue.get()
            if token is None:
                break
            yield sse_event("token", {"token": token})

        if "error" in result_holder:
            yield sse_event("error", {"message": str(result_holder["error"])})
            return

        response = result_holder.get("response")
        if not isinstance(response, ResearchResponse):
            yield sse_event("error", {"message": "No valid response from agent."})
            return

        assistant_reply = format_response(response)
        user_echo = user_message if user_message else "[Analyze uploaded documents]"
        history.append({"role": "user", "content": user_echo})
        history.append({"role": "assistant", "content": assistant_reply})
        session["chat_history"] = history
        session["uploaded_docs"] = uploaded_docs
        session.modified = True

        yield sse_event(
            "done",
            {
                "reply": assistant_reply,
                "structured": response.model_dump(),
                "docs": [doc["name"] for doc in uploaded_docs],
                "upload_errors": upload_errors,
            },
        )

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset():
    session["chat_history"] = []
    session["uploaded_docs"] = []
    session.modified = True
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
