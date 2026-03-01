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

from main import ResearchResponse, build_research_runtime, run_research

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "local-dev-secret")

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


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Local Research Agent Chat</title>
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
      grid-template-rows: auto auto auto 1fr auto;
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
    .send-btn {
      border: 0;
      background: var(--accent);
      color: #fff;
      padding: 9px 14px;
      border-radius: 11px;
      cursor: pointer;
      font-weight: 600;
    }
    .topbar button:hover,
    .send-btn:hover {
      background: var(--accent-2);
    }
    .topbar button:disabled,
    .send-btn:disabled {
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
      grid-template-columns: 1fr auto;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 6px 7px 6px 12px;
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
      <div class="brand">LocalGPT Research</div>
      <button id="sidebarNewBtn" type="button">+ New chat</button>
      <div class="history-title">Recent Prompts</div>
      <ul id="historyList" class="history-list">
        <li class="history-empty">No prompts yet</li>
      </ul>
      <div class="sidebar-foot">Model: Claude 3.5 Sonnet</div>
    </aside>

    <main class="chat-shell">
      <header class="topbar">
        <div>
          <h1 class="topbar-title">Research Assistant</h1>
          <div class="topbar-sub">Web + Wikipedia + file save tools enabled</div>
        </div>
        <button id="resetBtn" type="button">New Chat</button>
      </header>

      <section id="messages" class="messages"></section>

      <form id="chatForm" class="composer-wrap">
        <div class="composer-box">
          <textarea id="messageInput" placeholder="Message Research Assistant" rows="1" required></textarea>
          <button id="sendBtn" class="send-btn" type="submit" title="Send">↑</button>
        </div>
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
    let promptHistory = [];

    function addMessage(text, role) {
      const row = document.createElement("div");
      row.className = `msg-row ${role}`;

      const avatar = document.createElement("div");
      avatar.className = `avatar ${role === "system" ? "assistant" : role}`;
      avatar.textContent = role === "user" ? "U" : "AI";

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
      if (!isBusy) inputEl.focus();
    }

    async function streamMessage(message, targetBubble) {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream"
        },
        body: JSON.stringify({message})
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
      if (!message) return;

      pushPromptHistory(message);
      addMessage(message, "user");
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

    async function resetChat() {
      setBusy(true);
      try {
        const res = await fetch("/api/reset", {method: "POST"});
        if (!res.ok) throw new Error("Could not reset chat.");
        messagesEl.innerHTML = "";
        promptHistory = [];
        renderPromptHistory();
        addMessage("New chat started. Ask another research question.", "assistant");
      } catch (err) {
        addMessage(err.message, "system");
      } finally {
        setBusy(false);
      }
    }

    resetBtn.addEventListener("click", resetChat);
    sidebarNewBtn.addEventListener("click", resetChat);

    renderPromptHistory();
    addMessage("Hi. Ask me anything to research, and I will stream the reply as it is generated.", "assistant");
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

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = session.get("chat_history", [])
    try:
        response = run_research(query=user_message, chat_history=history, runtime=RUNTIME)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    assistant_reply = format_response(response)

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": assistant_reply})
    session["chat_history"] = history
    session.modified = True

    return jsonify({"reply": assistant_reply, "structured": response.model_dump()})


@app.post("/api/chat/stream")
def chat_stream():
    if RUNTIME_ERROR:
        return jsonify({"error": RUNTIME_ERROR}), 500

    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = session.get("chat_history", [])
    token_queue: queue.Queue = queue.Queue()
    result_holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            response = run_research(
                query=user_message,
                chat_history=history,
                runtime=RUNTIME,
                callbacks=[TokenQueueHandler(token_queue)],
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
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_reply})
        session["chat_history"] = history
        session.modified = True

        yield sse_event(
            "done",
            {"reply": assistant_reply, "structured": response.model_dump()},
        )

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset():
    session["chat_history"] = []
    session.modified = True
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
