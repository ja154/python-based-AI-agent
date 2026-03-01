# Jengo AI Research Agent (CLI + Local Web Chat)

Jengo is an AI research agent with LangChain tools and a switchable model provider:

- `anthropic` (online API)
- `ollama` (local model)

## 1) Setup

```bash
cd /home/jay/Desktop/python-based-AI-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `python3 -m venv` fails on Ubuntu/Debian:

```bash
sudo apt-get install -y python3.12-venv
```

## 2) Environment Variables

Create `.env` from `sample.env` and set values.

### Anthropic mode

```env
MODEL_PROVIDER="anthropic"
ANTHROPIC_API_KEY="your-key"
ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"
```

### Ollama mode

```env
MODEL_PROVIDER="ollama"
OLLAMA_MODEL="qwen2.5:3b-instruct"
OLLAMA_BASE_URL="http://127.0.0.1:11434"
```

Keep in `.env` for web session storage:

```env
FLASK_SECRET_KEY="any-random-local-string"
```

## 3) Ollama Local Model Setup

```bash
ollama pull qwen2.5:3b-instruct
ollama run qwen2.5:3b-instruct "hello"
```

## 4) Run CLI Agent

Interactive:

```bash
python main.py
```

Single query:

```bash
python main.py --query "Research edge AI chip trends in 2026"
```

## 5) Run Browser Chat UI (Local)

```bash
python web_chat.py
```

Open: `http://127.0.0.1:5000`

## 6) Upload Documents For Analysis

In the web chat UI:

1. Click `Attach`
2. Select one or more files
3. Ask a question, or send an empty message to analyze the uploaded docs directly

Supported formats:

- Text/code/config files (`.txt`, `.md`, `.csv`, `.json`, `.yaml`, `.py`, etc.)
- PDF (`.pdf`)
- Word (`.docx`)

## Notes

- Provider is selected by `MODEL_PROVIDER` in `.env`.
- Jengo has a fixed identity/personality and can explain capabilities on demand.
- Chat history is session-based in your browser.
- Uploaded docs are stored per chat session and cleared on `New Chat`.
- Browser chat streams response tokens live while the agent is generating.
- `research_output.txt` is ignored by git and can be written by the save tool.
