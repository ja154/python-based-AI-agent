# Python AI Agent (CLI + Local Web Chat)

Simple research agent built with LangChain tools and Anthropic.

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

Create `.env` from `sample.env` and set values:

```env
ANTHROPIC_API_KEY="your-key"
FLASK_SECRET_KEY="any-random-local-string"
```

`OPENAI_API_KEY` is optional and not used by default.

## 3) Run CLI Agent

Interactive:

```bash
python main.py
```

Single query:

```bash
python main.py --query "Research edge AI chip trends in 2026"
```

## 4) Run Browser Chat UI (Local)

```bash
python web_chat.py
```

Open: `http://127.0.0.1:5000`

## Notes

- Chat history is session-based in your browser.
- Browser chat streams response tokens live while the agent is generating.
- `research_output.txt` is ignored by git and can be written by the save tool.
