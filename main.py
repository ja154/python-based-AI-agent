from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tools import (
    extract_urls,
    save_tool,
    search_tool,
    web_search_text,
    wiki_lookup_text,
    wiki_tool,
)

DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b-instruct"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
SUPPORTED_PROVIDERS = {"anthropic", "ollama"}
AGENT_NAME = "Jengo"
AGENT_DESCRIPTION = (
    "an AI research agent in this project that combines local or cloud LLMs with "
    "web/Wikipedia lookup and structured output."
)
AGENT_CAPABILITIES = [
    "Research and summarize a topic into concise notes.",
    "Fetch external context from web search and Wikipedia.",
    "Analyze user-provided text and uploaded documents as part of research.",
    "Return structured output with topic, summary, sources, and tools used.",
    "Support both local Ollama models and online Anthropic models.",
    "Help break down tasks, compare options, and suggest practical next steps.",
]
PROFILE_QUERY_HINTS = [
    "who are you",
    "what are you",
    "your name",
    "your capabilities",
    "what can you do",
    "how can you help",
    "how to use you",
    "your use",
    "use case",
    "what do you do",
]
SMALLTALK_HINTS = [
    "hello",
    "hi",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "what's up",
    "whats up",
]


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


@dataclass
class ResearchRuntime:
    provider: str
    parser: PydanticOutputParser
    llm: Any
    agent_executor: AgentExecutor | None = None


def redact(text: str, max_len: int = 1000) -> str:
    if len(text) > max_len:
        return text[:max_len] + f"\n... ({len(text) - max_len} more characters)"
    return text


def _capabilities_text() -> str:
    lines = [f"{idx}. {item}" for idx, item in enumerate(AGENT_CAPABILITIES, start=1)]
    return "\n".join(lines)


def _is_profile_query(query: str) -> bool:
    lowered = query.lower()
    return any(hint in lowered for hint in PROFILE_QUERY_HINTS)


def _build_profile_response() -> ResearchResponse:
    summary = (
        f"I am {AGENT_NAME}, {AGENT_DESCRIPTION}\n\n"
        "I can help with:\n"
        f"{_capabilities_text()}\n\n"
        "Which sector or area do you want help with right now? "
        "Examples: software engineering, business, education, healthcare, finance, operations, or policy."
    )
    return ResearchResponse(
        topic="Agent Identity and Capabilities",
        summary=summary,
        sources=[],
        tools_used=["self_profile"],
    )


def _is_smalltalk_query(query: str) -> bool:
    lowered = query.lower().strip()
    if not lowered:
        return False
    return any(hint in lowered for hint in SMALLTALK_HINTS)


def _build_smalltalk_response() -> ResearchResponse:
    summary = (
        "Hi, I am Jengo. I can help you research topics or analyze uploaded documents.\n\n"
        "Tell me what you want to work on and the sector/area (for example: software engineering, "
        "business, education, healthcare, finance, operations, or policy)."
    )
    return ResearchResponse(
        topic="Greeting",
        summary=summary,
        sources=[],
        tools_used=["smalltalk_router"],
    )


def _provider_from_env() -> str:
    provider = os.getenv("MODEL_PROVIDER", "anthropic").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise RuntimeError(
            f"Unsupported MODEL_PROVIDER='{provider}'. Use one of: {sorted(SUPPORTED_PROVIDERS)}"
        )
    return provider


def _validate_environment(provider: str) -> None:
    if provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY. Set it in your environment or .env for Anthropic."
            )
        return

    if provider == "ollama":
        return

    raise RuntimeError(f"Unsupported provider: {provider}")


def _build_llm(provider: str) -> Any:
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "langchain-anthropic is not installed. Install requirements and retry."
            ) from exc

        anthropic_model = os.getenv("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        return ChatAnthropic(model=anthropic_model, streaming=True)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError(
                "langchain-ollama is not installed. Add it to requirements and install dependencies."
            ) from exc

        ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            streaming=True,
            temperature=0.2,
        )

    raise RuntimeError(f"Unsupported provider: {provider}")


def _build_agent_prompt(parser_obj: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a research assistant that helps generate research notes.
Identity: You are Jengo, an AI research agent.
Core capabilities:
1) Research and summarize topics.
2) Use tools for web and Wikipedia lookups.
3) Analyze uploaded user documents and text context.
4) Return structured output.
5) Work with local Ollama or online Anthropic backends.
Answer the user query and use tools when needed.
If the user asks who you are, what your capabilities are, or how to use you,
explain your identity and capabilities clearly and ask which sector/area they want help with.
Return only valid JSON matching this schema and no additional text:
{format_instructions}""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser_obj.get_format_instructions())


def _build_ollama_prompt(query: str, context_text: str, parser_obj: PydanticOutputParser) -> str:
    return f"""You are a focused research assistant.
Identity: You are Jengo, an AI research agent.
Core capabilities:
1) Research and summarize topics.
2) Use web and Wikipedia findings as context.
3) Analyze uploaded user documents and text context.
4) Return structured output.
5) Help users define next steps.
If the user asks who you are, what your capabilities are, or how to use you,
explain your identity/capabilities and ask which sector or area they want help with.
Use the provided context to answer the query.
Keep the summary concise and factual.
If context is weak or conflicting, mention that uncertainty in the summary.

Return only JSON. Do not include markdown or code fences.
Schema:
{parser_obj.get_format_instructions()}

Query:
{query}

Context:
{context_text}
"""


def _to_langchain_messages(chat_history: list[Any] | None) -> list[BaseMessage]:
    if not chat_history:
        return []

    messages: list[BaseMessage] = []
    for item in chat_history:
        if isinstance(item, BaseMessage):
            messages.append(item)
            continue

        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", ""))
        elif isinstance(item, tuple) and len(item) == 2:
            role = str(item[0]).strip().lower()
            content = str(item[1])
        else:
            continue

        if role in {"user", "human"}:
            messages.append(HumanMessage(content=content))
        elif role in {"assistant", "ai"}:
            messages.append(AIMessage(content=content))
    return messages


def _text_from_llm_response(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "content" in part:
                    parts.append(str(part.get("content", "")))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "\n".join([p for p in parts if p]).strip()

    return str(content)


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for idx, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _parse_structured_response(
    output_text: str,
    parser_obj: PydanticOutputParser,
    default_topic: str,
    default_sources: list[str],
    default_tools: list[str],
) -> ResearchResponse:
    cleaned = str(output_text).strip()
    candidates = [cleaned]

    json_candidate = _extract_json_object(cleaned)
    if json_candidate and json_candidate not in candidates:
        candidates.append(json_candidate)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return parser_obj.parse(candidate)
        except Exception:
            pass
        try:
            loaded = json.loads(candidate)
            return ResearchResponse.model_validate(loaded)
        except Exception:
            pass

    fallback_summary = cleaned if cleaned else "No model output was returned."
    if len(fallback_summary) > 3000:
        fallback_summary = fallback_summary[:3000] + f"\n... ({len(fallback_summary) - 3000} more characters)"

    return ResearchResponse(
        topic=default_topic[:180],
        summary=fallback_summary,
        sources=default_sources[:8],
        tools_used=default_tools[:8],
    )


def _collect_context_for_local_model(query: str) -> tuple[str, list[str], list[str]]:
    context_parts: list[str] = []
    sources: list[str] = []
    tools_used: list[str] = []

    search_text = web_search_text(query)
    if search_text:
        context_parts.append(f"Web findings:\n{search_text}")
        tools_used.append("search_web")
        sources.extend(extract_urls(search_text))

    wiki_text = wiki_lookup_text(query)
    if wiki_text:
        context_parts.append(f"Wikipedia findings:\n{wiki_text}")
        tools_used.append("search_wikipedia")
        sources.extend(extract_urls(wiki_text))
        if not extract_urls(wiki_text):
            sources.append("wikipedia")

    if not context_parts:
        context_parts.append("No external context could be retrieved.")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_sources: list[str] = []
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        unique_sources.append(source)

    return "\n\n".join(context_parts), unique_sources[:8], tools_used


def _append_document_context_to_query(query: str, document_context: str | None) -> str:
    if not document_context:
        return query
    return (
        f"{query}\n\n"
        "Additional uploaded document context from the user:\n"
        f"{document_context}"
    )


def build_research_runtime(verbose: bool = False) -> ResearchRuntime:
    load_dotenv()
    provider = _provider_from_env()
    _validate_environment(provider)
    llm = _build_llm(provider)
    parser_obj = PydanticOutputParser(pydantic_object=ResearchResponse)

    if provider == "anthropic":
        prompt = _build_agent_prompt(parser_obj)
        tools = [search_tool, wiki_tool, save_tool]
        agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
        return ResearchRuntime(
            provider=provider,
            parser=parser_obj,
            llm=llm,
            agent_executor=agent_executor,
        )

    return ResearchRuntime(provider=provider, parser=parser_obj, llm=llm, agent_executor=None)


def _run_anthropic_research(
    query: str,
    chat_history: list[Any] | None,
    runtime: ResearchRuntime,
    callbacks: list[Any] | None,
    document_context: str | None,
    document_sources: list[str] | None,
) -> ResearchResponse:
    if runtime.agent_executor is None:
        raise ValueError("Agent executor is not configured for Anthropic runtime.")

    effective_query = _append_document_context_to_query(query.strip(), document_context)
    payload = {
        "query": effective_query,
        "chat_history": _to_langchain_messages(chat_history),
    }
    if callbacks:
        raw_response = runtime.agent_executor.invoke(payload, config={"callbacks": callbacks})
    else:
        raw_response = runtime.agent_executor.invoke(payload)

    if not isinstance(raw_response, dict) or "output" not in raw_response:
        raise ValueError(f"Agent response missing output field: {raw_response}")

    output_text = raw_response.get("output", "")
    if not isinstance(output_text, str):
        output_text = str(output_text)

    return _parse_structured_response(
        output_text=output_text,
        parser_obj=runtime.parser,
        default_topic=query,
        default_sources=(document_sources or [])[:8],
        default_tools=["uploaded_docs"] if document_context else [],
    )


def _run_ollama_research(
    query: str,
    runtime: ResearchRuntime,
    callbacks: list[Any] | None,
    document_context: str | None,
    document_sources: list[str] | None,
) -> ResearchResponse:
    context_text, context_sources, tools_used = _collect_context_for_local_model(query)
    if document_context:
        context_text = (
            f"Uploaded user documents:\n{document_context}\n\n"
            f"{context_text}"
        )
        tools_used.append("uploaded_docs")
        if document_sources:
            context_sources.extend(document_sources)

    seen_sources: set[str] = set()
    dedup_sources: list[str] = []
    for source in context_sources:
        if source in seen_sources:
            continue
        seen_sources.add(source)
        dedup_sources.append(source)
    context_sources = dedup_sources[:8]
    prompt_text = _build_ollama_prompt(query=query, context_text=context_text, parser_obj=runtime.parser)

    if callbacks:
        response = runtime.llm.invoke(prompt_text, config={"callbacks": callbacks})
    else:
        response = runtime.llm.invoke(prompt_text)

    output_text = _text_from_llm_response(response)
    parsed = _parse_structured_response(
        output_text=output_text,
        parser_obj=runtime.parser,
        default_topic=query,
        default_sources=context_sources,
        default_tools=tools_used,
    )

    if not parsed.sources and context_sources:
        parsed.sources = context_sources
    if not parsed.tools_used and tools_used:
        parsed.tools_used = tools_used
    return parsed


def run_research(
    query: str,
    chat_history: list[Any] | None = None,
    runtime: ResearchRuntime | None = None,
    callbacks: list[Any] | None = None,
    document_context: str | None = None,
    document_sources: list[str] | None = None,
) -> ResearchResponse:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    cleaned_query = query.strip()

    # Deterministic identity/capability reply so the agent stays consistent.
    if _is_profile_query(cleaned_query):
        return _build_profile_response()
    if _is_smalltalk_query(cleaned_query):
        return _build_smalltalk_response()

    active_runtime = runtime if runtime is not None else build_research_runtime()

    if active_runtime.provider == "anthropic":
        return _run_anthropic_research(
            query=cleaned_query,
            chat_history=chat_history,
            runtime=active_runtime,
            callbacks=callbacks,
            document_context=document_context,
            document_sources=document_sources,
        )

    return _run_ollama_research(
        query=cleaned_query,
        runtime=active_runtime,
        callbacks=callbacks,
        document_context=document_context,
        document_sources=document_sources,
    )


def main(argv: list[str] | None = None) -> None:
    cli = argparse.ArgumentParser(description="Run the research agent from the command line.")
    cli.add_argument("--query", type=str, default=None, help="Research query text")
    cli.add_argument("--debug", action="store_true", help="Enable verbose logs and stack traces")
    args = cli.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        runtime = build_research_runtime(verbose=args.debug)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)

    query = args.query if args.query else input("What can I help you research? ")

    try:
        response = run_research(query=query, chat_history=[], runtime=runtime)
        print(response.model_dump_json(indent=2))
    except Exception as exc:
        if args.debug:
            logger.exception("Agent request failed:")
        else:
            logger.error("Agent request failed: %s", redact(str(exc)))
        sys.exit(2)


if __name__ == "__main__":
    main()
