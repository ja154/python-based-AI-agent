import argparse
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tools import save_tool, search_tool, wiki_tool


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def redact(text: str, max_len: int = 1000) -> str:
    if len(text) > max_len:
        return text[:max_len] + f"\n... ({len(text) - max_len} more characters)"
    return text


def _validate_environment() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY. Set it in your environment or .env before running."
        )


def _build_prompt(parser_obj: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a research assistant that helps generate research notes.
Answer the user query and use tools when needed.
Return only valid JSON matching this schema and no additional text:
{format_instructions}""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser_obj.get_format_instructions())


def build_research_runtime(verbose: bool = False) -> tuple[PydanticOutputParser, AgentExecutor]:
    load_dotenv()
    _validate_environment()

    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", streaming=True)
    parser_obj = PydanticOutputParser(pydantic_object=ResearchResponse)
    prompt = _build_prompt(parser_obj)

    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=verbose)
    return parser_obj, agent_executor


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


def run_research(
    query: str,
    chat_history: list[Any] | None = None,
    runtime: tuple[PydanticOutputParser, AgentExecutor] | None = None,
    callbacks: list[Any] | None = None,
) -> ResearchResponse:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    parser_obj, agent_executor = runtime if runtime is not None else build_research_runtime()

    payload = {
        "query": query.strip(),
        "chat_history": _to_langchain_messages(chat_history),
    }
    if callbacks:
        raw_response = agent_executor.invoke(payload, config={"callbacks": callbacks})
    else:
        raw_response = agent_executor.invoke(payload)

    if not isinstance(raw_response, dict) or "output" not in raw_response:
        raise ValueError(f"Agent response missing output field: {raw_response}")

    output_text = raw_response.get("output", "")
    if not isinstance(output_text, str):
        output_text = str(output_text)

    return parser_obj.parse(output_text)


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
