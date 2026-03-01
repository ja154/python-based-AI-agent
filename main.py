import argparse
import logging
import os
import sys
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def redact(s: str, max_len: int = 1000) -> str:
    """Truncate a string for safe logging (avoid leaking sensitive data)."""
    if len(s) > max_len:
        return s[:max_len] + f"\n... ({len(s) - max_len} more characters)"
    return s


def main(argv: list | None = None) -> None:
    """Main entry point for the research assistant."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Research assistant that generates research papers using tools."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Research query (if not provided, prompt interactively)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging with full stack traces"
    )
    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()

    # Fail fast if ANTHROPIC_API_KEY is missing
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error(
            "Missing ANTHROPIC_API_KEY environment variable. "
            "Please set ANTHROPIC_API_KEY in your environment or .env file."
        )
        sys.exit(1)

    logger.debug("ANTHROPIC_API_KEY found in environment")

    # Initialize LLM and parser
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    parser_obj = PydanticOutputParser(pydantic_object=ResearchResponse)

    # Construct the prompt with tightened system instructions
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Return only valid JSON matching the schema. Wrap the output exactly in this format with no additional text:
{format_instructions}""",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser_obj.get_format_instructions())

    # Create agent
    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=args.debug)

    # Get query from CLI argument or interactive prompt
    if args.query:
        query = args.query
        logger.debug("Query from --query argument: %s", redact(query, 100))
    else:
        query = input("What can i help you research? ")
        logger.debug("Query from interactive input: %s", redact(query, 100))

    # Invoke the agent with error handling
    logger.info("Starting agent invocation...")
    try:
        raw_response = agent_executor.invoke({"query": query, "chat_history": []})
        logger.debug("Agent invocation successful")
    except Exception as e:
        if args.debug:
            logger.exception("Agent invocation failed with exception:")
        else:
            logger.error("Agent invocation failed: %s", str(e))
        sys.exit(2)

    # Verify response structure
    if not isinstance(raw_response, dict) or "output" not in raw_response:
        logger.error(
            "Agent response missing 'output' key. Response keys: %s",
            list(raw_response.keys()) if isinstance(raw_response, dict) else type(raw_response),
        )
        logger.debug("Full response: %s", redact(str(raw_response)))
        sys.exit(3)

    output_text = raw_response.get("output", "")

    # Parse the output with error handling
    logger.info("Parsing agent output...")
    try:
        structured_response = parser_obj.parse(output_text)
        logger.info("Successfully parsed research response")
        print(structured_response)
    except Exception as e:
        if args.debug:
            logger.exception("Failed to parse agent output:")
        else:
            logger.error("Failed to parse agent output: %s", str(e))
        logger.error("Raw output (first 1000 chars): %s", redact(output_text, 1000))
        sys.exit(3)


if __name__ == "__main__":
    main()