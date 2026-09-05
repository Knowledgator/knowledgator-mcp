import sys
import argparse

from mcp.server.fastmcp import FastMCP

from knowledgator_mcp.tools import inference_tools
from knowledgator_mcp.client import close_kg_client
from knowledgator_mcp.config import settings

mcp = FastMCP(
    name=settings.server_name,
    debug=settings.debug,
)


# Register Inference Tools
mcp.tool()(inference_tools.instruct_inference)
mcp.tool()(inference_tools.extract_inference)
mcp.tool()(inference_tools.comprehend_inference)


@mcp.resource("config://app")
async def get_app_config() -> str:
    """Provides application configuration overview as an MCP resource."""
    return (
        f"Server Name: {settings.server_name}\n"
        f"Debug Mode: {settings.debug}\n"
        f"Auth Type: {settings.auth_type}\n"
        f"Base URL: {settings.base_url}"
    )


def main() -> None:
    """CLI entrypoint for running the Knowledgator FastMCP server."""
    parser = argparse.ArgumentParser(description="Run Knowledgator FastMCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type to run the server on (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for SSE server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE server (default: 8000)",
    )

    args = parser.parse_args()

    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            mcp.settings.host = args.host
            mcp.settings.port = args.port
            mcp.run(transport="sse")
        else:
            sys.exit(1)
    finally:
        close_kg_client()


if __name__ == "__main__":
    main()
