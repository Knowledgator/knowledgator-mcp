import argparse
import sys
from mcp.server.fastmcp import FastMCP

from knowledgator_mcp.client import close_kg_client
from knowledgator_mcp.config import settings
from knowledgator_mcp.tools import dataset_tools, finetuning_tools, inference_tools

mcp = FastMCP(
    name=settings.server_name,
    debug=settings.debug,
)

# Register Dataset Tools
mcp.tool()(dataset_tools.generate_dataset_plan)
mcp.tool()(dataset_tools.start_dataset_generation)
mcp.tool()(dataset_tools.list_datasets)
mcp.tool()(dataset_tools.get_dataset_details)
mcp.tool()(dataset_tools.download_dataset)
mcp.tool()(dataset_tools.upload_dataset)


# Register Fine-Tuning Tools
mcp.tool()(finetuning_tools.start_training_job)
mcp.tool()(finetuning_tools.get_training_status)
mcp.tool()(finetuning_tools.list_trainings)
mcp.tool()(finetuning_tools.stop_training_job)

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
