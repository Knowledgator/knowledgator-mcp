import argparse
import sys
from typing import Any
from mcp.server.fastmcp import FastMCP

from knowledgator_mcp.client import close_kg_client
from knowledgator_mcp.config import settings
from knowledgator_mcp.tools import dataset_tools, finetuning_tools, inference_tools
from mcp.server.transport_security import TransportSecuritySettings

transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=settings.enable_dns_rebinding_protection,
    allowed_hosts=list(settings.allowed_hosts),
    allowed_origins=list(settings.allowed_origins),
)

mcp = FastMCP(
    name=settings.server_name,
    debug=settings.debug,
    transport_security=transport_security,
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
mcp.tool()(inference_tools.open_model_inference)


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
    parser.add_argument(
        "--allowed-hosts",
        default=None,
        help="Comma-separated list of allowed Host header values (e.g. 'mcp.example.com,localhost:*')",
    )
    parser.add_argument(
        "--allowed-origins",
        default=None,
        help="Comma-separated list of allowed Origin header values (e.g. 'https://example.ai')",
    )
    parser.add_argument(
        "--disable-dns-protection",
        action="store_true",
        default=False,
        help="Disable DNS rebinding protection (not recommended for production)",
    )

    args = parser.parse_args()

    # Apply transport security overrides if provided via CLI
    if mcp.settings.transport_security:
        if args.allowed_hosts is not None:
            mcp.settings.transport_security.allowed_hosts = [
                h.strip() for h in args.allowed_hosts.split(",") if h.strip()
            ]
        if args.allowed_origins is not None:
            mcp.settings.transport_security.allowed_origins = [
                o.strip() for o in args.allowed_origins.split(",") if o.strip()
            ]
        if args.disable_dns_protection:
            mcp.settings.transport_security.enable_dns_rebinding_protection = False

    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
        elif args.transport == "sse":
            mcp.settings.host = args.host
            mcp.settings.port = args.port

            # Managed Proxy (auth_type == 'api_proxy') -> wrap with per-request key capture
            if settings.auth_type.lower() == "api_proxy":
                import asyncio
                import uvicorn
                from knowledgator_mcp.auth.middleware import APIKeyCaptureMiddleware

                starlette_app = mcp.sse_app()
                wrapped_app = APIKeyCaptureMiddleware(starlette_app)

                config = uvicorn.Config(
                    wrapped_app,
                    host=mcp.settings.host,
                    port=mcp.settings.port,
                    log_level=mcp.settings.log_level.lower(),
                )
                server = uvicorn.Server(config)
                asyncio.run(server.serve())
            else:
                # Dedicated / Self-Hosted (auth_type == 'api_key') -> run native FastMCP directly
                mcp.run(transport="sse")
        else:
            sys.exit(1)
    finally:
        close_kg_client()


if __name__ == "__main__":
    main()
