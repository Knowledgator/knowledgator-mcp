import argparse
import logging
import sys
from typing import Any

import anyio
from mcp.server.fastmcp import FastMCP

from knowledgator_mcp.client import close_kg_client
from knowledgator_mcp.config import settings
from knowledgator_mcp.tools import dataset_tools, finetuning_tools, inference_tools
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger(__name__)


def _patch_fastmcp_streamable_http_cleanup() -> None:
    """Patches FastMCP StreamableHTTPServerTransport to cleanly close streams on session termination."""
    try:
        from mcp.server.streamable_http import StreamableHTTPServerTransport

        stream_logger = logging.getLogger("mcp.server.streamable_http")

        async def _graceful_clean_up_memory_streams(self: Any, request_id: Any) -> None:
            if request_id in self._request_streams:
                stream_pair = self._request_streams.get(request_id)
                if stream_pair:
                    send_stream, recv_stream = stream_pair
                    # Close writer first: reader receives EOF and exits async for loop cleanly
                    try:
                        await send_stream.aclose()
                    except Exception as e:
                        stream_logger.error(
                            "Error closing send stream for request %s: %s", request_id, e, exc_info=True
                        )

                    # Yield control to event loop so the reader task processes EOF and exits
                    try:
                        await anyio.sleep(0)
                    except Exception as e:
                        stream_logger.error(
                            "Error yielding to event loop for request %s: %s", request_id, e, exc_info=True
                        )

                    # Close reader safely
                    try:
                        await recv_stream.aclose()
                    except Exception as e:
                        stream_logger.error(
                            "Error closing recv stream for request %s: %s", request_id, e, exc_info=True
                        )
                self._request_streams.pop(request_id, None)

        StreamableHTTPServerTransport._clean_up_memory_streams = _graceful_clean_up_memory_streams
    except Exception as e:
        logger.debug("Failed to patch FastMCP stream cleanup: %s", e)


_patch_fastmcp_streamable_http_cleanup()

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
mcp.tool()(dataset_tools.get_dataset_analytics_report)
mcp.tool()(dataset_tools.list_data)
mcp.tool()(dataset_tools.download_dataset)
mcp.tool()(dataset_tools.upload_dataset)


# Register Fine-Tuning Tools
mcp.tool()(finetuning_tools.start_training_job)
mcp.tool()(finetuning_tools.get_training_status)
mcp.tool()(finetuning_tools.list_trainings)
mcp.tool()(finetuning_tools.stop_training_job)
mcp.tool()(finetuning_tools.get_model_details)
mcp.tool()(finetuning_tools.get_inference_details)
mcp.tool()(finetuning_tools.get_training_metrics_report)

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
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="Transport type to run the server on (default: streamable-http)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for server (default: 8000)",
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
        elif args.transport in ("sse", "streamable-http"):
            mcp.settings.host = args.host
            mcp.settings.port = args.port

            # Managed Proxy (auth_type == 'api_proxy') -> wrap with per-request key capture
            if settings.auth_type.lower() == "api_proxy":
                import asyncio
                import uvicorn
                from knowledgator_mcp.auth.middleware import APIKeyCaptureMiddleware

                if args.transport == "streamable-http":
                    starlette_app = mcp.streamable_http_app()
                else:
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
                mcp.run(transport=args.transport)
        else:
            sys.exit(1)
    finally:
        close_kg_client()


if __name__ == "__main__":
    main()
