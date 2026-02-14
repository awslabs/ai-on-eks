# # https://modelcontextprotocol.io/quickstart/server
# from typing import Any
# import httpx
# from mcp.server.fastmcp import FastMCP

# # Initialize FastMCP server
# mcp = FastMCP("weather")


# @mcp.tool()
# async def get_personal_info(state: str) -> str:
#     """Get weather alerts for a US state.

#     Args:
#         state: Two-letter US state code (e.g. CA, NY)
#     """
#     url = f"{NWS_API_BASE}/alerts/active/area/{state}"
#     data = await make_nws_request(url)

#     if not data or "features" not in data:
#         return "Unable to fetch alerts or no alerts found."

#     if not data["features"]:
#         return "No active alerts for this state."

#     alerts = [format_alert(feature) for feature in data["features"]]
#     return "\n---\n".join(alerts)