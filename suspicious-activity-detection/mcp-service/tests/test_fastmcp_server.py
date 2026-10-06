from __future__ import annotations

import asyncio

from fastmcp import Client

from tools import mcp


def test_fastmcp_lists_expected_sad_tools_and_calls_count_tool():
    async def exercise():
        async with Client(mcp) as client:
            listed = await client.list_tools()
            names = {item.name for item in listed}
            assert {
                "describe",
                "Get_all_activities",
                "Get_activity_by_zone",
                "Get_activity_by_zone_timestamp",
                "Search_retrospective_frames",
                "Get_trend_counts",
                "Get_event_count",
                "Get_all_zones",
            } <= names
            assert "subscribe" not in names
            result = await client.call_tool("Get_event_count", {"use_case": "kitchen"})
            assert not result.is_error

    asyncio.run(exercise())