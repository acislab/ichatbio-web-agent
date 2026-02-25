import pytest
from ichatbio.agent_response import ProcessLogResponse

import pytest_asyncio

from agent import WebSearchAgent


@pytest_asyncio.fixture()
def agent():
    return WebSearchAgent()


@pytest.mark.asyncio
async def test_web_search(agent, context, messages):
    await agent.run(context, "What's iChatbio?", "web_search", None)

    process_messages = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert process_messages[0].text == "Searching the Internet"
    assert "sources" in process_messages[1].data
    assert len(process_messages[1].data["sources"]) > 0

