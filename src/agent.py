from typing import override, Optional

import dotenv
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard
from pydantic import BaseModel
from starlette.applications import Starlette

from entrypoints import web_search


class WebSearchAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Web Search",
            description="Searches the internet.",
            icon=None,
            entrypoints=[
                web_search.entrypoint,
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        match entrypoint:
            case web_search.entrypoint.id:
                await web_search.run(context, request)
            case _:
                raise ValueError()

def create_app() -> Starlette:
    dotenv.load_dotenv()
    agent = WebSearchAgent()
    app = build_agent_app(agent)
    return app
