import os

import httpx
from ichatbio.agent_response import IChatBioAgentProcess, ResponseContext
from ichatbio.types import AgentEntrypoint
from openai import AsyncOpenAI

description = """\
Searches the internet using Google's Vertex AI.
"""

entrypoint = AgentEntrypoint(
    id="web_search",
    description=description,
    parameters=None
)


async def run(context: ResponseContext, request: str):
    async with context.begin_process("Searching the Internet") as process:
        process: IChatBioAgentProcess

        await process.log("Searching the Internet")
        sources, quotes = await search_the_world_wide_web(request)

        await process.log(quotes, data={"sources": sources})


WEB_SEARCH_PROMPT = """\
You find information on the internet and extract quotes that are relevant to the user's request. You do not add or
change any information. Quote information exactly as it appears on the web, without correcting typos.
"""

web_search_options = {
    "search_context_size": "low"
}

async def search_the_world_wide_web(request: str) -> (list[dict], str):
    client = AsyncOpenAI(api_key=os.getenv("WEB_SEARCH_API_KEY"))
    completion = await client.chat.completions.create(
        model="gemini-2.5-flash",
        web_search_options=web_search_options,
        messages=[
            {"role": "system", "content": WEB_SEARCH_PROMPT},
            {"role": "user", "content": request}
        ],
    )
    result = completion.choices[0].message
    
    content_text = result.content if isinstance(result.content, str) else ""
    sources: list[dict] = await parse_vertex_ai_grounding(completion)

    return sources, content_text


async def resolve_redirect(url: str) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10) as http:
            response = await http.head(url)
            if not response.is_success:
                response = await http.get(url)
            return str(response.url)
    except Exception:
        return url


async def parse_vertex_ai_grounding(completion) -> list[dict]:
    """Extract {quote, source} pairs from Vertex AI grounding metadata.
    Resolves redirect URIs to final URLs; falls back to original URI on failure.

    "groundingChunks": [
    {
        "web": {
        "uri": "https://vertexaisearch.cloud.google.com/...idigbio.org",
        "title": "idigbio.org",
        "domain": "idigbio.org"
        }
    },
]

  "groundingSupports": [
    {
        "segment": {
            "startIndex": 341,
            "endIndex": 521,
            "text": "iChatBio can provide information about biodiversity..."
        },
        "groundingChunkIndices": [1]
    }
]
    """
    def _get(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    sources: list[dict] = []
    vertex_meta = _get(completion, "vertex_ai_grounding_metadata") or []
    if not vertex_meta:
        return sources

    resolved_cache: dict[str, str] = {}
    added_pairs: set[tuple[str, str]] = set()

    async def get_resolved(url: str) -> str:
        cached = resolved_cache.get(url)
        if cached is not None:
            return cached
        try:
            resolved = await resolve_redirect(url)
        except Exception:
            resolved = url
        resolved_cache[url] = resolved
        return resolved

    for vm in vertex_meta:
        grounding_chunks = _get(vm, "groundingChunks") or []
        idx_to_uri: dict[int, str] = {}
        for idx, chunk in enumerate(grounding_chunks):
            web = _get(chunk, "web") or {}
            uri = _get(web, "uri")
            if uri:
                idx_to_uri[idx] = uri

        supports = _get(vm, "groundingSupports") or []
        for support in supports:
            segment = _get(support, "segment") or {}
            quote_text = _get(segment, "text")
            indices = _get(support, "groundingChunkIndices") or []
            for i in indices:
                uri = idx_to_uri.get(i)
                if quote_text and uri:
                    final_url = await get_resolved(uri)
                    key = (quote_text, final_url)
                    if key not in added_pairs:
                        added_pairs.add(key)
                        sources.append({
                            "quote": quote_text,
                            "source": final_url
                        })

    return sources
