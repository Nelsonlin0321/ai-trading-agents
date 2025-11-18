import asyncio
import os
from google import genai
from google.genai import types

from src.tools.actions.base import Action
from src.services.utils import redis_cache
from src.utils import async_wrap
from src.utils import get_current_date


class GoogleMarketResearch(Action):

    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        self.config = types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            max_output_tokens=65535,
            tools=tools,
            thinking_config=types.ThinkingConfig(
                # include_thoughts=True,
                thinking_budget=-1,
            ),
        )

    @property
    def name(self):
        return "Google Market Research"

    @redis_cache(function_name="GoogleMarketResearch.arun", ttl=180)
    async def arun(self):  # type: ignore
        return await self.run()  # type: ignore

    @async_wrap
    def run(self):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )

        with open("./src/tools/actions/google_research.md", mode="r", encoding="utf-8") as f:
            prompt_template = f.read()

        datetime_str = get_current_date()
        prompt = prompt_template.format(datetime=datetime_str)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=self.config,
        )

        text = ""
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    if part.text:
                        text += part.text

        return text


if __name__ == "__main__":
    # python -m src.tools.actions.research
    google_market_research_action = GoogleMarketResearch()
    result = asyncio.run(google_market_research_action.arun())  # type: ignore
    print(result)
