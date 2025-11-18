from src.services.tradingeconomics.api_market_news import get_news
from src.tools.actions.base import Action


class MarketNewsAct(Action):
    @property
    def name(self):
        return "Get US Market News"

    async def arun(self):
        data = await get_news()
        return data


if __name__ == "__main__":
    import asyncio
    # python -m src.tools.actions.news
    market_news_action = MarketNewsAct()
    result = asyncio.run(market_news_action.arun())  # type: ignore
    print(result)
