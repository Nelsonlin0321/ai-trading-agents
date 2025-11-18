from src.services.tradingeconomics.api_market_news import get_news
from src.tools.actions.base import Action


class MarketNewsTool(Action):
    @property
    def name(self):
        return "US Market News"

    async def arun(self):
        data = await get_news()
        return data
