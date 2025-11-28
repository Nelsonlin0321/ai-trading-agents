from src.tools.actions.base import Action
from src.services.yfinance.api_info import async_get_ticker_info
from src.tools.actions import utils


class FundamentalAct(Action):
    @property
    def name(self):
        return "get_comprehensive_fundamental_data"

    async def arun(self, ticker: str) -> str:
        info = await async_get_ticker_info(ticker)
        categorized_data = utils.get_categorized_metrics(info)
        md = utils.format_fundamentals_markdown(categorized_data, ticker)
        return md
