from src.tools_adaptors.base import Action
from src.services.yfinance.api_info import async_get_ticker_info
from src.tools_adaptors import utils
from src.utils import constants
from src.utils import async_timeout


class FundamentalDataAct(Action):
    @property
    def name(self):
        return "get_comprehensive_fundamental_data"

    @async_timeout(30)
    async def arun(self, ticker: str) -> str:
        info = await async_get_ticker_info(ticker)
        info = utils.preprocess_info_dict(info)
        categorized_data = utils.get_categorized_metrics(
            info, categories_map=constants.FUNDAMENTAL_CATEGORIES
        )
        md = utils.format_fundamentals_markdown(categorized_data, ticker)
        return md


__all__ = ["FundamentalDataAct"]
