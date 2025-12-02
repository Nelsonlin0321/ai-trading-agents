from datetime import date, timedelta

from src.services.alpaca import get_historical_price_bars
from src.services.yfinance.api_info import async_get_ticker_info
from src.tools_adaptors import utils
from src.tools_adaptors.base import Action
from src.utils import constants


class FundamentalRiskDataAct(Action):
    @property
    def name(self):
        return "get_comprehensive_fundamental_risk_data"

    async def arun(self, ticker: str) -> str:
        info = await async_get_ticker_info(ticker)
        info = utils.preprocess_info_dict(info)
        categorized_data = utils.get_categorized_metrics(
            info, categories_map=constants.FUNDAMENTAL_RISK_CATEGORIES
        )
        md = utils.format_fundamentals_markdown(categorized_data, ticker)
        return md


class VolatilityRiskAct(Action):
    @property
    def name(self):
        return "get_volatility_risk_indicators"

    async def arun(self, ticker: str) -> str:
        """Get volatility risk indicators for a ticker"""
        start = (date.today() - timedelta(days=356 + 7)).isoformat()
        end = date.today().isoformat()
        price_bars = await get_historical_price_bars(
            symbols=[ticker], timeframe="1Day", start=start, end=end, sort="asc"
        )
        price_bars = price_bars[ticker]
        risk = utils.calculate_volatility_risk(price_bars)
        md = utils.format_volatility_risk_markdown(risk, ticker)
        return md


class PriceRiskAct(Action):
    @property
    def name(self):
        return "get_price_risk_indicators"

    async def arun(self, ticker: str) -> str:
        """Get price risk indicators for a ticker"""
        start = (date.today() - timedelta(days=356 + 7)).isoformat()
        end = date.today().isoformat()
        price_bars = await get_historical_price_bars(
            symbols=[ticker], timeframe="1Day", start=start, end=end, sort="asc"
        )
        price_bars = price_bars[ticker]
        risk = utils.calculate_price_risk(price_bars)
        md = utils.format_price_risk_markdown(risk, ticker)
        return md


__all__ = ["FundamentalRiskDataAct", "VolatilityRiskAct", "PriceRiskAct"]
