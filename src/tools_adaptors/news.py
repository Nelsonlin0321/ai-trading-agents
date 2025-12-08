from datetime import date, timedelta
from src.services.tradingeconomics import get_news
from src.services.alpaca import get_news as get_alpaca_news
from src.tools_adaptors.base import Action
from src.utils import convert_html_to_markdown


class MarketNewsAct(Action):
    @property
    def name(self):
        return "get_latest_market_news"

    async def arun(self):
        data = await get_news()
        return data


class EquityNewsAct(Action):
    @property
    def name(self):
        return "get_latest_equity_news"

    async def arun(self, symbol: str):
        data = await get_alpaca_news(
            symbols=[symbol],
            start=(date.today() - timedelta(days=5)).isoformat(),
            end=date.today().isoformat(),
            sort="desc",
            limit=15,
        )

        news_list = data["news"]

        image_size_map = {"large": 3, "medium": 2, "small": 1, "thumb": 0}
        top_content = 3
        formatted_news: list[str] = []
        for i, new in enumerate(news_list):
            headline = new["headline"]
            summary = new["summary"]
            published_at = new["created_at"]
            image_url = (
                min(new["images"], key=lambda x: image_size_map.get(x["size"], 1))[
                    "url"
                ]
                if new.get("images")
                else ""
            )
            if i <= top_content:
                content = convert_html_to_markdown(new["content"])
            else:
                content = summary

            article = f"### {headline}\n"
            article += f"Published: {published_at}\n\n"
            article += f"![{headline}]({image_url})\n\n"
            article += f"{content}\n\n"
            formatted_news.append(article)

        heading = f"## Latest {symbol} News"
        return heading + "\n\n" + "\n\n".join(formatted_news)


__all__ = ["MarketNewsAct", "EquityNewsAct"]

if __name__ == "__main__":
    import asyncio

    # python -m src.tools.actions.news
    market_news_action = MarketNewsAct()
    result = asyncio.run(market_news_action.arun())  # type: ignore
    print(result)
