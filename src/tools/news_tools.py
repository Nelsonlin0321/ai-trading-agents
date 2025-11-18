from langchain.tools import tool
from src.services.tradingeconomics.api_market_news import News
from src.tools.actions.news import MarketNewsAct

market_news_action = MarketNewsAct()


def convert_news_to_markdown_text(news: News):
    title = news["title"]
    markdown_text = f"### {title}\n"
    markdown_text += (
        f"Importance: {news['importance']} Date: {news['date']}  {news['time_ago']} "
    )
    markdown_text += f"Expiration: {news['expiration']} \n\n"
    markdown_text += f"{news['description']}\n"
    return markdown_text


@tool(market_news_action.name)
async def get_latest_market_news():
    """
    Retrieve the most recent market news headlines and summaries for the United States.

    This tool is useful for:
    - Keeping track of breaking economic events that may move markets
    - Gathering quick context before making trading or investment decisions
    - Monitoring scheduled data releases and their market impact
    - Staying informed on Fed policy hints, inflation updates, employment figures, and other macro drivers
    - Comparing the relative importance (high/medium/low) of each news item
    - Filtering news by expiration date to focus only on still-relevant stories

    Returns a markdown-formatted string containing the title, importance level, date, expiration, and description of each news item.
    """

    news = await market_news_action.arun()

    # Convert each news item to markdown text
    markdown_news = "\n\n".join(convert_news_to_markdown_text(n) for n in news)

    # markdown_news = utils.dicts_to_markdown_table(news)
    heading = "## United States Market News"
    return heading + "\n" + markdown_news


__all__ = ["get_latest_market_news"]

if __name__ == "__main__":
    #  python -m src.tools.news_tools
    import asyncio

    result = asyncio.run(get_latest_market_news())  # type: ignore
    print(result)
