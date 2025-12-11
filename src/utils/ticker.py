from src import db


async def is_valid_ticker(ticker: str):
    ticker = ticker.replace("-", ".")
    ticker_record = await db.prisma.ticker.find_unique(where={"ticker": ticker})

    return ticker_record is not None


__all__ = ["is_valid_ticker"]
