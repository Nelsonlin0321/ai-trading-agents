import os
import dotenv
from alpaca.trading.client import TradingClient  # pylint: disable=import-error,no-name-in-module

dotenv.load_dotenv()

client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=True,
)

__all__ = ["client"]
