from typing import Sequence, TypeVar, Generic, cast, Any
import httpx
import dotenv
from src.services.utils import async_retry_on_status_code
from src.utils import get_env
dotenv.load_dotenv()


BASE_URL = "https://data.alpaca.markets"

ALPACA_API_KEY_ID = get_env("ALPACA_API_KEY")
ALPACA_API_SECRET_KEY = get_env("ALPACA_API_SECRET")


T = TypeVar("T")


class AlpacaAPIClient(Generic[T]):  # pylint:disable=too-few-public-methods
    """Async Alpaca Data API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: https://data.alpaca.markets
    - Adds required Alpaca auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
    - Exponential backoff with jitter

    Usage:
        client = AlpacaAPIClient(endpoint="/v2/stocks/snapshots")
        data = await client.get(symbols=["AAPL", "MSFT"])
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    @async_retry_on_status_code(status_codes=[429, 500, 502, 503, 504])
    async def get(
        self,
        *,
        symbols: Sequence[str] | str | None = None,
        endpoint: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> T:
        """Perform a GET request with retries.

        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
            "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
            "accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)

        if symbols:
            params = params or {}
            params["symbols"] = _normalize_symbols(symbols)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=self.timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


def _normalize_symbols(symbols: Sequence[str] | str) -> str:
    if isinstance(symbols, str):
        s = symbols.strip()
        if not s:
            raise ValueError("symbols must not be empty")
        return s
    cleaned = [sym.strip() for sym in symbols if sym and sym.strip()]
    if not cleaned:
        raise ValueError("symbols must contain at least one non-empty symbol")
    return ",".join(cleaned)


__all__ = [
    "AlpacaAPIClient",
]
