import random
from typing import Any, Mapping, Sequence, TypeVar, Generic, cast
import httpx
import dotenv
from src.services.utils import async_retry_on_status_code

dotenv.load_dotenv()


BASE_URL = "https://tradingeconomics.com"


T = TypeVar("T")


class TradingEconomicsAPIClient(Generic[T]):  # pylint: disable=too-few-public-methods
    """Async TradingEconomics API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: https://tradingeconomics.com
    - Adds required auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
    - Exponential backoff with jitter

    Usage:
        client = TradingEconomicsAPIClient(endpoint="/ws/stream.ashx")
        # https://tradingeconomics.com/ws/stream.ashx?start=15&size=20&c=united states
        data = await client.get(params={"start": "15", "size": "20", "c": "united states"})
    """

    def __init__(
        self,
        endpoint: str,
        *,
        retries: int = 10,
        retry_statuses: Sequence[int] | None = None,
        base_delay_seconds: float = 0.3,
        max_delay_seconds: float = 30.0,
        jitter_seconds: float = 0.2,
    ) -> None:
        self.endpoint = endpoint
        self.retries = retries
        self.retry_statuses = set(retry_statuses or (429, 500, 502, 503, 504))
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds

    def _retry_delay(self, attempt: int) -> float:
        # Exponential backoff with jitter, similar to axiosRetry.exponentialDelay
        delay = min(self.base_delay_seconds * (2**attempt), self.max_delay_seconds)
        jitter = random.uniform(0.0, self.jitter_seconds)
        return delay + jitter

    @async_retry_on_status_code()
    async def get(
        self,
        *,
        endpoint: str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> T:
        """Perform a GET request with retries.
        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:144.0) Gecko/20100101 Firefox/144.0",
            "accept": "application/json, text/javascript, */*; q=0.01",
        }

        if headers:
            merged_headers.update(headers)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


__all__ = [
    "TradingEconomicsAPIClient",
]
