import os
from typing import Any, Generic, Mapping, TypeVar, cast
import dotenv
import httpx
from src.services.utils import async_retry_on_status_code
from src.utils import get_env

dotenv.load_dotenv()


BASE_URL = os.getenv("SANDX_AI_URL", "http://localhost:3000/api")

API_KEY = get_env("API_KEY")


T = TypeVar("T")


class SandxAPIClient(Generic[T]):  # pylint: disable=too-few-public-methods
    """Async Sandx AI API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: http://localhost:3000
    - Adds required Sandx AI auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
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
        endpoint: str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> T:
        """Perform a GET request with retries.

        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "Authorization": f"Bearer {API_KEY}",
            "accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=self.timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


__all__ = [
    "SandxAPIClient",
]
