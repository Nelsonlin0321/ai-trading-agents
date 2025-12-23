import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, TypeVar, cast, Coroutine, Any
import httpx
from loguru import logger
from prisma import types
from src.db import prisma, redis
import traceback

T = TypeVar("T")


class APIError(RuntimeError):
    pass


def extract_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str):
                return msg
    except Exception:
        # Fallback to text if JSON parsing fails
        pass
    text = response.text
    return text if text else None


def _extract_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str):
                return msg
    except Exception:
        # Fallback to text if JSON parsing fails
        pass
    text = response.text
    return text if text else None


def in_db_cache(
    function_name: str, ttl: int
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async decorator to cache function results in the database.

    - Stores results in the `Cache` table with a unique key derived from args/kwargs.
    - Respects TTL (seconds) via `expiresAt` and returns cached content when valid.

    Args:
        function_name: Logical function identifier to namespace cache keys.
        ttl: Time-to-live in seconds for the cached entry.

    Returns:
        A decorator that caches the async function's JSON-serializable result.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            try:
                pos_args = args[1:] if len(args) > 0 else args
                if "start" in kwargs:
                    start = kwargs["start"]
                    kwargs["start"] = datetime.fromisoformat(
                        start.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")

                if "end" in kwargs:
                    end = kwargs["end"]
                    kwargs["end"] = datetime.fromisoformat(
                        end.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")

                if "symbols" in kwargs:
                    symbols = kwargs["symbols"]
                    kwargs["symbols"] = sorted(symbols)

                key_payload = {"args": pos_args, "kwargs": kwargs}
                cache_key = json.dumps(key_payload, sort_keys=True)
                now = datetime.now(timezone.utc)

                # Try existing unexpired cache
                existing = await prisma.cache.find_first(
                    where={
                        "function": function_name,
                        "key": cache_key,
                        "expiresAt": {"gt": now},
                    }
                )
                if existing is not None and isinstance(existing.content, str):
                    logger.info(f"Cache hit for {function_name} with key {cache_key}")
                    return cast(T, json.loads(existing.content))

                # Compute fresh result
                result = await func(*args, **kwargs)
                expires_at = now + timedelta(seconds=ttl)

                # Upsert by (function, key) if present; otherwise create
                any_existing = await prisma.cache.find_first(
                    where={
                        "function": function_name,
                        "key": cache_key,
                    }
                )
                if any_existing is not None:
                    await prisma.cache.update(
                        where={"id": any_existing.id},
                        data={
                            "content": json.dumps(result),
                            "expiresAt": expires_at,
                        },
                    )
                else:
                    await prisma.cache.create(
                        data=types.CacheCreateInput(
                            function=function_name,
                            key=cache_key,
                            content=json.dumps(result),
                            expiresAt=expires_at,
                        )
                    )
                return result

            except Exception as e:
                logger.error(
                    f"Error in {function_name} with args {args} and kwargs {kwargs}: {e} Traceback: {traceback.format_exc()}"
                )
                raise

        return wrapper

    return decorator


def redis_cache(
    function_name: str, ttl: int
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]
]:
    """Async decorator to cache function results in the database.

    - Stores results in Redis with a unique key derived from args/kwargs.
    - Respects TTL (seconds) via `expiresAt` and returns cached content when valid.

    Args:
        function_name: Logical function identifier to namespace cache keys.
        ttl: Time-to-live in seconds for the cached entry.

    Returns:
        A decorator that caches the async function's JSON-serializable result.
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args, **kwargs) -> T:
            # print(
            #     f"function_name:{function_name} args:{args}, kwargs:{kwargs}")
            if "start" in kwargs:
                start = kwargs["start"]
                kwargs["start"] = datetime.fromisoformat(
                    start.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")

            if "end" in kwargs:
                end = kwargs["end"]
                kwargs["end"] = datetime.fromisoformat(
                    end.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")

            if "symbols" in kwargs:
                symbols = kwargs["symbols"]
                kwargs["symbols"] = sorted(symbols)

            key_payload = {
                "args": sorted([str(a) for a in args]),
                "kwargs": kwargs,
                "function_name": function_name,
            }
            cache_key = json.dumps(key_payload, sort_keys=True)

            existing = await redis.get(cache_key)

            if existing:
                logger.info(f"Cache Redis hit for {function_name} with key {cache_key}")
                return cast(T, json.loads(existing))

            # Compute fresh result
            result = await func(*args, **kwargs)
            await redis.set(cache_key, json.dumps(result), ex=ttl)
            return result

        return wrapper

    return decorator


def async_retry_on_status_code(
    base_delay: float = 1,
    max_retries: int = 5,
    status_codes: list[int] = [],
    max_delay_seconds: float = 30.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if (
                        len(status_codes) == 0 or e.response.status_code in status_codes
                    ) and retries < max_retries:
                        retries += 1
                        delay = min(
                            max_delay_seconds, base_delay * (2 ** (retries - 1))
                        )
                        logger.info(
                            f"Retrying in {delay}s due to status code: {e.response.status_code}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        last_error_message = _extract_error_message(e.response) or str(
                            e
                        )
                        logger.error(last_error_message)
                        raise APIError(last_error_message)

        return wrapper

    return decorator


__all__ = ["redis_cache", "async_retry_on_status_code", "APIError", "in_db_cache"]
