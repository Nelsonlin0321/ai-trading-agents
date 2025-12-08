import asyncio
import os
from typing import Awaitable, Callable, Any, TypeVar, Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial, wraps
import pytz
from tqdm import tqdm
from html_to_markdown import convert, ConversionOptions


def multi_threading(function, parameters, max_workers=5, desc=""):
    pbar = tqdm(total=len(parameters), desc=desc, leave=True, position=0)
    event = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # not need chucksize
        for result in executor.map(function, parameters):
            event.append(result)
            pbar.update(1)
    pbar.close()

    return event


def get_current_date() -> str:
    return datetime.now(tz=pytz.timezone("America/New_York")).strftime("%Y-%m-%d")


def get_current_timestamp() -> str:
    return datetime.now(tz=pytz.timezone("America/New_York")).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def dicts_to_markdown_table(data: list[dict[str, Any]]):
    """
    Convert a list of dictionaries into a Markdown table string.

    Args:
        data (list[dict]): List of dictionaries with the same keys.

    Returns:
        str: Markdown formatted table.
    """
    if not data:
        return ""

    # Extract headers from the first dictionary
    headers = list(data[0].keys())

    # Build header row
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Build data rows
    for row in data:
        table += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"

    return table


def dict_to_markdown_table(data: dict[str, Any]):
    """
    Convert a dictionary into a Markdown table string.

    Args:
        data (dict): Dictionary with the same keys.

    Returns:
        str: Markdown formatted table.
    """
    if not data:
        return ""

    # Extract headers from the first dictionary
    headers = list(data.keys())

    # Build header row
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Build data rows
    table += "| " + " | ".join(str(data[h]) for h in headers) + " |\n"

    return table


def format_percent_change(p: float, precision: int = 2) -> str:
    p = p * 100
    return f"{p:+.{precision}f}%"


def format_percent(p: float, precision: int = 2) -> str:
    p = p * 100
    return f"{p:.{precision}f}%"


def format_currency(c: float, precision: int = 2) -> str:
    return f"${c:.{precision}f}"


def format_float(f: float, precision: int = 2) -> str:
    return f"{f:.{precision}f}"


def human_format(num, precision=2):
    suffixes = ["", "K", "M", "B", "T"]
    num = float(num)
    if num == 0:
        return "0"

    magnitude = 0
    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        num /= 1000.0
        magnitude += 1

    return f"{num:.{precision}f}{suffixes[magnitude]}"


def format_date(date_string: str = "2025-11-15T00:59:00.095784453Z") -> str:
    date: datetime = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
    dt_ny = date.astimezone(pytz.timezone("America/New_York"))
    return dt_ny.strftime("%b %d, %Y") + " EST"


def format_datetime(datetime_string: str = "2025-11-15T00:59:00.095784453Z") -> str:
    date: datetime = datetime.fromisoformat(datetime_string.replace("Z", "+00:00"))
    dt_ny = date.astimezone(pytz.timezone("America/New_York"))
    return dt_ny.strftime("%b %d, %Y %H:%M:%S") + " EST"


def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


def get_new_york_datetime() -> datetime:
    return datetime.now(tz=pytz.timezone("America/New_York"))


def convert_html_to_markdown(html: str) -> str:
    return convert(html, options=ConversionOptions(strip_tags={"a"}))


def async_retry(
    base_delay: float = 0.5,
    max_retries: int = 5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    max_delay_seconds: float = 5.0,
) -> Callable[[Callable[..., Awaitable[object]]], Callable[..., Awaitable[object]]]:
    def decorator(
        func: Callable[..., Awaitable[object]],
    ) -> Callable[..., Awaitable[object]]:
        async def wrapper(*args, **kwargs) -> object:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions:
                    if retries < max_retries:
                        retries += 1
                        delay = min(
                            max_delay_seconds, base_delay * (2 ** (retries - 1))
                        )
                        await asyncio.sleep(delay)
                    else:
                        raise

        return wrapper

    return decorator


T = TypeVar("T")


def async_timeout(
    seconds: float,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Coroutine[Any, Any, T]]]:
    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator
