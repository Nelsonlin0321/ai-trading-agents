import os
import time
import pytz
import boto3
import smtplib
import asyncio
import traceback


from tqdm import tqdm
from loguru import logger
from datetime import datetime
from email.mime.text import MIMEText
from functools import partial, wraps
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any, TypeVar, Coroutine
from markdownify import markdownify as md


from src.typings import ErrorLiteral, ERROR


T = TypeVar("T")


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


def get_env(name: str, default="") -> str:
    value = os.environ.get(name)
    if not value and default == "":
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or default


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


def async_wrap():
    def decorator(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args, loop=None, executor=None, **kwargs):
            if loop is None:
                loop = asyncio.get_event_loop()
            pfunc = partial(func, *args, **kwargs)
            return await loop.run_in_executor(executor, pfunc)

        return wrapper

    return decorator


def get_new_york_datetime() -> datetime:
    return datetime.now(tz=pytz.timezone("America/New_York"))


def convert_html_to_markdown(html: str) -> str:
    return md(html, strip=["a"])


def async_retry(
    base_delay: float = 0.5,
    max_retries: int = 5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    max_delay_seconds: float = 5.0,
    silence_error: bool = os.getenv("SILENCE_ERROR") == "1",
):
    def decorator(func: Callable[..., Coroutine[Any, Any, T]]):
        async def wrapper(*args, **kwargs) -> T | ErrorLiteral:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    error_message = (
                        f"Error running {func.__name__} after {max_retries} retries: {e}"
                        f"\nArgs: {args}\nKwargs: {kwargs}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                    logger.error(error_message)
                    if retries < max_retries:
                        retries += 1
                        delay = min(
                            max_delay_seconds, base_delay * (2 ** (retries - 1))
                        )
                        await asyncio.sleep(delay)
                    else:
                        if recipient := os.getenv("GMAIL"):
                            send_gmail_email(
                                subject=f"Error running {func.__name__}",
                                recipient=recipient,
                                html_body=error_message,
                            )

                        if not silence_error:
                            raise
                        else:
                            return ERROR

        return wrapper

    return decorator


def async_timeout(
    seconds: float,
):
    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)

        return wrapper

    return decorator


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    silence_error: bool = os.getenv("SILENCE_ERROR") == "1",
):
    def decorator(func: Callable[..., T]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> T | ErrorLiteral:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = (
                        f"Error running {func.__name__} after {max_retries} retries: {e}"
                        f"\nArgs: {args}\nKwargs: {kwargs}\n"
                        f"Traceback: {traceback.format_exc()}"
                    )
                    logger.error(error_message)
                    attempt += 1
                    if attempt >= max_retries:
                        if not silence_error:
                            raise
                        return ERROR
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_time)

        return wrapper

    return decorator


@retry(max_retries=3, silence_error=True)
def send_gmail_email(subject: str, recipient: str, html_body: str):
    msg = MIMEMultipart()
    sender = os.getenv("GMAIL")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    if not sender:
        logger.warning("GMAIL environment variable to send email is not set")
        return

    if not gmail_password:
        logger.warning(
            "GMAIL_APP_PASSWORD environment variable to send email is not set"
        )
        return

    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject

    # Attach HTML content
    msg.attach(MIMEText(html_body, "html"))

    # Create SMTP session
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()  # Enable TLS

    # Login to Gmail
    server.login(sender, gmail_password)

    # Send email
    server.send_message(msg)

    # Close the connection
    server.quit()
    logger.info(f"Email Sent successfully to {recipient}")
    return f"Email Sent successfully to {recipient}"


@retry(max_retries=3, silence_error=True)
def send_ses_email(
    subject: str,
    recipient: str,
    sender: str = "notifications@sandx.ai",
    html_body: str = "",
):
    client = boto3.client("ses", region_name=os.getenv("AWS_REGION", "us-east-1"))

    response = client.send_email(
        Destination={
            "ToAddresses": [
                recipient,
            ],
        },
        Message={
            "Body": {
                "Html": {
                    "Charset": "UTF-8",
                    "Data": html_body,
                }
            },
            "Subject": {
                "Charset": "UTF-8",
                "Data": subject,
            },
        },
        Source=sender,
    )

    logger.info(f"Email sent! Message ID: {response['MessageId']}")

    return f"Email Sent successfully to {recipient}"


if __name__ == "__main__":
    #  python -m src.utils.__init__
    send_ses_email(
        subject="Test Email",
        recipient="sandx.ai.contact@gmail.com",
        html_body="<h1>Hello, World!</h1>",
    )
