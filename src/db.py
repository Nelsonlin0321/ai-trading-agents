from typing import List
import asyncio
from upstash_redis.asyncio import Redis
from prisma import Prisma
from prisma.engine.errors import AlreadyConnectedError, NotConnectedError
from langchain_core.messages import AnyMessage
from src.utils import get_env


prisma = Prisma(auto_register=True)


async def connect():
    try:
        await prisma.connect()
    except AlreadyConnectedError:
        pass
        # logger.warning(
        #     f"Already connected to Prisma: {e} Traceback: {traceback.format_exc()}")
    finally:
        return prisma


async def disconnect():
    try:
        await prisma.disconnect()
    except NotConnectedError:
        pass


redis = Redis(
    url=get_env("UPSTASH_REDIS_REST_URL"), token=get_env("UPSTASH_REDIS_REST_TOKEN")
)

# Ephemera Cache agent messages and db message ids to reduce db / redis queries
AGENT_CACHED_MESSAGES: List[AnyMessage] = []
DB_CACHED_MSG_IDS: set[str] = set()
AGENT_CACHED_MESSAGES_LOCK: asyncio.Lock = asyncio.Lock()
DB_CACHED_MSG_IDS_LOCK: asyncio.Lock = asyncio.Lock()

__all__ = [
    "prisma",
    "redis",
    "connect",
    "disconnect",
    "AGENT_CACHED_MESSAGES",
    "DB_CACHED_MSG_IDS",
    "AGENT_CACHED_MESSAGES_LOCK",
    "DB_CACHED_MSG_IDS_LOCK",
]
