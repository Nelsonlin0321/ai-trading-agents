from prisma import Prisma
from upstash_redis.asyncio import Redis
from prisma.engine.errors import AlreadyConnectedError, NotConnectedError
from typing import TypedDict, Any
from src.typings.agent_roles import AgentRole


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

CachedAgentMessage = TypedDict(
    "CachedAgentMessage",
    {
        "id": str,
        "role": AgentRole,
        "botId": str,
        "runId": str,
        "createdAt": str,
        "updatedAt": str,
        "messages": dict[str, Any],  # Serialized langchain message content
    },
)

CACHED_AGENTS_MESSAGES: list[CachedAgentMessage] = []

__all__ = [
    "prisma",
    "redis",
    "connect",
    "disconnect",
    "CachedAgentMessage",
    "CACHED_AGENTS_MESSAGES",
]
