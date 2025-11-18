from loguru import logger
from upstash_redis.asyncio import Redis
from prisma import Prisma
from prisma.engine.errors import AlreadyConnectedError, NotConnectedError
from src.utils import get_env


prisma = Prisma(auto_register=True)


async def connect():
    try:
        await prisma.connect()
    except AlreadyConnectedError:
        logger.info("Already connected to Prisma")
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
