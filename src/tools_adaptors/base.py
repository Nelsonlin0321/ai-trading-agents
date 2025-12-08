from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from src.utils import async_timeout

T = TypeVar("T")


class Action(ABC, Generic[T]):
    @abstractmethod
    @async_timeout(30)
    async def arun(self, *args, **kwargs) -> T:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
