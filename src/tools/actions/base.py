from abc import ABC, abstractmethod
from typing import TypeVar, Generic
T = TypeVar("T")


class Action(ABC, Generic[T]):
    @abstractmethod
    async def arun(self, *args, **kwargs) -> T:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
