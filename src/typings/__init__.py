from typing import Final, Literal, TypedDict

ERROR: Final = "ERROR"
ErrorLiteral = Literal["ERROR"]


TimelineValue = TypedDict(
    "TimelineValue",
    {
        "date": str,
        "value": float,
    },
)
