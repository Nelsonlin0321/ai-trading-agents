from langchain_core.messages import AIMessage, BaseMessage


def combine_ai_messages(messages: list[BaseMessage]) -> str:
    return "\n".join(
        [str(msg.content) for msg in messages if isinstance(msg, AIMessage)]
    )
