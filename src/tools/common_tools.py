from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage
from src.utils.message import combine_ai_messages
from src.context import Context

from src.tools_adaptors import (
    GetUserInvestmentStrategyAct,
    WriteInvestmentReportEmailAct,
    SendInvestmentReportEmailAct,
)


get_user_investment_strategy_act = GetUserInvestmentStrategyAct()
send_investment_report_email_act = SendInvestmentReportEmailAct()
write_investment_report_email_act = WriteInvestmentReportEmailAct()


@tool("get_user_investment_strategy")
async def get_user_investment_strategy(runtime: ToolRuntime[Context]):
    """
    Retrieve the current investment strategy for the trading portfolio.

    This tool fetches the current investment strategy for the trading portfolio.

    Possible tool purposes:
    - Allow you to decide which assets or sectors to trade based on the user’s stated risk tolerance or philosophy.
    - Allow you to decide which analysts to heavily use for the investment strategy.
    - Surface the strategy to a dashboard or chat interface so the user can confirm or update it before orders are placed.
    - Act as a guard-rail that prevents trades violating the strategy (e.g., no crypto for a “dividend-income” strategy).

    Returns
    -------
    Investment Strategy
        A string representing the current investment strategy for the trading portfolio.
    """
    return await get_user_investment_strategy_act.arun(runtime.context.bot.id)


@tool(write_investment_report_email_act.name)
async def write_summary_report(runtime: ToolRuntime[Context]):
    """
    Write the final investment report email for the current trading run.
    """
    states = runtime.state
    messages: list[BaseMessage] = states["messages"]  # type: ignore
    conversation = combine_ai_messages(messages)
    context = runtime.context
    return await write_investment_report_email_act.arun(
        model_name=context.model_name,
        botId=context.bot.id,
        run_id=context.run.id,
        conversation=conversation,
    )


@tool(send_investment_report_email_act.name)
async def send_summary_email_tool(runtime: ToolRuntime[Context]):
    """
    Send the final investment report email for the current trading run.
    """
    context = runtime.context
    return await send_investment_report_email_act.arun(
        run_id=context.run.id,
        user_id=context.bot.userId,
        bot_name=context.bot.name,
    )
