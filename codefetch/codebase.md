<filetree>
Project Structure:
├── scripts
│   └── clear_notebook_outputs.py
├── src
│   ├── agents
│   │   ├── chief_investment_officer
│   │   │   └── agent.py
│   │   ├── equity_research_analyst
│   │   │   └── agent.py
│   │   ├── fundamental_analyst
│   │   │   └── agent.py
│   │   ├── market_analyst
│   │   │   └── agent.py
│   │   ├── risk_analyst
│   │   │   └── agent.py
│   │   ├── trading_executive
│   │   │   └── agent.py
│   │   └── __init__.py
│   ├── middleware
│   │   └── __init__.py
│   ├── prompt
│   │   ├── __init__.py
│   │   ├── background.py
│   │   ├── roles.py
│   │   └── system.py
│   ├── services
│   │   ├── alpaca
│   │   │   ├── __init__.py
│   │   │   ├── api_client.py
│   │   │   ├── api_historical_bars.py
│   │   │   ├── api_latest_quotes.py
│   │   │   ├── api_most_active_stockers.py
│   │   │   ├── api_news.py
│   │   │   ├── api_snapshots.py
│   │   │   ├── sdk_trading_client.py
│   │   │   └── typing.py
│   │   ├── sandx_ai
│   │   │   ├── __init__.py
│   │   │   ├── api_client.py
│   │   │   ├── api_portfolio_timeline_value.py
│   │   │   ├── api_position.py
│   │   │   └── typing.py
│   │   ├── tradingeconomics
│   │   │   ├── __init__.py
│   │   │   ├── api_client.py
│   │   │   └── api_market_news.py
│   │   ├── yfinance
│   │   │   └── api_info.py
│   │   └── utils.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── common_tools.py
│   │   ├── fundamental_data_tools.py
│   │   ├── handoff_tools.py
│   │   ├── news_tools.py
│   │   ├── portfolio_tools.py
│   │   ├── research_tools.py
│   │   ├── risk_tools.py
│   │   ├── stock_tools.py
│   │   └── trading_tools.py
│   ├── tools_adaptors
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── fundamental_data_utils.py
│   │   │   ├── portfolio_timeline_value.py
│   │   │   ├── recommendations.py
│   │   │   └── risk_analysis.py
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── common.py
│   │   ├── fundamental_data.py
│   │   ├── google_equity_research.md
│   │   ├── google_research.md
│   │   ├── news.py
│   │   ├── portfolio.py
│   │   ├── research.py
│   │   ├── risk.py
│   │   ├── stocks.py
│   │   └── trading.py
│   ├── typings
│   │   ├── __init__.py
│   │   ├── agent_roles.py
│   │   ├── context.py
│   │   └── langgraph_agents.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── message.py
│   │   └── ticker.py
│   ├── context.py
│   ├── db.py
│   ├── main.py
│   └── models.py
├── .dockerignore
├── .python-version
├── Dockerfile
├── TODO.md
├── pyproject.toml
├── requirements.txt
└── schema.prisma

</filetree>

<source_code>
.dockerignore
```
.git
.gitignore
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.DS_Store
.mypy_cache
.pytest_cache
.hypothesis
.DS_Store
images/
notebooks/
uv.lock
```

.python-version
```
3.12
```

Dockerfile
```
FROM python:3.12-slim

# Create a non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set working directory
WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Make the main.py file executable
RUN chmod +x src/main.py

# Set the entrypoint to python and default command to show usage
ENTRYPOINT ["python", "main.py"]
```

TODO.md
```
chat: https://chat.deepseek.com/a/chat/s/0aa783f0-51fc-4e79-bd43-6221268b26a5

# Agents:

## Agent Architecture:
- MiddleWare:
  - SummarizationMiddleware: Done
  - TodoListMiddleware: Done
- Portfolio Manager: SKIP


## Context:
- Construct the context to be LLM-friendly content: Done
- Make tool result LLM-friendly in markdown format: Done
  - Us Market News: Done
  - Major ETF Price Statistics: Done
  - List Positions: Done
  

## Technical Implementation:
- in-database cache the results of the tools to avoid redundant calls: Done
- Async Cache Decorator: Done
- Refactor in-database cache to be decorator: Done
- Sync Agent Message to Sandx AI Monitoring Dashboard: Done
- Pylint: Done
- Yahoo Finance Fundamental Data API: Done
- Write to DB middleware: Done
- SES Send Email: Done

## Market Research Agent: Done
## Chief Investment Officer Agent: WIP
## Trader Agent: WIP


### Tools:
- Get the latest news from : https://tradingeconomics.com/united-states/news: Done
- Google Market Research: Done
- Optimize Google Market Research Prompt: Done
- Get the major ETF price statistics Done
- Get the latest stock price statistics: Done
- Get the most active stocks: Done
- Get Jina DeepSearch Tool: Not Impressive and Slow - Avoid using it
- BUY: Done
- SELL: Done
- Handoff Tool: Done


## Common Tools:
- Get positions with latest price and change: Done

## Deep Agent
- Learn About Langchain Deep Agent: Done

## Resources:
- https://github.com/The-Swarm-Corporation/AutoHedge: Less Useful
- https://docs.langchain.com/oss/python/langchain/supervisor

## Orchestration Design:
- Handoff Tool: Done

### Others
- Only Open Hour Can Run: TO Decide
- Try google gemini flash lite: TO Decide
- Add To Do Middleware To CIO: Done
- Display tool args: Done
- Try Different Strategy: Done
- Rationale when BUY/SELL: Done
- Add the records of the ticker of reviewed: Done
- Add the CIO summarization: Done
- Compile to be graph: Done
- Restore To Run: Done
- Deserialize the message to be langchain message: Done
- And Based on the prediction of price increase or decrease, decide whether to buy or sell the stock: TODO
- Dockerize the project: TODO
- Avoid revising the same ticker multiple times: TODO
```

pyproject.toml
```
[project]
name = "ai-trading-agents"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "alpaca-py>=0.43.2",
    "boto3>=1.42.9",
    "google-genai>=1.50.1",
    "langchain>=1.0.7",
    "langchain-deepseek>=1.0.1",
    "langchain-openai>=1.0.3",
    "loguru>=0.7.3",
    "markdownify>=1.2.2",
    "numpy>=2.3.5",
    "pre-commit>=4.4.0",
    "prisma>=0.15.0",
    "pytz>=2025.2",
    "tqdm>=4.67.1",
    "types-pytz>=2025.2.0.20251108",
    "types-tqdm>=4.67.0.20250809",
    "upstash-redis>=1.5.0",
    "yfinance>=0.2.66",
]

[dependency-groups]
dev = [
    "ipykernel>=7.1.0",
    "jupyter>=1.1.1",
    "ruff>=0.14.5",
]
[tool.pyright]
include = ["src", "main.py"]
exclude = [".venv/",
    "**/__pycache__",
    "uv.lock"
]
venvPath = "."
venv = ".venv"
```

requirements.txt
```
alpaca-py>=0.43.2
boto3>=1.42.9
google-genai>=1.50.1
langchain>=1.0.7
langchain-deepseek>=1.0.1
langchain-openai>=1.0.3
loguru>=0.7.3
numpy>=2.3.5
prisma>=0.15.0
pytz>=2025.2
tqdm>=4.67.1
upstash-redis>=1.5.0
yfinance>=0.2.66
markdownify>=1.2.2
```

schema.prisma
```
generator client {
    provider             = "prisma-client-py"
    recursive_type_depth = 10
}

datasource db {
    provider = "postgresql"
    url      = env("DATABASE_URL")
}

model User {
    id        String   @id @default(uuid())
    clerkId   String   @unique
    email     String   @unique
    imageUrl  String?
    firstName String?
    lastName  String?
    isDeleted Boolean?
    Bot       Bot[]
}

model Bot {
    id                         String                       @id @default(uuid())
    name                       String
    strategy                   String // prompt
    status                     BotStatus
    intervalInSecond           Int
    startHour                  Int?
    startMinute                Int?
    startTimeZone              String?
    active                     Boolean                      @default(false)
    lastRunAt                  DateTime?
    nextRunAt                  DateTime?
    userId                     String
    user                       User                         @relation(fields: [userId], references: [clerkId], onDelete: Cascade)
    public                     Boolean?                     @default(false)
    portfolio                  Portfolio?
    accumulatedPL              Float?
    accumulatedPLPercent       Float?
    llmModel                   String?                      @default("deepseek-v3.2")
    watchlist                  WatchlistItem[]
    depositWithdrawCash        DepositWithdrawCash[]
    trades                     Trade[]
    InitPortfolio              InitPortfolio?
    DailyPortfolioSnapshot     DailyPortfolioSnapshot[]
    InitDailyPortfolioSnapshot InitDailyPortfolioSnapshot[]
    QQQBenchmarkPointsCache    QQQBenchmarkPointsCache[]
    QQQInitShareCache          QQQInitShareCache?
    runs                       Run[]
    agentMessages              AgentMessage[]
    recommends                 Recommend[]
    createdAt                  DateTime                     @default(now())
    updatedAt                  DateTime                     @updatedAt

    @@index([userId])
    @@index([active])
    @@index([public])
}

enum BotStatus {
    STOPPED
    RUNNING
    IDLE
}

model Portfolio {
    id        String     @id @default(uuid())
    cash      Float      @default(0)
    botId     String     @unique
    bot       Bot        @relation(fields: [botId], references: [id], onDelete: Cascade)
    positions Position[]
    createdAt DateTime   @default(now())
    updatedAt DateTime   @updatedAt
}

model Position {
    id          String    @id @default(uuid())
    ticker      String
    volume      Float
    cost        Float
    portfolioId String
    portfolio   Portfolio @relation(fields: [portfolioId], references: [id], onDelete: Cascade)
    createdAt   DateTime  @default(now())
    updatedAt   DateTime  @updatedAt

    @@unique([portfolioId, ticker])
}

model InitPortfolio {
    id            String         @id @default(uuid())
    cash          Float          @default(0)
    botId         String         @unique
    bot           Bot            @relation(fields: [botId], references: [id], onDelete: Cascade)
    initPositions InitPosition[]
    createdAt     DateTime       @default(now())
    updatedAt     DateTime       @updatedAt
}

model InitPosition {
    id              String        @id @default(uuid())
    ticker          String
    volume          Float
    cost            Float
    portfolioId     String
    initPortfolio   InitPortfolio @relation(fields: [portfolioId], references: [id], onDelete: Cascade)
    createdAt       DateTime      @default(now())
    updatedAt       DateTime      @updatedAt
    initPortfolioId String?
}

model WatchlistItem {
    id     String @id @default(uuid())
    ticker String
    botId  String
    bot    Bot    @relation(fields: [botId], references: [id], onDelete: Cascade)
}

model DepositWithdrawCash {
    id        String   @id @default(uuid())
    botId     String
    bot       Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    amount    Float
    createdAt DateTime @default(now())
}

model WaitList {
    id        String   @id @default(uuid())
    email     String
    createdAt DateTime @default(now())

    @@unique([email])
}

model Ticker {
    id      String @id @default(uuid())
    cik_str String
    ticker  String
    title   String

    @@unique([ticker])
}

model DailyPortfolioSnapshot {
    id             String   @id @default(uuid())
    botId          String
    bot            Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    date           DateTime // Date at midnight (UTC), e.g., 2025-04-05T00:00:00.000Z
    portfolioValue Float
    positionsValue Float
    cash           Float
    createdAt      DateTime @default(now())
    updatedAt      DateTime @updatedAt

    @@unique([botId, date])
    @@index([botId])
    @@index([date])
}

model InitDailyPortfolioSnapshot {
    id             String   @id @default(uuid())
    botId          String
    bot            Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    date           DateTime // Date at midnight (UTC), e.g., 2025-04-05T00:00:00.000Z
    portfolioValue Float
    positionsValue Float
    cash           Float
    createdAt      DateTime @default(now())
    updatedAt      DateTime @updatedAt

    @@unique([botId, date])
    @@index([botId])
    @@index([date])
}

model QQQBenchmarkPointsCache {
    id        String   @id @default(uuid())
    botId     String
    bot       Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    points    String
    startISO  DateTime // Date at midnight (UTC), e.g., 2025-04-05T00:00:00.000Z
    endISO    DateTime
    createdAt DateTime @default(now())
    updatedAt DateTime @updatedAt

    @@unique([botId, startISO, endISO])
}

model QQQInitShareCache {
    id        String   @id @default(uuid())
    botId     String   @unique
    bot       Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    share     Float
    createdAt DateTime @default(now())
    updatedAt DateTime @updatedAt
}

enum RunStatus {
    SUCCESS
    FAILURE
    RUNNING
    STARTING
}

model Run {
    id            String         @id @default(uuid())
    botId         String
    bot           Bot            @relation(fields: [botId], references: [id], onDelete: Cascade)
    status        RunStatus
    instruction   String?
    summary       String?
    tickers       String?
    trades        Trade[]
    agentMessages AgentMessage[]
    createdAt     DateTime       @default(now())
    updatedAt     DateTime       @updatedAt
    completedAt   DateTime?
    recommends    Recommend[]

    @@index([botId, status])
}

enum Role {
    MARKET_ANALYST
    EQUITY_RESEARCH_ANALYST
    CHIEF_INVESTMENT_OFFICER
    RISK_ANALYST
    TRADING_EXECUTOR
    QUANTITATIVE_ANALYST
    PORTFOLIO_MANAGER
    FUNDAMENTAL_ANALYST
    USER
}

model AgentMessage {
    id        String   @id @default(uuid())
    role      Role
    botId     String
    bot       Bot      @relation(fields: [botId], references: [id], onDelete: Cascade)
    messages  String //json: list[AIMessage|HumanMessage|ToolMessage]: Change to json of AIMessage|HumanMessage|ToolMessage
    createdAt DateTime @default(now())
    updatedAt DateTime @updatedAt
    run       Run      @relation(fields: [runId], references: [id], onDelete: Cascade)
    runId     String
}

model Trade {
    id                  String    @id @default(uuid())
    type                TradeType
    price               Float
    ticker              String
    amount              Float
    run                 Run       @relation(fields: [runId], references: [id], onDelete: Cascade)
    runId               String
    bot                 Bot       @relation(fields: [botId], references: [id], onDelete: Cascade)
    botId               String
    rationale           String
    confidence          Float
    realizedPL          Float? // Only applicable for sell trades
    realizedPLPercent   Float? // Only applicable for sell trades
    unrealizedPL        Float? // Only applicable for buy trades
    unrealizedPLPercent Float? // Only applicable for buy trades
    createdAt           DateTime  @default(now())
}

model Recommend {
    id         String    @id @default(uuid())
    type       TradeType
    ticker     String
    amount     Float
    rationale  String
    confidence Float
    role       Role
    run        Run       @relation(fields: [runId], references: [id], onDelete: Cascade)
    runId      String
    bot        Bot       @relation(fields: [botId], references: [id], onDelete: Cascade)
    botId      String
    createdAt  DateTime  @default(now())
}

enum TradeType {
    SELL
    BUY
    HOLD
}

model Cache {
    id        String   @id @default(uuid())
    function  String
    key       String
    content   String
    createdAt DateTime @default(now())
    expiresAt DateTime

    @@index([function, key, expiresAt])
}
```

scripts/clear_notebook_outputs.py
```
#!/usr/bin/env python3
"""
Script to clear outputs from Jupyter notebooks.
Used by git pre-commit hook to ensure clean notebooks are committed.
"""

import json
import sys
import os


def clear_notebook_outputs(notebook_path):
    """Clear all outputs from a Jupyter notebook."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Iterate through all cells and clear outputs
    for cell in notebook.get("cells", []):
        if cell["cell_type"] == "code":
            # Clear outputs
            cell["outputs"] = []
            # Reset execution count
            cell["execution_count"] = None

    # Write back to the file
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)  # Use indent=1 for Jupyter's standard
        f.write("\n")  # Ensure a final newline


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python clear_notebook_outputs.py <notebook_path> [notebook_path...]",
            file=sys.stderr,
        )
        sys.exit(1)

    notebook_paths = sys.argv[1:]

    for path in notebook_paths:
        if not os.path.exists(path):
            print(f"File does not exist: {path}", file=sys.stderr)
            continue

        if not path.endswith(".ipynb"):
            print(f"Skipping non-notebook file: {path}", file=sys.stderr)
            continue

        try:
            clear_notebook_outputs(path)
            print(f"Cleared outputs from: {path}")
        except Exception as e:
            print(f"Error processing {path}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
```

src/context.py
```
import json
from prisma.types import AgentMessageWhereInput
from prisma.models import Run
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from src.db import prisma
from src.typings.context import Context
from src.utils import async_retry

message_type_map = {"ai": AIMessage, "human": HumanMessage, "tool": ToolMessage}


@async_retry(silence_error=False)
async def build_context(run: Run):
    bot = await prisma.bot.find_unique(
        where={"id": run.botId},
        include={
            "user": True,
            "portfolio": {"include": {"positions": True}},
            "watchlist": True,
            "trades": True,
        },
    )
    if not bot:
        raise ValueError(f"Bot with ID {run.botId} not found.")

    if not bot.active:
        raise ValueError(f"Bot with ID {bot.id} is not active.")

    context = Context(run=run, bot=bot, llm_model=bot.llmModel or "deepseek-v3.2")

    return context


@async_retry(silence_error=False)
async def restore_messages(run_id: str) -> list[BaseMessage] | None:
    agent_messages = await prisma.agentmessage.find_many(
        where={"runId": run_id}, order={"createdAt": "asc"}
    )

    if len(agent_messages) == 0:
        return

    #  only Retry CIO messages
    cio_agent_msgs = [
        agt_msg
        for agt_msg in agent_messages
        if agt_msg.role == "CHIEF_INVESTMENT_OFFICER"
    ]

    deserialized_messages = [json.loads(agt_msg.messages) for agt_msg in cio_agent_msgs]

    last_msg = deserialized_messages[-1]
    while last_msg["type"] == "ai":
        deserialized_messages.pop(-1)
        last_msg = deserialized_messages[-1]

    last_msg = deserialized_messages[-1]

    timestamp = cio_agent_msgs[-1].createdAt

    #  delete the unfinished messages after the latest CIO message
    await prisma.agentmessage.delete_many(
        where=AgentMessageWhereInput(runId=run_id, createdAt={"gt": timestamp})
    )

    serialized_messages = [
        message_type_map[msg["type"]](**msg) for msg in deserialized_messages
    ]

    return serialized_messages
```

src/db.py
```
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
        "messages": dict[str, Any],
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
```

src/main.py
```
import os
import sys
import traceback
import asyncio
from loguru import logger
from prisma.enums import RunStatus
from langchain_core.messages import HumanMessage

from src import secrets
from src import db
from src.context import build_context, restore_messages
from src.agents.chief_investment_officer.agent import (
    build_chief_investment_officer_agent,
)


SECRETS = secrets.load()

ENV = os.environ.get("ENV", "dev")

DEFAULT_USER_PROMPT = """ As AI Agentic Chief Investment Officer,Now,you're tasked to review your portfolio performance,
identify new opportunities, and recommend appropriate actions to optimize portfolio performance aligned with the user's strategy by orchestrating different investment agents.
Now, please review the user's strategy and portfolio performance, and provide your recommendations working with your teams of investment agents.
"""


async def run_agent(run_id: str):
    run = await db.prisma.run.find_unique(where={"id": run_id})

    if not run:
        logger.error(f"Run {run_id} not found")
        exit(2)

    if run.status == "SUCCESS":
        logger.error(f"Run {run_id} is finished")
        exit(2)

    instruction = run.instruction or DEFAULT_USER_PROMPT
    start_messages = [HumanMessage(content=instruction)]
    deserialized_messages = await restore_messages(run_id)

    if deserialized_messages == "ERROR":
        logger.error(f"Failed to restore messages for run {run_id}")
        exit(2)

    if deserialized_messages:
        start_messages = deserialized_messages

    await db.prisma.run.update(where={"id": run_id}, data={"status": RunStatus.RUNNING})

    context = await build_context(run=run)

    if context == "ERROR":
        logger.error(f"Failed to build context for run {run_id}")
        exit(2)

    agent_graph = await build_chief_investment_officer_agent(context)

    events = agent_graph.astream(
        input={
            "messages": start_messages,  # type: ignore
        },
        context=context,
        stream_mode="values",
    )
    async for event in events:
        if "messages" in event:
            message = event["messages"][-1]
            message.pretty_print()

    await db.prisma.run.update(where={"id": run_id}, data={"status": RunStatus.SUCCESS})


async def main(run_id: str):
    try:
        await db.connect()
        await run_agent(run_id)
    except Exception as e:
        run = await db.prisma.run.find_unique(where={"id": run_id})
        if run:
            await db.prisma.run.update(
                where={"id": run_id}, data={"status": RunStatus.FAILURE}
            )
        logger.error(
            f"Failed to run agent {run_id}: {e}. Traceback: {traceback.format_exc()}"
        )
    finally:
        await db.disconnect()
        exit(1)


if __name__ == "__main__":
    runId = sys.argv[1]
    asyncio.run(main(runId))
```

src/models.py
```
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from src.utils import get_env

OPENAI_API_KEY = SecretStr(get_env("OPENAI_API_KEY"))
OPENAI_API_URL = get_env("OPENAI_API_URL")


def get_model(model_name: str):
    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_URL,
    )
```

src/agents/__init__.py
```
from src.agents.market_analyst.agent import build_market_analyst_agent
from src.agents.fundamental_analyst.agent import build_fundamental_analyst_agent
from src.agents.risk_analyst.agent import build_risk_analyst_agent
from src.agents.equity_research_analyst.agent import build_equity_research_analyst_agent
from src.agents.trading_executive.agent import build_trading_executor_agent
from src.agents.chief_investment_officer.agent import (
    build_chief_investment_officer_agent,
)

__all__ = [
    "build_trading_executor_agent",
    "build_market_analyst_agent",
    "build_fundamental_analyst_agent",
    "build_risk_analyst_agent",
    "build_equity_research_analyst_agent",
    "build_chief_investment_officer_agent",
]
```

src/middleware/__init__.py
```
import asyncio
import json
from loguru import logger
from datetime import datetime, timezone
from langchain_core.messages import AnyMessage
from langgraph.runtime import Runtime
from langchain.agents import middleware
from prisma.types import (
    AgentMessageCreateInput,
)
from src import db
from src.db import CachedAgentMessage, CACHED_AGENTS_MESSAGES
from src.typings.context import Context
from src.models import get_model
from src.utils.constants import ALL_ROLES
from src.typings.agent_roles import AgentRole
from src.utils import async_retry

PERSISTED_MSG_IDS: set[str] = set()

langchain_model = get_model("deepseek")


summarization_middleware = middleware.SummarizationMiddleware(
    model=langchain_model,
    max_tokens_before_summary=128_000,
    messages_to_keep=20,
)


todo_list_middleware = middleware.TodoListMiddleware()


class LoggingMiddleware(middleware.AgentMiddleware[middleware.AgentState, Context]):
    def __init__(self, role: AgentRole):
        assert role in ALL_ROLES
        self.role: AgentRole = role

    #  Only one agent called this middleware at a time
    async def aafter_model(
        self, state: middleware.AgentState, runtime: Runtime[Context]
    ) -> None:
        context = runtime.context
        messages = state["messages"]
        # Fire-and-forget: schedule persistence in the background without awaiting

        asyncio.create_task(
            persist_agent_messages(  # type: ignore
                role=self.role,
                bot_id=context.bot.id,
                run_id=context.run.id,
                messages=messages,
            )
        )

        await cache_agent_messages(
            role=self.role,
            bot_id=context.bot.id,
            run_id=context.run.id,
            messages=messages,
        )


@async_retry()
async def persist_agent_message(
    role: AgentRole, bot_id: str, run_id: str, message: AnyMessage
):
    if not message.id:
        return

    if message.id in PERSISTED_MSG_IDS:
        return

    existed_message = await db.prisma.agentmessage.find_unique(
        where={"id": message.id},
    )

    if not existed_message:
        await db.prisma.agentmessage.create(
            data=AgentMessageCreateInput(
                id=message.id,
                role=role,
                botId=bot_id,
                messages=message.model_dump_json(),
                runId=run_id,
            )
        )
    PERSISTED_MSG_IDS.add(message.id)


async def persist_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    for msg in messages:
        await persist_agent_message(
            role=role,
            bot_id=bot_id,
            run_id=run_id,
            message=msg,
        )


@async_retry()
async def cache_agent_messages(
    role: AgentRole, bot_id: str, run_id: str, messages: list[AnyMessage]
):
    # Save and consolidated agent messages to local file because cross agent with different message states

    existing_msg_ids = set(msg["id"] for msg in CACHED_AGENTS_MESSAGES if msg["id"])

    new_agent_messages = [
        CachedAgentMessage(
            id=msg.id,
            role=role,
            botId=bot_id,
            runId=run_id,
            createdAt=(datetime.now(timezone.utc)).isoformat(),
            updatedAt=(datetime.now(timezone.utc)).isoformat(),
            messages=msg.model_dump(),
        )
        for msg in messages
        if msg.id and msg.id not in existing_msg_ids
    ]

    CACHED_AGENTS_MESSAGES.extend(new_agent_messages)

    content = json.dumps(CACHED_AGENTS_MESSAGES)

    is_success = await db.redis.set(
        key=f"agent_messages:{run_id}", value=content, ex=60 * 60 * 24
    )

    if not is_success:
        logger.warning(f"Failed to cache agent messages for run_id: {run_id}")
```

src/prompt/__init__.py
```
from src.prompt.background import SANDX_AI_INTRODUCTION
from src.prompt.roles import RECOMMENDATION_PROMPT, ROLE_PROMPTS_MAP
from src.prompt.system import build_agent_system_prompt

__all__ = [
    "SANDX_AI_INTRODUCTION",
    "RECOMMENDATION_PROMPT",
    "ROLE_PROMPTS_MAP",
    "build_agent_system_prompt",
]
```

src/prompt/background.py
```
SANDX_AI_INTRODUCTION = (
    "The sandx.ai is an Agentic AI US Stock Sandbox Platform that empowers investors to experiment, learn, and refine "
    "their investment strategies through an ensemble of specialized AI agents. By simulating real-market dynamics "
    "with zero financial risk, users gain actionable insights, test hypotheses, and build confidence before deploying "
    "capital in live markets."
)
```

src/prompt/roles.py
```
from typing import TypedDict
from prisma.enums import Role
from src.typings.agent_roles import SubAgentRole


class AgentDescription(TypedDict):
    title: str
    description: str
    key_capabilities: list[str]
    strength_weight: float


# Agent descriptions for reference in the CIO prompt
AGENT_DESCRIPTIONS: dict[SubAgentRole, AgentDescription] = {
    "MARKET_ANALYST": {
        "title": "Market Analyst",
        "description": "Senior US-market analyst who delivers concise, actionable briefings on market catalysts, drivers, and sentiment inflections.",
        "key_capabilities": [
            "Overnight and breaking headline catalysts",
            "Key macro, sector, and single-stock drivers",
            "Imminent event risk (earnings, Fed speakers, data releases)",
            "Cross-asset flow and sentiment inflections",
        ],
        "strength_weight": 0.25,  # Market context is crucial
    },
    "RISK_ANALYST": {
        "title": "Risk Analyst",
        "description": "Meticulous risk analyst who quantifies downside scenarios, stress-tests portfolios, and designs hedging frameworks.",
        "key_capabilities": [
            "Quantifies downside scenarios and potential losses",
            "Stress-tests portfolios under various market conditions",
            "Designs hedging frameworks for risk mitigation",
            "Assesses tail events and regulatory constraints",
        ],
        "strength_weight": 0.2,  # Risk management is equally important
    },
    "EQUITY_RESEARCH_ANALYST": {
        "title": "Equity Research Analyst",
        "description": "Senior equity research analyst focused on catalysts, drivers, and market inflections for specific securities.",
        "key_capabilities": [
            "Catalyst-driven analysis of specific equities",
            "Sector and thematic research",
            "Event risk assessment for individual stocks",
            "Market sentiment and flow analysis for securities",
        ],
        "strength_weight": 0.5,  # Catalysts and timing are important
    },
    "FUNDAMENTAL_ANALYST": {
        "title": "Fundamental Analyst",
        "description": "Fundamental equity analyst who builds conviction from first principles using comprehensive financial metrics.",
        "key_capabilities": [
            "Data sanity checks and anomaly detection",
            "Extracts 5-7 high-signal insights with metric labels and values",
            "Assesses quality, durability, margins, returns, cash conversion",
            "Evaluates valuation vs growth using DCF/comps",
            "Analyzes capital returns and payout sustainability",
            "Outlines key catalysts and risks with monitoring indicators",
        ],
        "strength_weight": 0.30,  # Fundamentals provide the valuation anchor
    },
    "TRADING_EXECUTOR": {
        "title": "Trading Executor",
        "description": "Trading executor who executes trades based on instructions from the Chief Investment Officer.",
        "key_capabilities": [
            "Verify watchlist/position, market hours, cash, holdings",
            "Execute BUY/SELL exactly as instructed",
            "Confirm trade booked, cash/position updated",
            "Ensure cash sufficiency for buys",
            "Never short-sell (≤ current holdings)",
        ],
        "strength_weight": 0,  # Execution role, lower weight in decision synthesis
    },
}

RECOMMENDATION_PROMPT: str = "For each analysis, you should frame your final recommendation and state BUY, SELL, or HOLD with your rationale, and confidence level (0.0-1.0)"


AGENT_TEAM_DESCRIPTION: str = "## YOUR INVESTMENT TEAM:\n\n" + "\n".join(
    f"### {idx}. {AGENT_DESCRIPTIONS[role]['title']}\n"
    f"**Description:** {AGENT_DESCRIPTIONS[role]['description']}\n"
    f"**Capabilities:** {AGENT_DESCRIPTIONS[role]['key_capabilities']}\n"
    f"**Default Strength weight in decisions:** {int(AGENT_DESCRIPTIONS[role]['strength_weight'] * 100)}%\n"
    for idx, role in enumerate(AGENT_DESCRIPTIONS.keys(), start=1)
)

CHIEF_INVESTMENT_OFFICER_ROLE_PROMPT: str = (
    "## CHIEF INVESTMENT OFFICER - STRATEGIC ORCHESTRATOR ##\n\n"
    "You are the CIO of Sandx AI, the conductor of a world-class investment team. "
    "Your expertise is not in doing the analysis yourself only, but also in orchestrating your team by assigning tasks to each teammates "
    "with clear instructions effectively to deliver superior investment recommendation and actions (BUY/SELL/HOLD) through strategic coordination.\n\n"
    "Here are the steps or framework to follow for performing scheduled regular tasks:\n"
    "1. Firstly the investment recommendation should start with the market analysis perform by the market analyst agent.\n"
    "2. Then, based on the market analysis and current portfolio, previous tickers reviewed in the last 7 analysis to avoid reviewing the same tickers,"
    "you should decide which 1-3 equities, or the tickers that user specified to focus on for the next analysis.\n"
    "3. For each ticker, delegate analysis to the analyst below and request a BUY/SELL/HOLD recommendation by following below workflow:\n"
    "3.1 Equity Research Analyst -> Fundamental Analyst -> Risk Analyst \n"
    "3.2 Based on the equity research analysis, fundamental analysis, and risk analysis, and their investment recommendation, "
    "you should provide comprehensive investment recommendations summary to BUY/SELL/HOLD action with rationale for the ticker.\n"
    "3.3 After that, you should handoff the recommended actions (BUY/SELL/HOLD) with rationale for all tickers to the trading executor to execute if the market is open.\n"
    "3.4 Finally, with consolidated investment rationales from all analysts, "
    "you should send the comprehensive well-styled html-based investment recommendation summary email with final executed actions and rationale for all tickers to the user.\n"
) + f"{AGENT_TEAM_DESCRIPTION}"


RolePromptMap = dict[Role, str]

ROLE_PROMPTS_MAP: RolePromptMap = {
    Role.MARKET_ANALYST: (
        "You are a senior US-market analyst on the Sandx AI investment desk. You report to the Chief Investment Officer. "
        "Leverage every available data source to deliver a concise, actionable briefing that captures: "
        "1) Overnight and breaking headline catalysts, "
        "2) Key macro, sector, and single-stock drivers, "
        "3) Imminent event risk (earnings, Fed speakers, data releases), "
        "4) Cross-asset flow and sentiment inflections. "
        "Synthesize into a single paragraph prioritizing highest-conviction opportunities and clear risk flags."
    ),
    Role.EQUITY_RESEARCH_ANALYST: (
        "You are a senior equity research analyst on the Sandx AI investment desk. You report to the Chief Investment Officer. "
        "TOOLS: do_google_equity_research(ticker), get_latest_equity_news(symbol). "
        "Analyze one ticker at a time; focus solely on the requested symbol. "
        "Protocol: 1) Start with get_latest_equity_news(symbol) to capture the freshest headlines and company-specific events; "
        "2) Run do_google_equity_research(ticker) to synthesize the current equity narrative, key drivers, risks, and opportunities; "
        "3) Deliver a decision-ready brief for the specified ticker"
    ),
    Role.CHIEF_INVESTMENT_OFFICER: CHIEF_INVESTMENT_OFFICER_ROLE_PROMPT,
    Role.RISK_ANALYST: (
        "You are a data-driven risk analyst who transforms raw market, fundamental, and macro data into risk analytics, volatility-adjusted position limits, and early-warning report. "
        "You report to the Chief Investment Officer. "
    ),
    Role.FUNDAMENTAL_ANALYST: (
        "You are a fundamental equity analyst who builds conviction from first principles. You report to the Chief Investment Officer. "
        "Use the provided markdown tables of fundamentals (Valuation, Profitability & Margins, Financial Health & Liquidity, "
        "Growth, Dividend & Payout, Market & Trading Data, Analyst Estimates, Company Info, Ownership & Shares, Risk & Volatility, "
        "Technical Indicators, Additional Financial Metrics) to produce a decision-ready thesis. "
    ),
    Role.TRADING_EXECUTOR: (
        "You are the Sandx AI Trading Executor. You report to the CIO and execute only on their explicit instructions.\n"
        "TOOLS: buy_stock(), sell_stock()\n"
        "PROTOCOL:\n"
        "1. Verify: watchlist/position, market hours, cash, holdings\n"
        "2. Execute: BUY/SELL exactly as instructed\n"
        "3. Confirm: trade booked, cash/position updated\n"
        "RULES:\n"
        "- Trade only watchlist or current positions\n"
        "- Markets closed weekends/holidays\n"
        "- Cash sufficiency for buys\n"
        "- Never short-sell (≤ current holdings)"
    ),
    Role.USER: (
        "You are an intellectually curious investor eager to understand how markets function, why prices move, and how "
        "professional-grade analysis can improve your decision-making. You ask incisive questions, challenge assumptions, "
        "and actively apply lessons learned in the sandbox to your real-world investment journey."
    ),
}


__all__ = ["RECOMMENDATION_PROMPT", "ROLE_PROMPTS_MAP", "AGENT_TEAM_DESCRIPTION"]
```

src/prompt/system.py
```
from prisma.enums import Role

from src.context import Context
from src.prompt import ROLE_PROMPTS_MAP
from src.tools_adaptors import ListPositionsAct


async def build_agent_system_prompt(context: Context, role: Role) -> str:
    user = context.bot.user
    if not user:
        raise ValueError("User not found.")

    tickers = context.run.tickers

    tickers = (
        f"Here's the tickers that user specified, you can trade only on these stocks: {tickers}"
        if tickers
        else ""
    )

    watchlist = context.bot.watchlist or []

    watchlist_prompt: str = (
        (
            "Here's the watchlist of user, you can trade only on these stocks or stock in the current positions:"
            ", ".join([w.ticker for w in watchlist])
        )
        if tickers
        else ""
    )

    positions_markdown = await ListPositionsAct().arun(bot_id=context.bot.id)
    # performance_narrative = await PortfolioPerformanceAnalysisAct().arun(
    #     bot_id=context.bot.id
    # )
    sections = [
        # SANDX_AI_INTRODUCTION,
        ROLE_PROMPTS_MAP[role],
        # user_name,
        tickers,
        watchlist_prompt,
        positions_markdown,
        # performance_narrative,
    ]
    return "\n\n".join([s for s in sections if s])
```

src/services/utils.py
```
import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, TypeVar, cast
import httpx
from loguru import logger
from prisma import types
from src.db import prisma, redis
import traceback

T = TypeVar("T")


class APIError(RuntimeError):
    pass


def extract_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str):
                return msg
    except Exception:
        # Fallback to text if JSON parsing fails
        pass
    text = response.text
    return text if text else None


def _extract_error_message(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            msg = payload.get("message")
            if isinstance(msg, str):
                return msg
    except Exception:
        # Fallback to text if JSON parsing fails
        pass
    text = response.text
    return text if text else None


def in_db_cache(
    function_name: str, ttl: int
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async decorator to cache function results in the database.

    - Stores results in the `Cache` table with a unique key derived from args/kwargs.
    - Respects TTL (seconds) via `expiresAt` and returns cached content when valid.

    Args:
        function_name: Logical function identifier to namespace cache keys.
        ttl: Time-to-live in seconds for the cached entry.

    Returns:
        A decorator that caches the async function's JSON-serializable result.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            try:
                pos_args = args[1:] if len(args) > 0 else args
                if "start" in kwargs:
                    start = kwargs["start"]
                    kwargs["start"] = datetime.fromisoformat(
                        start.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")

                if "end" in kwargs:
                    end = kwargs["end"]
                    kwargs["end"] = datetime.fromisoformat(
                        end.replace("Z", "+00:00")
                    ).strftime("%Y-%m-%d")

                if "symbols" in kwargs:
                    symbols = kwargs["symbols"]
                    kwargs["symbols"] = sorted(symbols)

                key_payload = {"args": pos_args, "kwargs": kwargs}
                cache_key = json.dumps(key_payload, sort_keys=True)
                now = datetime.now(timezone.utc)

                # Try existing unexpired cache
                existing = await prisma.cache.find_first(
                    where={
                        "function": function_name,
                        "key": cache_key,
                        "expiresAt": {"gt": now},
                    }
                )
                if existing is not None and isinstance(existing.content, str):
                    logger.info(f"Cache hit for {function_name} with key {cache_key}")
                    return cast(T, json.loads(existing.content))

                # Compute fresh result
                result = await func(*args, **kwargs)
                expires_at = now + timedelta(seconds=ttl)

                # Upsert by (function, key) if present; otherwise create
                any_existing = await prisma.cache.find_first(
                    where={
                        "function": function_name,
                        "key": cache_key,
                    }
                )
                if any_existing is not None:
                    await prisma.cache.update(
                        where={"id": any_existing.id},
                        data={
                            "content": json.dumps(result),
                            "expiresAt": expires_at,
                        },
                    )
                else:
                    await prisma.cache.create(
                        data=types.CacheCreateInput(
                            function=function_name,
                            key=cache_key,
                            content=json.dumps(result),
                            expiresAt=expires_at,
                        )
                    )
                return result

            except Exception as e:
                logger.error(
                    f"Error in {function_name} with args {args} and kwargs {kwargs}: {e} Traceback: {traceback.format_exc()}"
                )
                raise

        return wrapper

    return decorator


def redis_cache(
    function_name: str, ttl: int
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Async decorator to cache function results in the database.

    - Stores results in Redis with a unique key derived from args/kwargs.
    - Respects TTL (seconds) via `expiresAt` and returns cached content when valid.

    Args:
        function_name: Logical function identifier to namespace cache keys.
        ttl: Time-to-live in seconds for the cached entry.

    Returns:
        A decorator that caches the async function's JSON-serializable result.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            # print(
            #     f"function_name:{function_name} args:{args}, kwargs:{kwargs}")
            if "start" in kwargs:
                start = kwargs["start"]
                kwargs["start"] = datetime.fromisoformat(
                    start.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")

            if "end" in kwargs:
                end = kwargs["end"]
                kwargs["end"] = datetime.fromisoformat(
                    end.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")

            if "symbols" in kwargs:
                symbols = kwargs["symbols"]
                kwargs["symbols"] = sorted(symbols)

            key_payload = {
                "args": sorted([str(a) for a in args]),
                "kwargs": kwargs,
                "function_name": function_name,
            }
            cache_key = json.dumps(key_payload, sort_keys=True)

            existing = await redis.get(cache_key)

            if existing:
                logger.info(f"Cache Redis hit for {function_name} with key {cache_key}")
                return cast(T, json.loads(existing))

            # Compute fresh result
            result = await func(*args, **kwargs)
            await redis.set(cache_key, json.dumps(result), ex=ttl)
            return result

        return wrapper

    return decorator


def async_retry_on_status_code(
    base_delay: float = 1,
    max_retries: int = 5,
    status_codes: list[int] = [],
    max_delay_seconds: float = 30.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if (
                        len(status_codes) == 0 or e.response.status_code in status_codes
                    ) and retries < max_retries:
                        retries += 1
                        delay = min(
                            max_delay_seconds, base_delay * (2 ** (retries - 1))
                        )
                        logger.info(
                            f"Retrying in {delay}s due to status code: {e.response.status_code}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        last_error_message = _extract_error_message(e.response) or str(
                            e
                        )
                        logger.error(last_error_message)
                        raise APIError(last_error_message)

        return wrapper

    return decorator


__all__ = ["redis_cache", "async_retry_on_status_code", "APIError", "in_db_cache"]
```

src/tools/__init__.py
```
from src.tools.news_tools import get_latest_market_news, get_latest_equity_news

from src.tools.stock_tools import (
    get_etf_live_historical_price_change,
    get_stock_live_historical_price_change,
    get_most_active_stockers,
    get_latest_quotes,
    get_latest_quote,
    get_price_trend,
)
from src.tools.portfolio_tools import (
    list_current_positions,
    get_portfolio_performance_analysis,
)
from src.tools.research_tools import (
    do_google_market_research,
    do_google_equity_research,
)
from src.tools.common_tools import (
    get_user_investment_strategy,
    send_summary_email_tool,
    write_summary_report,
    get_historical_reviewed_tickers,
)

from src.tools.fundamental_data_tools import (
    get_fundamental_data,
)
from src.tools.risk_tools import (
    get_fundamental_risk_data,
    get_volatility_risk_indicators,
    get_price_risk_indicators,
)
from src.tools.trading_tools import (
    buy_stock,
    sell_stock,
    get_market_status,
    get_recommend_stock_tool,
    get_analysts_recommendations,
    write_down_tickers_to_review,
)

# avoid circular import
# from src.tools.handoff_tools import handoff_to_specialist

__all__ = [
    "get_price_trend",
    "buy_stock",
    "sell_stock",
    "get_recommend_stock_tool",
    "get_analysts_recommendations",
    "send_summary_email_tool",
    "write_summary_report",
    "get_market_status",
    "get_latest_quotes",
    "get_latest_quote",
    "get_latest_market_news",
    "get_latest_equity_news",
    "list_current_positions",
    "get_etf_live_historical_price_change",
    "get_stock_live_historical_price_change",
    "get_most_active_stockers",
    "get_portfolio_performance_analysis",
    "do_google_market_research",
    "do_google_equity_research",
    "get_user_investment_strategy",
    "get_fundamental_data",
    "get_fundamental_risk_data",
    "get_volatility_risk_indicators",
    "get_price_risk_indicators",
    "write_down_tickers_to_review",
    "get_historical_reviewed_tickers",
    # "handoff_to_specialist", # Avoid circular import
]
```

src/tools/common_tools.py
```
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage
from src.utils.message import combine_ai_messages
from src.context import Context

from src.tools_adaptors import (
    GetUserInvestmentStrategyAct,
    WriteInvestmentReportEmailAct,
    SendInvestmentReportEmailAct,
    GetHistoricalReviewedTickersAct,
)


get_user_investment_strategy_act = GetUserInvestmentStrategyAct()
send_investment_report_email_act = SendInvestmentReportEmailAct()
write_investment_report_email_act = WriteInvestmentReportEmailAct()
get_historical_reviewed_tickers_act = GetHistoricalReviewedTickersAct()


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
        llm_model=context.llm_model,
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


@tool(get_historical_reviewed_tickers_act.name)
async def get_historical_reviewed_tickers(runtime: ToolRuntime[Context]):
    """
    Get the latest 7 analysis's tickers that have been reviewed for avoiding reviewing the same tickers again.
    """
    return await get_historical_reviewed_tickers_act.arun(runtime.context.bot.id)
```

src/tools/fundamental_data_tools.py
```
from langchain.tools import tool
from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.fundamental_data import (
    FundamentalDataAct,
)

fundamental_act = FundamentalDataAct()


@tool(fundamental_act.name)
async def get_fundamental_data(ticker: str):
    """Get fundamental data for a given ticker.

    Returns a markdown report structured into decision-ready sections:
    - Valuation Metrics
    - Profitability & Margins
    - Financial Health & Liquidity
    - Growth Metrics
    - Dividend & Payout
    - Market & Trading Data
    - Analyst Estimates & Ratings
    - Company Information
    - Ownership & Shares
    - Risk & Volatility
    - Technical Indicators

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    ticker = ticker.upper().strip()
    is_valid = await is_valid_ticker(ticker)
    if not is_valid:
        return f"{ticker} is an invalid ticker symbol"
    return await fundamental_act.arun(ticker)
```

src/tools/handoff_tools.py
```
from typing import List, Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import BaseMessage
from src.context import Context
from src import agents
from src.typings.agent_roles import SubAgentRole
from src.typings.langgraph_agents import LangGraphAgent
from src.utils.message import combine_ai_messages
from src.utils.constants import SPECIALIST_ROLES


agent_building_map = {
    "MARKET_ANALYST": agents.build_market_analyst_agent,
    "RISK_ANALYST": agents.build_risk_analyst_agent,
    "EQUITY_RESEARCH_ANALYST": agents.build_equity_research_analyst_agent,
    "FUNDAMENTAL_ANALYST": agents.build_fundamental_analyst_agent,
    "TRADING_EXECUTOR": agents.build_trading_executor_agent,
}


@tool("handoff_to_specialist")
async def handoff_to_specialist(
    role: Annotated[
        SubAgentRole,
        f"Role of the specialist agent to handoff to. Must be one of: {' | '.join(SPECIALIST_ROLES)}",
    ],
    task: Annotated[
        str,
        "A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.",
    ],
    runtime: ToolRuntime[Context],
) -> str:
    """
    Delegate a specific investment-related task to a specialist agent and return the agent's response.

    This function:
    1. Validates that the requested role is one of the registered specialist roles.
    2. Combines the agent's response messages into a single string and returns it.

    The returned string consolidates all AI-generated messages from the specialist, providing
    a concise summary or detailed output depending on the task performed.

    Args:
        role: Role of the specialist agent to handoff to.
        task: A clear, detailed description of the task the specialist should perform. Include relevant context, and expected deliverables to ensure the specialist can act effectively.

    Returns:
        A single string containing the combined AI messages from the specialist agent's response.

    Raises:
        ValueError: If the provided role is not in the list of valid specialist roles.
    """
    context = runtime.context
    if role not in agent_building_map:
        return f"Invalid role: {role}. Must be one of: {SPECIALIST_ROLES}"

    agent: LangGraphAgent = await agent_building_map[role](context)

    response = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Task from the chief investment officer to you: {task}",
                }
            ]
        },
        context=context,
    )

    messages: List[BaseMessage] = response["messages"]

    content = combine_ai_messages(messages)

    return content


__all__ = ["handoff_to_specialist"]
```

src/tools/news_tools.py
```
from langchain.tools import tool
from src.utils.ticker import is_valid_ticker
from src.services.tradingeconomics.api_market_news import News
from src.tools_adaptors.news import MarketNewsAct, EquityNewsAct

market_news_action = MarketNewsAct()
equity_news_action = EquityNewsAct()


def convert_news_to_markdown_text(news: News):
    title = news["title"]
    markdown_text = f"### {title}\n"
    markdown_text += (
        f"Importance: {news['importance']} Date: {news['date']}  {news['time_ago']} "
    )
    markdown_text += f"Expiration: {news['expiration']} \n\n"
    markdown_text += f"{news['description']}\n"
    return markdown_text


@tool(market_news_action.name)
async def get_latest_market_news():
    """
    Retrieve the most recent market news headlines and summaries for the United States.

    This tool is useful for:
    - Keeping track of breaking economic events that may move markets
    - Gathering quick context before making trading or investment decisions
    - Monitoring scheduled data releases and their market impact
    - Staying informed on Fed policy hints, inflation updates, employment figures, and other macro drivers
    - Comparing the relative importance (high/medium/low) of each news item
    - Filtering news by expiration date to focus only on still-relevant stories

    Returns a markdown-formatted string containing the title, importance level, date, expiration, and description of each news item.
    """

    news = await market_news_action.arun()
    if news == "ERROR":
        return "Unknown error to get market news."

    # Convert each news item to markdown text
    markdown_news = "\n\n".join(convert_news_to_markdown_text(n) for n in news)

    # markdown_news = utils.dicts_to_markdown_table(news)
    heading = "## United States Market News"
    return heading + "\n" + markdown_news


@tool(equity_news_action.name)
async def get_latest_equity_news(symbol: str):
    """
    Retrieve the most recent equity news headlines and summaries for the United States.

    This tool is useful for:
    - Keeping track of breaking news that may affect specific equities
    - Gathering quick context before making trading or investment decisions
    - Monitoring scheduled data releases and their market impact
    - Staying informed on company-specific events, product launches, or regulatory changes

    Returns a markdown-formatted string containing the title, date, and content of each news item.
    """
    symbol = symbol.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"

    equity_news = await equity_news_action.arun(symbol)
    return equity_news


__all__ = ["get_latest_market_news", "get_latest_equity_news"]

if __name__ == "__main__":
    #  python -m src.tools.news_tools
    import asyncio

    result = asyncio.run(get_latest_market_news())  # type: ignore
    print(result)
```

src/tools/portfolio_tools.py
```
from langchain.tools import tool, ToolRuntime
from src.context import Context
from src.tools_adaptors.portfolio import (
    ListPositionsAct,
    PortfolioPerformanceAnalysisAct,
)

list_positions_act = ListPositionsAct()
portfolio_performance_analysis_act = PortfolioPerformanceAnalysisAct()


@tool(list_positions_act.name)
async def list_current_positions(runtime: ToolRuntime[Context]):
    """
    Retrieve the current open positions for the trading portfolio, enriched with live market data.

    This tool fetches every active stock position held by the trading portfolio,
    augmenting each record with the latest market price, percentage change since open,
    and the total current market value.  The returned data is ordered by allocation
    (largest first) and can be used to monitor P&L, rebalance allocations, or feed
    downstream analytics, etc.

    Possible tool purposes:
    - Monitor real-time portfolio exposure and sector weightings.
    - Generate daily P&L snapshots for compliance or client reporting.
    - Feed position-level data into risk models (VaR, beta, concentration limits).
    - Detect drift from target allocations and trigger rebalancing alerts.
    - Provide input for tax-loss harvesting by flagging underwater positions.
    - Supply holdings to automated strategies that scale in/out based on allocation caps.
    - Quickly identify top gainers/losers by PnL or PnL percent for intraday reviews.
    - Screen for positions exceeding a PnL percent threshold to automate profit-taking or stop-loss.
    - Aggregate PnL across positions to compute total portfolio return on capital at risk.
    - Export position-level PnL data to Excel for deeper attribution analysis (sector, factor, etc.).
    - Flag large PnL swings outside expected volatility bands for risk-manager alerts.
    - Feed PnL percent into rebalancing algorithms that trim winners and add to laggards.
    - Generate client statements showing dollar and percent gain/loss per holding.
    - Compare realized vs. unrealized PnL to estimate upcoming tax impacts.
    - Provide chatbot answers like “Which positions are up more than 5 % today?” or
      “What is my total unrealized PnL?”

    Notes
    -----
    - Prices reflect the consolidated feed from the exchange with which the
      broker is connected; delays are typically < 500 ms during market hours.
    - CASH is a special position that represents the cash balance in the account.
    - Allocation percentages are computed against the sum of currentValue across
      **all** positions plus any cash held in the same account.
    - Each position now includes `pnl` (dollar profit/loss) and `pnl_percent` (return on cost basis).
    """

    bot_id = runtime.context.bot.id
    position_markdown = await list_positions_act.arun(bot_id=bot_id)
    return position_markdown


@tool(portfolio_performance_analysis_act.name)
async def get_portfolio_performance_analysis(runtime: ToolRuntime[Context]):
    """
    Analyze the performance of the trading portfolio over different periods.

    This tool calculates the total return, annualized return, maximum drawdown,
    and Sharpe ratio for the trading portfolio.  It provides a snapshot of the
    portfolio's risk-adjusted performance over the specified time period.

    Possible tool purposes:
    - Evaluate the historical performance of the trading strategy.
    - Compare the performance of different trading strategies or portfolios.
    - Identify periods of strong and weak performance.
    - Assess the riskiness of the trading strategy.
    - Guide decision-making on when to rebalance or exit the portfolio.

    Notes
    -----
    - The Sharpe ratio is computed using a risk-free rate of 0% for simplicity.
    - Maximum drawdown is the largest percentage decline from a peak to a trough.
    """

    bot_id = runtime.context.bot.id
    performance_analysis = await portfolio_performance_analysis_act.arun(bot_id=bot_id)
    return performance_analysis


__all__ = ["list_current_positions", "get_portfolio_performance_analysis"]
```

src/tools/research_tools.py
```
from langchain.tools import tool
from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.research import GoogleMarketResearchAct, GoogleEquityResearchAct

google_market_research_action = GoogleMarketResearchAct()
google_equity_research_action = GoogleEquityResearchAct()


@tool(google_market_research_action.name)
async def do_google_market_research():
    """
    Performs market research to synthesize and present the comprehensive current market narrative,
    key drivers, risks, and opportunities based on real-time data and recent news using Google’s grounded LLM.
    returns up-to-date market insights relevant to that strategy.
    Use this tool when you need to:
    - Understand the current market landscape and sentiment
    - Identify key macro and micro drivers moving markets
    - Surface latent risks (geopolitical, regulatory, earnings, etc.)
    - Spot emerging opportunities across sectors or asset classes
    - Obtain a concise, evidence-based narrative for portfolio positioning
    - Make informed investment or allocation decisions grounded in the latest public information
    """
    # user_investment_strategy = runtime.context.bot.strategy
    market_research = await google_market_research_action.arun()
    return market_research


@tool(google_equity_research_action.name)
async def do_google_equity_research(ticker: str):
    """
    Performs equity research to synthesize and present the comprehensive current market narrative,
    key drivers, risks, and opportunities based on real-time data and recent news using Google’s grounded LLM.
    Returns up-to-date equity insights relevant to that strategy.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'TSLA') for which to generate research.
    """
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"

    equity_research = await google_equity_research_action.arun(ticker)
    return equity_research


__all__ = ["do_google_market_research", "do_google_equity_research"]
```

src/tools/risk_tools.py
```
from langchain.tools import tool
from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.risk import (
    FundamentalRiskDataAct,
    VolatilityRiskAct,
    PriceRiskAct,
)


fundamental_risk_act = FundamentalRiskDataAct()
volatility_risk_act = VolatilityRiskAct()
price_risk_act = PriceRiskAct()


@tool(fundamental_risk_act.name)
async def get_fundamental_risk_data(ticker: str):
    """Get fundamental risk data for a given ticker.

    Returns a markdown report structured into decision-ready sections:
    - Valuation Metrics
    - Profitability & Margins
    - Financial Health & Liquidity
    - Growth Metrics
    - Dividend & Payout
    - Market & Trading Data
    - Analyst Estimates & Ratings
    - Company Information
    - Ownership & Shares
    - Risk & Volatility
    - Technical Indicators

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
    return await fundamental_risk_act.arun(ticker)


@tool(volatility_risk_act.name)
async def get_volatility_risk_indicators(ticker: str):
    """Get volatility and tail-risk indicators for a ticker.

    Fetches ~1 year of daily historical price and compute multi-metric risks

    Includes:
    - Historical volatility (20d, 60d, 252d; annualized)
    - Garman–Klass and Parkinson range-based volatility (annualized)
    - Realized volatility and volatility clustering
    - Maximum drawdown and drawdown duration
    - VaR (95%, 99%) and CVaR (95%)
    - Jump detection (large jumps count, jump intensity)

    Returns a markdown summary of the computed indicators.

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
    return await volatility_risk_act.arun(ticker)


@tool(price_risk_act.name)
async def get_price_risk_indicators(ticker: str):
    """Get price action risk indicators for a ticker.

    Computes multi-horizon support/resistance, momentum, ATR, and breakout signals
    from ~1 year of daily bars.

    Includes:
    - Support/Resistance over lookbacks (default: 5, 10, 20, 50 days)
    - Distance to nearest support/resistance normalized by `current_price(T-1)`
    - Momentum per lookback: `(current_price - close[-h]) / close[-h]`
    - Average True Range per lookback and percent of price
    - Breakout/Breakdown flags based on support/resistance breaches
    - `current_price(T-1)`

    Returns a markdown table summarizing the computed indicators.

    Args:
        ticker: Stock symbol, e.g., "AAPL".
    """
    symbol = ticker.upper().strip()
    is_valid = await is_valid_ticker(symbol)
    if not is_valid:
        return f"{symbol} is an invalid ticker symbol"
    return await price_risk_act.arun(ticker)


__all__ = [
    "get_fundamental_risk_data",
    "get_volatility_risk_indicators",
    "get_price_risk_indicators",
]
```

src/tools/stock_tools.py
```
from pydantic import BaseModel, Field
from langchain.tools import tool
from src import tools_adaptors
from src import utils
from src.utils.ticker import is_valid_ticker

eft_live_price_change_act = tools_adaptors.ETFLivePriceChangeAct()
stock_live_price_change_act = tools_adaptors.StockLivePriceChangeAct()
most_active_stockers_act = tools_adaptors.MostActiveStockersAct()
single_latest_quotes_act = tools_adaptors.SingleLatestQuotesAct()
multi_latest_quotes_act = tools_adaptors.MultiLatestQuotesAct()
get_price_trend_act = tools_adaptors.GetPriceTrendAct()


@tool(eft_live_price_change_act.name)
async def get_etf_live_historical_price_change():
    """
    Fetch live and historical percent-change metrics for the most-traded U.S. equity ETFs (SPY, QQQ, IWM, etc.) in the different sectors.

    The returned string is a markdown snippet that contains:
    - A level-2 heading
    - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year etc)
    - A table whose columns are derived from the record fields
      (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Use this tool when you need a quick, snapshot of ETF momentum and relative strength

    Possible purposes:
    - Compare sector momentum at a glance
    - Identify ETFs with strongest/weakest 1-week or 1-month trends
    - Spot divergences between intraday and longer-term performance
    - Quickly screen for mean-reversion or breakout candidates among liquid ETFs
    - Build relative-strength rankings for portfolio rotation strategies
    """
    results = await eft_live_price_change_act.arun()
    etc_info_dict_list = list(results.values())
    heading = "## ETF Current and Historical Percent Changes"
    note = """
**Note:**
- The percent-change metrics are based on the current trading day and common historical windows such as 1-week, 1-month, and 1-year.
- The current intraday percent is the percent-change of the ETF's current price relative to the previous close."""
    markdown_table = utils.dicts_to_markdown_table(etc_info_dict_list)  # type: ignore
    return heading + "\n\n" + note + "\n\n" + markdown_table


class TickersInput(BaseModel):
    """Input for querying stock current price and historical price changes"""

    tickers: list[str] = Field(
        description="List of stock tickers, e.g. ['AAPL', 'MSFT', 'GOOGL']"
    )


@tool(stock_live_price_change_act.name, args_schema=TickersInput)
async def get_stock_live_historical_price_change(tickers: list[str]):
    """
    Fetch comprehensive percent-change metrics for the provided stock tickers.

    This function queries live and historical price data for each ticker and
    calculates percentage changes over multiple time windows for momentum analysis:
    - Current intraday percent change (vs. previous close)
    - 1-day, 1-week, 1-month, 3-month, 1-year, and 3-year percent changes.

    The returned data enables quick assessment of momentum, trend strength,
    and relative performance for portfolio monitoring, screening, or market
    commentary.

    Parameters
    ----------
    tickers : list[str]
        A list of valid stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL']).

    Returns
    -------
    str
        A markdown snippet that contains:
        - A level-2 heading
        - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year)
        - A table whose columns are derived from the record fields
          (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Possible purposes:
    - Compare momentum across holdings or watch-list names
    - Identify stocks with strongest/weakest 1-week or 1-month trends
    - Spot divergences between intraday and longer-term performance
    - Screen for mean-reversion or breakout candidates
    - Build relative-strength rankings for sector rotation or portfolio re-balancing
    - Generate quick market commentary on price action
    - Monitor portfolio positions for risk or opportunity signals
    """
    tickers = [ticker.upper().strip() for ticker in tickers]
    for ticker in tickers:
        is_valid = await is_valid_ticker(ticker)
        if not is_valid:
            return f"{ticker} is an invalid ticker symbol"

    results = await stock_live_price_change_act.arun(tickers)
    metrics_list = []
    for ticker, metrics in results.items():
        _dict = {"ticker": ticker, **metrics}
        metrics_list.append(_dict)

    markdown_table = utils.dicts_to_markdown_table(metrics_list)
    heading = "## Stock Current and Historical Percent Changes"
    datetime = utils.get_current_timestamp()
    note = f"""
**Note:**
- The data is fetched at {datetime} in New York time.
- The percent-change metrics are based on the current trading day and common historical windows such as 1-day, 1-week, 1-month, 3-month, 1-year, and 3-year.
- The current intraday percent is the percent-change of the stock's current price relative to the previous close."""
    return heading + "\n\n" + note + "\n\n" + markdown_table


@tool(most_active_stockers_act.name)
async def get_most_active_stockers():
    """
    Fetch the most active stockers by trading volume in the U.S. stock market.

    The returned string is a markdown snippet that contains:
    - A level-2 heading
    - A short note explaining the calculation windows (1-day, 1-week, 1-month, 3-month, 1-year, 3-year etc)
    - A table whose columns are derived from the record fields
      (typically including current intraday % change, 1-day, 1-week, 1-month, 3-month, 1-year, 3-year % changes).

    Use this tool when you need a quick snapshot of the most active stocks by trading volume in the U.S. market—ideal for spotting liquidity leaders,
    momentum surges, or unusual activity at a glance.

    Possible purposes:
    - Identify high-volume breakouts or breakdowns in real time
    - Screen for momentum-driven day-trading candidates
    - Spot unusual volume spikes that may signal news or earnings reactions
    - Build a liquidity-ranked watch-list for swing or intraday strategies
    - Compare relative volume intensity across leading names
    - Generate quick market commentary on where the action is today
    - Monitor for potential volatility expansion plays based on volume leadership
    """
    results = await most_active_stockers_act.arun()
    if results == "ERROR":
        return "Unknown error to get most active stockers."

    last_updated = results["last_updated"]
    most_active_stockers = results["most_actives"]
    markdown_table = utils.dicts_to_markdown_table(most_active_stockers)  # type: ignore
    heading = "## Most Active Stockers"
    note = f"""
**Note:**
- The data is fetched at {last_updated} in New York time.
- The table shows the most active stockers by trading volume in the U.S. stock market.
- The percent-change metrics are based on the current trading day and common historical windows such as 1-day, 1-week, 1-month, 3-month, 1-year, and 3-year.
- The current intraday percent is the percent-change of the stock's current price relative to the previous close.
"""
    return heading + "\n\n" + note + "\n\n" + markdown_table


@tool(multi_latest_quotes_act.name, args_schema=TickersInput)
async def get_latest_quotes(tickers: list[str]):
    """
    Fetch the latest bid/ask quotes and market data for multiple stock tickers.

    This tool provides real-time market data including:
    - Current bid and ask prices with sizes
    - Quote conditions and timestamp
    - Latest consolidated quote data

    Use this tool when you need:
    - Real-time pricing for trading decisions
    - To monitor bid/ask dynamics before placing orders
    - To get precise pricing for portfolio valuation

    Args:
        symbols: List of stock tickers (1-100 symbols recommended)

    Returns:
        A markdown-formatted table with latest quote data for each symbol.
    """
    tickers = [ticker.upper().strip() for ticker in tickers]
    for ticker in tickers:
        is_valid = await is_valid_ticker(ticker)
        if not is_valid:
            return f"{ticker} is an invalid ticker symbol"

    quotes_data = await multi_latest_quotes_act.arun(tickers)
    return quotes_data


class TickerInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")


@tool(single_latest_quotes_act.name, args_schema=TickerInput)
async def get_latest_quote(ticker: str):
    """
    Fetch the latest bid/ask quotes and market data for a single stock ticker.

    This tool provides real-time market data including:
    - Current bid and ask prices with sizes
    - Quote conditions and timestamp
    - Latest consolidated quote data

    Use this tool when you need:
    - Real-time pricing for trading decisions
    - To monitor bid/ask dynamics before placing orders
    - To get precise pricing for portfolio valuation

    Args:
        tickers: List of stock tickers (1-100 symbols recommended)

    Returns:
        A markdown-formatted table with latest quote data for each symbol.
    """
    ticker = ticker.upper().strip()
    is_valid = await is_valid_ticker(ticker)
    if not is_valid:
        return f"{ticker} is an invalid ticker symbol"
    quotes_data = await single_latest_quotes_act.arun(ticker)
    return quotes_data


@tool(get_price_trend_act.name)
async def get_price_trend(tickers: list[str]):
    """
    This tool provides the price trend for multiple stock tickers.

    Use this tool when you need:
    - To get the price trend for a single stock ticker.

    Args:
        tickers: List of stock tickers, e.g. ['AAPL', 'MSFT']

    Returns:
        A markdown-formatted string with the price trend for the stock tickers.
    """
    tickers = [ticker.upper().strip() for ticker in tickers]
    for ticker in tickers:
        is_valid = await is_valid_ticker(ticker)
        if not is_valid:
            return f"{ticker} is an invalid ticker symbol"
    price_trend = await get_price_trend_act.arun(tickers)
    return price_trend


__all__ = [
    "get_etf_live_historical_price_change",
    "get_stock_live_historical_price_change",
    "get_most_active_stockers",
    "get_latest_quotes",
    "get_latest_quote",
    "get_price_trend",
]
```

src/tools/trading_tools.py
```
# src/tools/trading_tools.py
from typing import Literal
from prisma.enums import Role, TradeType
from pydantic import BaseModel, Field
from langchain.tools import tool, ToolRuntime
from src.context import Context
from src.utils.ticker import is_valid_ticker
from src.tools_adaptors.trading import (
    BuyAct,
    SellAct,
    RecommendStockAct,
    GetAnalystsRecommendationsAct,
    WriteDownTickersToReviewAct,
)
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client

buy_act = BuyAct()
sell_act = SellAct()
recommend_stock_act = RecommendStockAct()
get_analysts_recommendations_act = GetAnalystsRecommendationsAct()
write_down_tickers_to_review_act = WriteDownTickersToReviewAct()


class BuyInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to buy")


class SellInput(BaseModel):
    """Input for querying latest quote for a single symbol"""

    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'")
    volume: float = Field(description="Number of shares to sell")


@tool(buy_act.name)
async def buy_stock(
    ticker: str,
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Execute a buy order for a stock.

    Args:
        ticker: Stock symbol to buy
        volume: Number of shares to buy
        rationale: Rationale for the buy order
        confidence: Confidence in the buy order (0.0-1.0)
    """
    ticker = ticker.upper().strip()
    is_valid = await is_valid_ticker(ticker)
    if not is_valid:
        return f"{ticker} is an invalid ticker symbol"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await buy_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
        volume=volume,
        rationale=rationale,
        confidence=confidence,
    )


@tool("write_down_tickers_to_review")
async def write_down_tickers_to_review(
    tickers: list[str],
    runtime: ToolRuntime[Context],
):
    """Write down the tickers to review.

    Args:
        tickers: List of tickers to review
    """

    tickers = [ticker.upper().strip() for ticker in tickers]
    for ticker in tickers:
        is_valid = await is_valid_ticker(ticker)
        if not is_valid:
            return f"{ticker} is an invalid ticker symbol"

    runId = runtime.context.run.id
    return await write_down_tickers_to_review_act.arun(
        run_id=runId,
        tickers=tickers,
    )


@tool(sell_act.name)
async def sell_stock(
    ticker: str,
    volume: float,
    rationale: str,
    confidence: float,
    runtime: ToolRuntime[Context],
):
    """Execute a sell order for a stock.

    Args:
        ticker: Stock symbol to sell
        volume: Number of shares to sell
        rationale: Rationale for the sell order
        confidence: Confidence in the sell order (0.0-1.0)
    """
    ticker = ticker.upper().strip()
    is_valid = await is_valid_ticker(ticker)
    if not is_valid:
        return f"{ticker} is an invalid ticker symbol"

    bot_id = runtime.context.bot.id
    runId = runtime.context.run.id
    return await sell_act.arun(
        runId=runId,
        bot_id=bot_id,
        ticker=ticker,
        volume=volume,
        rationale=rationale,
        confidence=confidence,
    )


@tool("get_market_status")
async def get_market_status():
    """Get the current market status. It's either open or closed.
    If the market is closed, you cannot buy or sell stock.
    """
    clock = alpaca_trading_client.get_clock()
    if not clock.is_open:  # type: ignore
        return "Market is closed. Cannot buy stock."

    return "Market is open. You can buy or sell stock."


def get_recommend_stock_tool(role: Role):
    @tool("recommend_stock")
    async def recommend_stock(
        ticker: str,
        amount: float,
        rationale: str,
        confidence: float,
        trade_type: Literal["BUY", "SELL", "HOLD"],
        runtime: ToolRuntime[Context],
    ):
        """Record a BUY, SELL, or HOLD recommendation for a stock.

        Calling tool is mandatory to log your trading suggestions when you recommend a stock.
        capturing the ticker symbol, desired action (buy, sell, or hold),
        the number of shares involved, the reasoning behind the recommendation with the confidence level.
        These recorded recommendations can later be reviewed or aggregated to guide final investment decisions.

        Args:
            ticker: Stock symbol to recommend to BUY, SELL, or HOLD
            amount: Number of shares to recommend to BUY, SELL, or HOLD
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)
            trade_type: Whether to buy or sell the stock: `BUY`, `SELL`, or `HOLD`
        """
        ticker = ticker.upper().strip()
        is_valid = await is_valid_ticker(ticker)
        if not is_valid:
            return f"{ticker} is an invalid ticker symbol"

        bot_id = runtime.context.bot.id
        run_id = runtime.context.run.id

        return await recommend_stock_act.arun(
            run_id=run_id,
            bot_id=bot_id,
            ticker=ticker,
            amount=amount,
            rationale=rationale,
            confidence=confidence,
            trade_type=TradeType(trade_type),
            role=role,
        )

    return recommend_stock


@tool("get_analysts_recommendations")
async def get_analysts_recommendations(
    runtime: ToolRuntime[Context],
) -> str:
    """Retrieve consolidated analyst recommendations for final investment decisions.

    Returns:
        List of recent recommendations with keys: ticker, action (BUY/SELL/HOLD),
        price_target, rationale.
    """
    run_id = runtime.context.run.id
    return await get_analysts_recommendations_act.arun(
        run_id=run_id,
    )
```

src/typings/__init__.py
```
from typing import Final, Literal

ERROR: Final = "ERROR"
ErrorLiteral = Literal["ERROR"]
```

src/typings/agent_roles.py
```
from typing import Literal

SubAgentRole = Literal[
    "MARKET_ANALYST",
    "RISK_ANALYST",
    "EQUITY_RESEARCH_ANALYST",
    "FUNDAMENTAL_ANALYST",
    "TRADING_EXECUTOR",
]

AgentRole = Literal[
    "CHIEF_INVESTMENT_OFFICER",
    "MARKET_ANALYST",
    "RISK_ANALYST",
    "EQUITY_RESEARCH_ANALYST",
    "FUNDAMENTAL_ANALYST",
    "TRADING_EXECUTOR",
]
```

src/typings/context.py
```
from dataclasses import dataclass
from prisma.models import Bot, Run


@dataclass
class Context:
    run: Run
    bot: Bot
    llm_model: str
```

src/typings/langgraph_agents.py
```
from langchain.agents.middleware.types import (
    AgentState,
    _InputAgentState,
    _OutputAgentState,
)
from langgraph.graph.state import CompiledStateGraph
from typing import TypeVar
from src.context import Context

Unknown = TypeVar("Unknown")


LangGraphAgent = CompiledStateGraph[
    AgentState[Unknown], Context, _InputAgentState, _OutputAgentState[Unknown]
]
```

src/utils/__init__.py
```
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
```

src/utils/constants.py
```
from typing import TypedDict, List
from src.typings.agent_roles import SubAgentRole, AgentRole


class ETFTickerInfo(TypedDict):
    ticker: str
    name: str
    description: str


ETF_TICKERS: list[ETFTickerInfo] = [
    {
        "ticker": "SPY",
        "name": "SPDR S&P 500 ETF Trust",
        "description": "Invests in 500 largest U.S. companies across all sectors",
    },
    {
        "ticker": "IVV",
        "name": "iShares Core S&P 500 ETF",
        "description": "Holds 500 largest U.S. stocks representing market leadership",
    },
    {
        "ticker": "VOO",
        "name": "Vanguard S&P 500 ETF",
        "description": "Contains 500 largest U.S. companies covering entire large-cap market",
    },
    {
        "ticker": "QQQ",
        "name": "Invesco QQQ Trust",
        "description": "Invests in 100 largest non-financial companies listed on Nasdaq, heavily weighted toward technology",
    },
    {
        "ticker": "VTI",
        "name": "Vanguard Total Stock Market ETF",
        "description": "Holds entire U.S. stock market including large, mid, small, and micro-cap companies",
    },
    {
        "ticker": "IWM",
        "name": "iShares Russell 2000 ETF",
        "description": "Invests in 2,000 small-cap U.S. companies",
    },
    {
        "ticker": "DIA",
        "name": "SPDR Dow Jones Industrial Average ETF",
        "description": "Holds 30 large-cap U.S. blue-chip companies",
    },
    {
        "ticker": "XLK",
        "name": "Technology Select Sector SPDR Fund",
        "description": "Concentrates on technology stocks from the S&P 500 including hardware, software, and services",
    },
    {
        "ticker": "XLF",
        "name": "Financial Select Sector SPDR Fund",
        "description": "Invests in financial sector companies including banks, insurance, and investment firms",
    },
    {
        "ticker": "XLV",
        "name": "Health Care Select Sector SPDR Fund",
        "description": "Holds healthcare companies including pharmaceuticals, biotech, and medical devices",
    },
    {
        "ticker": "XLE",
        "name": "Energy Select Sector SPDR Fund",
        "description": "Concentrates on energy sector companies including oil, gas, and energy equipment",
    },
    {
        "ticker": "XLI",
        "name": "Industrial Select Sector SPDR Fund",
        "description": "Invests in industrial companies including machinery, aerospace, and transportation",
    },
    {
        "ticker": "XLP",
        "name": "Consumer Staples Select Sector SPDR Fund",
        "description": "Holds consumer staples companies including food, beverages, and household products",
    },
    {
        "ticker": "XLY",
        "name": "Consumer Discretionary Select Sector SPDR Fund",
        "description": "Invests in consumer discretionary companies including retail, automotive, and entertainment",
    },
    {
        "ticker": "XLU",
        "name": "Utilities Select Sector SPDR Fund",
        "description": "Concentrates on utilities companies including electric, gas, and water utilities",
    },
    {
        "ticker": "XLB",
        "name": "Materials Select Sector SPDR Fund",
        "description": "Holds materials sector companies including chemicals, metals, and mining",
    },
    {
        "ticker": "XLRE",
        "name": "Real Estate Select Sector SPDR Fund",
        "description": "Invests in real estate investment trusts (REITs) and real estate management companies",
    },
    {
        "ticker": "VGT",
        "name": "Vanguard Information Technology ETF",
        "description": "Holds technology stocks including software, hardware, and IT services companies",
    },
    {
        "ticker": "VUG",
        "name": "Vanguard Growth ETF",
        "description": "Invests in large-cap U.S. growth stocks with strong earnings potential",
    },
    {
        "ticker": "VTV",
        "name": "Vanguard Value ETF",
        "description": "Holds large-cap U.S. value stocks trading at lower valuations",
    },
    {
        "ticker": "SCHD",
        "name": "Schwab U.S. Dividend Equity ETF",
        "description": "Invests in high-quality U.S. companies with strong dividend track records",
    },
    {
        "ticker": "VYM",
        "name": "Vanguard High Dividend Yield ETF",
        "description": "Holds U.S. companies with above-average dividend yields",
    },
    {
        "ticker": "BND",
        "name": "Vanguard Total Bond Market ETF",
        "description": "Invests in U.S. investment-grade bonds including government, corporate, and mortgage-backed securities",
    },
    {
        "ticker": "AGG",
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "description": "Holds U.S. investment-grade bonds across Treasury, corporate, and securitized sectors",
    },
    {
        "ticker": "LQD",
        "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "description": "Concentrates on U.S. investment-grade corporate bonds",
    },
    {
        "ticker": "HYG",
        "name": "iShares iBoxx $ High Yield Corporate Bond ETF",
        "description": "Invests in U.S. high-yield corporate bonds (junk bonds)",
    },
    {
        "ticker": "TLT",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "description": "Holds long-term U.S. Treasury bonds with maturities 20+ years",
    },
    {
        "ticker": "IEF",
        "name": "iShares 7-10 Year Treasury Bond ETF",
        "description": "Invests in intermediate-term U.S. Treasury bonds with 7-10 year maturities",
    },
    {
        "ticker": "SHY",
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "description": "Concentrates on short-term U.S. Treasury bonds with 1-3 year maturities",
    },
    {
        "ticker": "GLD",
        "name": "SPDR Gold Shares",
        "description": "Holds physical gold bullion stored in vaults",
    },
    {
        "ticker": "SLV",
        "name": "iShares Silver Trust",
        "description": "Invests in physical silver bullion stored in vaults",
    },
    {
        "ticker": "USO",
        "name": "United States Oil Fund",
        "description": "Holds futures contracts on West Texas Intermediate crude oil",
    },
    {
        "ticker": "VNQ",
        "name": "Vanguard Real Estate ETF",
        "description": "Invests in U.S. real estate investment trusts (REITs) across various property types",
    },
    {
        "ticker": "IYR",
        "name": "iShares U.S. Real Estate ETF",
        "description": "Holds U.S. real estate investment trusts (REITs) and real estate companies",
    },
    {
        "ticker": "VGK",
        "name": "Vanguard FTSE Europe ETF",
        "description": "Invests in European stocks across developed markets",
    },
    {
        "ticker": "VWO",
        "name": "Vanguard FTSE Emerging Markets ETF",
        "description": "Holds stocks from emerging markets including China, Taiwan, India, and Brazil",
    },
    {
        "ticker": "EEM",
        "name": "iShares MSCI Emerging Markets ETF",
        "description": "Invests in emerging market stocks across Asia, Latin America, and EMEA",
    },
    {
        "ticker": "EFA",
        "name": "iShares MSCI EAFE ETF",
        "description": "Holds developed market stocks from Europe, Australasia, and Far East",
    },
    {
        "ticker": "VEA",
        "name": "Vanguard FTSE Developed Markets ETF",
        "description": "Invests in developed market stocks excluding the United States",
    },
    {
        "ticker": "VT",
        "name": "Vanguard Total World Stock ETF",
        "description": "Holds global stocks from both developed and emerging markets worldwide",
    },
    {
        "ticker": "IWD",
        "name": "iShares Russell 1000 Value ETF",
        "description": "Invests in large and mid-cap U.S. value stocks",
    },
    {
        "ticker": "IWF",
        "name": "iShares Russell 1000 Growth ETF",
        "description": "Holds large and mid-cap U.S. growth stocks",
    },
    {
        "ticker": "IJR",
        "name": "iShares Core S&P Small-Cap ETF",
        "description": "Invests in small-cap U.S. stocks from the S&P SmallCap 600 Index",
    },
    {
        "ticker": "IJH",
        "name": "iShares Core S&P Mid-Cap ETF",
        "description": "Holds mid-cap U.S. stocks from the S&P MidCap 400 Index",
    },
    {
        "ticker": "MDY",
        "name": "SPDR S&P MidCap 400 ETF Trust",
        "description": "Invests in mid-cap U.S. stocks representing the S&P MidCap 400 Index",
    },
]

FUNDAMENTAL_CATEGORIES = {
    # Core Valuation Metrics
    "Valuation Metrics": [
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "marketCap",
        "enterpriseValue",
        "trailingPegRatio",
        "priceEpsCurrentYear",
    ],
    # Profitability & Earnings
    "Profitability & Margins": [
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "ebitdaMargins",
        "returnOnEquity",
        "returnOnAssets",
        "trailingEps",
        "forwardEps",
        "epsTrailingTwelveMonths",
        "epsForward",
        "epsCurrentYear",
        "netIncomeToCommon",
        "grossProfits",
        "ebitda",
        "operatingCashflow",
        "freeCashflow",
        "totalRevenue",
    ],
    # Financial Health & Balance Sheet
    "Financial Health & Liquidity": [
        "currentRatio",
        "quickRatio",
        "debtToEquity",
        "totalDebt",
        "totalCash",
        "totalCashPerShare",
        "bookValue",
        "floatShares",
        "sharesOutstanding",
        "impliedSharesOutstanding",
    ],
    # Growth & Performance
    "Growth Metrics": [
        "earningsGrowth",
        "revenueGrowth",
        "earningsQuarterlyGrowth",
        "52WeekChange",
        "fiftyTwoWeekChangePercent",
        "SandP52WeekChange",
        "fiftyDayAverageChange",
        "fiftyDayAverageChangePercent",
        "twoHundredDayAverageChange",
        "twoHundredDayAverageChangePercent",
    ],
    # Dividend & Shareholder Returns
    "Dividend & Payout": [
        # "dividendRate",
        "dividendYield",
        "payoutRatio",
        "lastDividendValue",
        # "lastDividendDate",
        # "dividendDate",
        # "exDividendDate",
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "fiveYearAvgDividendYield",
    ],
    # Market & Trading Data
    "Market & Trading Data": [
        "currentPrice",
        # 'regularMarketPrice',
        "previousClose",
        "open",
        "dayLow",
        "dayHigh",
        "regularMarketPreviousClose",
        "regularMarketOpen",
        "regularMarketDayLow",
        "regularMarketDayHigh",
        "volume",
        "regularMarketVolume",
        "averageVolume",
        "averageVolume10days",
        "averageDailyVolume10Day",
        "averageDailyVolume3Month",
        "bid",
        "ask",
        # 'bidSize',
        # 'askSize',
        "fiftyTwoWeekLow",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekRange",
        "fiftyDayAverage",
        "twoHundredDayAverage",
        "allTimeHigh",
        "allTimeLow",
    ],
    # Analyst Expectations
    "Analyst Estimates & Ratings": [
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
        "recommendationMean",
        "recommendationKey",
        "numberOfAnalystOpinions",
        "averageAnalystRating",
    ],
    # Company Profile
    "Company Information": [
        "longName",
        # 'shortName',
        "symbol",
        "exchange",
        "sector",
        # 'sectorDisp',
        # 'sectorKey',
        "industry",
        # 'industryDisp', 'industryKey',
        "fullTimeEmployees",
        "longBusinessSummary",
        # 'website', 'address1',
        # 'city', 'state', 'zip', 'country', 'phone',
        # 'companyOfficers'
    ],
    # Ownership & Capital Structure
    "Ownership & Shares": [
        "heldPercentInsiders",
        "heldPercentInstitutions",
        "sharesShort",
        "sharesShortPriorMonth",
        "sharesShortPreviousMonthDate",
        "dateShortInterest",
        "sharesPercentSharesOut",
        "shortRatio",
        "shortPercentOfFloat",
    ],
    # Risk Assessment
    "Risk & Volatility": [
        "beta",
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
        "overallRisk",
        # "maxAge",
    ],
    # Earnings & Financial Calendar
    # 'Earnings & Calendar': [
    #     'earningsTimestamp', 'earningsTimestampStart', 'earningsTimestampEnd',
    #     'earningsCallTimestampStart', 'earningsCallTimestampEnd',
    #     'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter',
    #     'lastSplitDate', 'lastSplitFactor', 'isEarningsDateEstimate'
    # ],
    # Technical Indicators
    "Technical Indicators": [
        "fiftyTwoWeekLowChange",
        "fiftyTwoWeekLowChangePercent",
        "fiftyTwoWeekHighChange",
        "fiftyTwoWeekHighChangePercent",
    ],
    # Additional Financial Metrics
    "Additional Financial Metrics": [
        "revenuePerShare",
        "financialCurrency",
        "currency",
        "priceHint",
        "totalCashPerShare",
    ],
    # Market Operations & Metadata
    # 'Market Operations': [
    #     'tradeable', 'triggerable', 'cryptoTradeable', 'esgPopulated',
    #     'quoteType', 'typeDisp', 'quoteSourceName', 'messageBoardId',
    #     'exchangeTimezoneName', 'exchangeTimezoneShortName',
    #     'gmtOffSetMilliseconds', 'market', 'exchangeDataDelayedBy',
    #     'sourceInterval', 'firstTradeDateMilliseconds',
    #     'hasPrePostMarketData', 'customPriceAlertConfidence'
    # ],
    # Post-Market & Extended Hours
    # 'Post-Market Data': [
    #     'postMarketChangePercent', 'postMarketPrice', 'postMarketChange',
    #     'postMarketTime', 'regularMarketTime', 'regularMarketChange',
    #     'regularMarketChangePercent', 'regularMarketDayRange',
    #     'fullExchangeName', 'displayName', 'marketState'
    # ]
}

FUNDAMENTAL_RISK_CATEGORIES = {
    # Market & Systematic Risk
    "Market & Systematic Risk": [
        "beta",
        "52WeekChange",
        "SandP52WeekChange",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
        "fiftyTwoWeekRange",
        "fiftyDayAverage",
        "twoHundredDayAverage",
        "allTimeHigh",
        "allTimeLow",
        "fiftyTwoWeekLowChange",
        "fiftyTwoWeekLowChangePercent",
        "fiftyTwoWeekHighChange",
        "fiftyTwoWeekHighChangePercent",
        "fiftyDayAverageChange",
        "fiftyDayAverageChangePercent",
        "twoHundredDayAverageChange",
        "twoHundredDayAverageChangePercent",
    ],
    # Financial & Leverage Risk
    "Financial & Leverage Risk": [
        "debtToEquity",
        "totalDebt",
        "currentRatio",
        "quickRatio",
        "totalCash",
        "totalCashPerShare",
        "bookValue",
        "enterpriseValue",
        "floatShares",
        "sharesOutstanding",
        "impliedSharesOutstanding",
    ],
    # Valuation Risk
    "Valuation Risk": [
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "priceToSalesTrailing12Months",
        "enterpriseToRevenue",
        "enterpriseToEbitda",
        "trailingPegRatio",
        "priceEpsCurrentYear",
        "marketCap",
    ],
    # Profitability & Business Risk
    "Profitability & Business Risk": [
        "profitMargins",
        "grossMargins",
        "operatingMargins",
        "ebitdaMargins",
        "returnOnEquity",
        "returnOnAssets",
        "trailingEps",
        "forwardEps",
        "epsTrailingTwelveMonths",
        "epsForward",
        "epsCurrentYear",
        "netIncomeToCommon",
        "grossProfits",
        "ebitda",
        "operatingCashflow",
        "freeCashflow",
        "totalRevenue",
        "revenuePerShare",
    ],
    # Growth & Momentum Risk
    "Growth & Momentum Risk": [
        "earningsGrowth",
        "revenueGrowth",
        "earningsQuarterlyGrowth",
    ],
    # Liquidity & Trading Risk
    "Liquidity & Trading Risk": [
        "volume",
        "regularMarketVolume",
        "averageVolume",
        "averageVolume10days",
        "averageDailyVolume10Day",
        "averageDailyVolume3Month",
        "currentPrice",
        "previousClose",
        "open",
        "dayLow",
        "dayHigh",
        "regularMarketPreviousClose",
        "regularMarketOpen",
        "regularMarketDayLow",
        "regularMarketDayHigh",
        "bid",
        "ask",
    ],
    # Dividend & Payout Risk
    "Dividend & Payout Risk": [
        "dividendYield",
        "payoutRatio",
        "lastDividendValue",
        "trailingAnnualDividendRate",
        "trailingAnnualDividendYield",
        "fiveYearAvgDividendYield",
    ],
    # Sentiment & Analyst Risk
    "Sentiment & Analyst Risk": [
        "targetHighPrice",
        "targetLowPrice",
        "targetMeanPrice",
        "targetMedianPrice",
        "recommendationMean",
        "recommendationKey",
        "numberOfAnalystOpinions",
        "averageAnalystRating",
    ],
    # Ownership & Short Interest Risk
    "Ownership & Short Interest Risk": [
        "heldPercentInsiders",
        "heldPercentInstitutions",
        "sharesShort",
        "sharesShortPriorMonth",
        "sharesShortPreviousMonthDate",
        "dateShortInterest",
        "sharesPercentSharesOut",
        "shortRatio",
        "shortPercentOfFloat",
    ],
    # Governance & Compliance Risk
    "Governance & Compliance Risk": [
        "auditRisk",
        "boardRisk",
        "compensationRisk",
        "shareHolderRightsRisk",
        "overallRisk",
    ],
    # Company & Sector Risk
    "Company & Sector Risk": [
        "longName",
        "symbol",
        "exchange",
        "sector",
        "industry",
        "fullTimeEmployees",
        "longBusinessSummary",
        "financialCurrency",
        "currency",
        "priceHint",
    ],
}

SPECIALIST_ROLES: List[SubAgentRole] = [
    "MARKET_ANALYST",
    "RISK_ANALYST",
    "EQUITY_RESEARCH_ANALYST",
    "FUNDAMENTAL_ANALYST",
    "TRADING_EXECUTOR",
]

ALL_ROLES: List[AgentRole] = [
    "CHIEF_INVESTMENT_OFFICER",
    "MARKET_ANALYST",
    "RISK_ANALYST",
    "EQUITY_RESEARCH_ANALYST",
    "FUNDAMENTAL_ANALYST",
    "TRADING_EXECUTOR",
]
```

src/utils/message.py
```
from langchain_core.messages import AIMessage, BaseMessage


def combine_ai_messages(messages: list[BaseMessage]) -> str:
    return "\n".join(
        [str(msg.content) for msg in messages if isinstance(msg, AIMessage)]
    )
```

src/utils/ticker.py
```
from src import db


async def is_valid_ticker(ticker: str):
    ticker = ticker.replace("-", ".")
    ticker_record = await db.prisma.ticker.find_unique(where={"ticker": ticker})

    return ticker_record is not None


__all__ = ["is_valid_ticker"]
```

src/tools_adaptors/__init__.py
```
from src.tools_adaptors.news import MarketNewsAct, EquityNewsAct
from src.tools_adaptors.research import GoogleMarketResearchAct, GoogleEquityResearchAct

from src.tools_adaptors.base import Action
from src.tools_adaptors.stocks import (
    ETFLivePriceChangeAct,
    StockCurrentPriceAndIntradayChangeAct,
    StockHistoricalPriceChangesAct,
    StockLivePriceChangeAct,
    MostActiveStockersAct,
    SingleLatestQuotesAct,
    MultiLatestQuotesAct,
    GetPriceTrendAct,
)
from src.tools_adaptors.portfolio import (
    ListPositionsAct,
    PortfolioPerformanceAnalysisAct,
)

from src.tools_adaptors.fundamental_data import FundamentalDataAct
from src.tools_adaptors.risk import FundamentalRiskDataAct, VolatilityRiskAct
from src.tools_adaptors.common import (
    GetUserInvestmentStrategyAct,
    SendInvestmentReportEmailAct,
    WriteInvestmentReportEmailAct,
    GetHistoricalReviewedTickersAct,
)

__all__ = [
    "Action",
    "GetPriceTrendAct",
    "SingleLatestQuotesAct",
    "MultiLatestQuotesAct",
    "MarketNewsAct",
    "EquityNewsAct",
    "GoogleMarketResearchAct",
    "ETFLivePriceChangeAct",
    "StockCurrentPriceAndIntradayChangeAct",
    "StockHistoricalPriceChangesAct",
    "StockLivePriceChangeAct",
    "MostActiveStockersAct",
    "ListPositionsAct",
    "PortfolioPerformanceAnalysisAct",
    "FundamentalDataAct",
    "FundamentalRiskDataAct",
    "VolatilityRiskAct",
    "GoogleEquityResearchAct",
    "GetUserInvestmentStrategyAct",
    "SendInvestmentReportEmailAct",
    "WriteInvestmentReportEmailAct",
    "GetHistoricalReviewedTickersAct",
]
```

src/tools_adaptors/base.py
```
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
```

src/tools_adaptors/common.py
```
import os
from datetime import datetime
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from src import db
from prisma.enums import RunStatus
from src import utils
from src.models import get_model
from src.utils import async_retry, send_ses_email
from src.tools_adaptors.base import Action


BASE_URL = os.getenv("BASE_URL", "http://localhost:3000")


class GetUserInvestmentStrategyAct(Action):
    @property
    def name(self) -> str:
        return "get_user_investment_strategy"

    @async_retry()
    async def arun(self, botId: str) -> str:
        bot = await db.prisma.bot.find_unique(where={"id": botId})
        if not bot:
            raise ValueError(f"Bot with ID {botId} not found.")

        return bot.strategy


class WriteInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "write_investment_report_email"

    @async_retry()
    async def arun(
        self, llm_model: str, botId: str, run_id: str, conversation: str
    ) -> str:
        system_prompt = (
            "You are the Synthesis & Communications Officer (SCO) for an AI investment team. "
            "You are NOT an analyst, strategist, or decision-maker. "
            "You are a neutral, meticulous scribe and editor whose sole purpose is to accurately capture, structure, "
            "and communicate the team's discussion into a professional investment report. "
            "You have no opinions, no biases, and no agenda beyond faithful representation and clarity. "
        )

        instruction = f"""
        Directly produce a single, self-contained HTML investment report summarizing the conversation below.


        Requirements:
        1. The investment summary must consolidate insights from every analyst involved in the run,
        presenting each analyst’s investment recommendation together with the detailed rationale
        behind it. It must conclude with the final trading decision (buy, sell, or hold) for
        the tickers, again including the rationale that led to that decision.
        
        2.A professionally formatted, self-contained, and stylish HTML string representing the consolidated investment report.
        It must include fully-styled inline CSS (e.g., font-family, color, padding, borders) to ensure
        consistent rendering across all major email clients. Plain text or Markdown are not acceptable.
        Ensure the content contains:
        - Aggregated insights from all analysts to comprehensively evaluate the tickers.
        - Each analyst’s recommendation and its rationale.
        - Final trading actions (buy/sell/hold) with supporting rationale for each ticker.
        - Line charts and bar charts rendered purely with HTML and CSS (no external images or scripts).
        
        3. Must Include A BUTTON with the link  {BASE_URL}/bots/{botId}/conversation?runId={run_id}
        to allow users to view the full conversation.

        Conversation:
        -------------
        {conversation}
        """

        langchain_model = get_model(llm_model)
        agent = create_agent(
            model=langchain_model,
            system_prompt=system_prompt,
        )
        result = agent.invoke({"messages": [HumanMessage(content=instruction)]})

        report = result["messages"][-1].content

        await db.prisma.run.update(where={"id": run_id}, data={"summary": report})
        return report


class SendInvestmentReportEmailAct(Action):
    @property
    def name(self) -> str:
        return "send_investment_report_email"

    @async_retry()
    async def arun(self, user_id, run_id: str, bot_name: str) -> str:
        run = await db.prisma.run.find_unique(where={"id": run_id})

        if not run:
            return "Run not found"

        investment_report = run.summary
        if not investment_report:
            return "Investment report not found. please write it first."

        date_str = datetime.now().strftime("%Y-%m-%d")

        recipient = await db.prisma.user.find_unique(where={"clerkId": user_id})

        if not recipient:
            return "User not found"

        email = recipient.email
        subject = f"SandX.AI Execution Summary — {bot_name} | {date_str} | {run_id}"

        investment_report = investment_report.strip()
        if investment_report.startswith("```html"):
            investment_report = investment_report["```html".__len__() :]
        if investment_report.startswith("```"):
            investment_report = investment_report[3:]
        if investment_report.endswith("```"):
            investment_report = investment_report[:-3]

        investment_report = investment_report.strip()

        send_ses_email(
            subject=subject,
            recipient=email,
            html_body=investment_report,
        )
        return "Sent investment report email successfully!"


class GetHistoricalReviewedTickersAct(Action):
    @property
    def name(self) -> str:
        return "get_historical_reviewed_tickers"

    @async_retry()
    async def arun(self, botId: str) -> str:
        bot = await db.prisma.bot.find_unique(where={"id": botId})
        if not bot:
            raise ValueError(f"Bot with ID {botId} not found.")

        runs = await db.prisma.run.find_many(
            where={"botId": botId, "status": RunStatus.SUCCESS},
            order={"createdAt": "desc"},
            take=7,
        )
        if not runs:
            return "No previous tickers reviewed found."

        run_tickers = []
        for run in runs:
            if run.tickers:
                tmp_dict = {
                    "tickers": run.tickers,
                    "reviewed_at": run.createdAt.strftime("%Y-%m-%d %H:%M:%S"),
                }
                run_tickers.append(tmp_dict)
        tickers_markdown = utils.dicts_to_markdown_table(run_tickers)
        return tickers_markdown
```

src/tools_adaptors/fundamental_data.py
```
from src.tools_adaptors.base import Action
from src.services.yfinance.api_info import async_get_ticker_info
from src.tools_adaptors import utils
from src.utils import constants, async_retry


class FundamentalDataAct(Action):
    @property
    def name(self):
        return "get_comprehensive_fundamental_data"

    @async_retry()
    async def arun(self, ticker: str) -> str:
        info = await async_get_ticker_info(ticker)
        info = utils.preprocess_info_dict(info)
        categorized_data = utils.get_categorized_metrics(
            info, categories_map=constants.FUNDAMENTAL_CATEGORIES
        )
        md = utils.format_fundamentals_markdown(categorized_data, ticker)
        return md


__all__ = ["FundamentalDataAct"]
```

src/tools_adaptors/google_equity_research.md
```
**Role:**
Act as a US Equity Research Analyst specializing in tactical, event-driven insights. Your task is to synthesize a comprehensive investment thesis for `{TICKER}` by integrating real-time company-specific news with the broader market narrative.

**Context:**
- **Analysis Date:** `{datetime}`
- **Target Ticker:** `{TICKER}`
- **Sources:** Use only reputable sources (Company IR, SEC Filings, Bloomberg, Reuters, WSJ, Industry Publications). Prioritize same-day news and releases.
- **Mandate:** Do not list facts—explain the “so what?” for `{TICKER}`. Connect macro and sector trends directly to the company's prospects.

**Execution Rules:**
1.  Use targeted web search for `{TICKER}`-specific news, press releases, and analyst commentary.
2.  Use broader market searches to contextualize `{TICKER}`'s price action within the sector and macro environment.
3.  Cite sources and timestamps for each insight.
4.  All data must be current (≤5 days).

---

### **Focus Areas for `{TICKER}`:**

**1. Executive Summary & Investment Thesis**
-   **Dominant Narrative:** What is the prevailing market narrative *for this specific stock*? (e.g., "AI beneficiary," "victim of higher rates," "turnaround story").
-   **Thesis Driver Ranking:** Rank the key drivers for `{TICKER}`: Company-Specific News (earnings, M&A, guidance) > Sector Trends > Macro Environment.
-   **Actionable Takeaway:** Provide a clear, concise recommendation framework (e.g., "BUY on product catalyst," "HOLD until earnings clarity," "SELL due to sector headwinds").

**2. Fundamental & Catalytic Drivers**
-   **Company-Specific Catalysts:**
    -   Recent earnings report: Key beats/misses, guidance changes, and management commentary.
    -   Recent press releases: M&A, product launches, regulatory approvals.
    -   Analyst Actions: Recent upgrades/downgrades and changes to price targets.
-   **Sector & Macro Impact:**
    -   How is the performance of the `{TICKER}`'s sector and the broader market (SPY/QQQ) influencing its price?
    -   Explain *why* recent news is causing the stock to move. (e.g., "The stock is down 5% today despite a beat because guidance was weak, signaling demand concerns.")

**3. Technical & Sentiment Positioning**
-   **Technical Health:**
    -   Key price levels: Support/Resistance, relation to key moving averages (50-day, 200-day).
    -   Momentum: RSI, MACD. Is the stock overbought or oversold?
    -   Unusual options activity (if available): Are traders betting on a big move?
-   **Market Sentiment:**
    -   What is the news sentiment (Bullish/Bearish/Neutral) around `{TICKER}`?
    -   How does this compare to the sentiment for its sector?

**4. Risk Assessment**
-   **Idiosyncratic Risks:** Company-specific risks (e.g., CEO departure, product delay, lawsuit, liquidity concerns).
-   **Systematic Risks:** How is `{TICKER}` exposed to broader risks? (e.g., "As a growth stock, it is highly sensitive to Fed rate expectations outlined in the CME FedWatch tool.").

**5. Forward Look & Monitoring Plan**
-   **Key Upcoming Events:** Next earnings date, investor day, product launch, Fed meeting, sector-specific reports.
-   **Bull & Bear Scenarios:**
    -   *Bull Case (Probability: X%):* What event or trend could drive the stock higher?
    -   *Bear Case (Probability: Y%):* What would cause a significant drop?
-   **Top 3 Metrics to Monitor:** The most crucial indicators for this stock (e.g., "Weekly sales data," "Changes in analyst estimates," "Sector ETF flows").
```

src/tools_adaptors/google_research.md
```
**Role:**  
Act as a US Financial Market Research Analyst. Your task is to synthesize and present the comprehensive current market state, key drivers, risks, and opportunities based on real-time data and recent news.

Focus on actionable insights for portfolio strategy.

**Context:**  
- Date: `{datetime}`
- Use only reputable, primary sources (Bloomberg, Reuters, WSJ, Fed, BLS, BEA, CME, CBOE).  
- All data must be current (≤5 days). Prioritize same-day news and releases.  
- Do not list facts—explain the “so what?” for portfolio managers. Highlight trade ideas, sector rotation, and hedging implications.

**Execution Rules:**  
1. Use targeted web search,url context tools.  
2. Cite sources and timestamps for each insight.  
---

### Focus Areas:

**1. Executive Summary**  
- What’s the dominant market narrative?  
- Rank key drivers: Fed, macro data, earnings, geopolitics.. 
- Provide comprehensive actionable portfolio takeaway (e.g., sector overweight, hedge, rotation).  

**2. Fundamental Drivers**  
- Fed commentary & CME FedWatch probabilities  
- Latest macro data (CPI, jobs, PCE) vs expectations  
- Notable earnings reactions and catalysts and explain *why* these factors caused the market to rise/fall.

**3. Sector & Asset Class Rotation**  
- Top/bottom 3 S&P sectors (last 5 days) with catalysts  
- Technical health: key levels, RSI, moving averages  
- Credit spreads and cross-asset signals

**4. Sentiment & Positioning**  
- Fear & Greed Index trend  
- Positioning extremes (CFTC, options, flows)

**5. Forward Look (Next 5–10 Days)**  
- Key upcoming events (Fed, macro, earnings)  
- Bull/Bear scenarios with probabilities  
- Top 3–5 metrics to monitor
```

src/tools_adaptors/news.py
```
from datetime import date, timedelta
from src.services.tradingeconomics import get_news
from src.services.alpaca import get_news as get_alpaca_news
from src.tools_adaptors.base import Action
from src.utils import convert_html_to_markdown, async_retry


class MarketNewsAct(Action):
    @property
    def name(self):
        return "get_latest_market_news"

    @async_retry()
    async def arun(self):
        data = await get_news()
        return data


class EquityNewsAct(Action):
    @property
    def name(self):
        return "get_latest_equity_news"

    @async_retry()
    async def arun(self, symbol: str):
        data = await get_alpaca_news(
            symbols=[symbol],
            start=(date.today() - timedelta(days=5)).isoformat(),
            end=date.today().isoformat(),
            sort="desc",
            limit=15,
        )

        news_list = data["news"]

        image_size_map = {"large": 3, "medium": 2, "small": 1, "thumb": 0}
        top_content = 3
        formatted_news: list[str] = []
        for i, new in enumerate(news_list):
            headline = new["headline"]
            summary = new["summary"]
            published_at = new["created_at"]
            image_url = (
                min(new["images"], key=lambda x: image_size_map.get(x["size"], 1))[
                    "url"
                ]
                if new.get("images")
                else ""
            )
            if i <= top_content:
                content = convert_html_to_markdown(new["content"])
            else:
                content = summary

            article = f"### {headline}\n"
            article += f"Published: {published_at}\n\n"
            article += f"![{headline}]({image_url})\n\n"
            article += f"{content}\n\n"
            formatted_news.append(article)

        heading = f"## Latest {symbol} News"
        return heading + "\n\n" + "\n\n".join(formatted_news)


__all__ = ["MarketNewsAct", "EquityNewsAct"]

if __name__ == "__main__":
    import asyncio

    # python -m src.tools_adaptors.news
    equity_news_action = EquityNewsAct()
    result = asyncio.run(equity_news_action.arun("AAPL"))  # type: ignore
    print(result)
```

src/tools_adaptors/portfolio.py
```
from typing import Sequence, Annotated, TypedDict
from src.tools_adaptors.base import Action
from src.services.sandx_ai import list_positions, get_timeline_values
from src.services.sandx_ai.typing import Position
from src.tools_adaptors import utils as action_utils
from src import utils
from src.utils import async_retry


class FormattedPosition(TypedDict):
    ticker: Annotated[str, "The stock ticker of the position"]
    allocation: Annotated[
        str, "The percentage allocation of the position in the portfolio"
    ]
    current_price: Annotated[str, "The current price of the stock position per share"]
    ptc_change_in_price: Annotated[
        str, "The percentage change in price relative to the open price"
    ]
    current_value: Annotated[
        str, "The total current value of the position in the portfolio"
    ]
    volume: Annotated[str, "The total share of the position in the portfolio"]
    cost: Annotated[str, "The average cost of the position in the portfolio"]
    pnl: Annotated[str, "Profit and Loss of the position in the portfolio"]
    pnl_percent: Annotated[
        str, "Profit and Loss percentage of the position in the portfolio"
    ]


def convert_positions_to_markdown_table(positions: Sequence[Position]) -> str:
    """
    Convert a list of Position objects to a markdown table.

    Parameters
    ----------
    positions : list[Position]
        A list of Position objects to be converted into a markdown table.

    Returns
    -------
    str
        A markdown table string representation of the positions.
    """
    positions = sorted(positions, key=lambda x: x["allocation"], reverse=True)
    formatted_positions = []
    for position in positions:
        formatted_position = FormattedPosition(
            ticker=position["ticker"],
            volume=utils.format_float(position["volume"]),
            cost=utils.format_float(position["cost"]),
            current_price=utils.format_float(position["current_price"]),
            ptc_change_in_price=utils.format_percent_change(
                position["ptc_change_in_price"]
            ),
            current_value=utils.format_currency(position["current_value"]),
            allocation=utils.format_percent(position["allocation"]),
            pnl=utils.format_currency(position["pnl"]),
            pnl_percent=utils.format_percent(position["pnl_percent"]),
        )
        formatted_positions.append(formatted_position)
    position_markdown = utils.dicts_to_markdown_table(formatted_positions)
    heading = "## User's Current Open Positions"

    datetime = utils.get_current_timestamp()

    note = f"""
- Datetime New York Time: {datetime}
- CASH is a special position that represents the cash balance in the account.
- ptc_change_in_price is the percentage change from the position’s open price to the current price.
- Allocation percentages are computed against the sum of currentValue across
    **all** positions plus any cash held in the same account.
    """

    return heading + "\n\n" + note + "\n\n" + position_markdown


class ListPositionsAct(Action):
    @property
    def name(self):
        return "list_current_positions"

    @async_retry()
    async def arun(self, bot_id: str) -> str:
        positions = await list_positions(bot_id)
        position_markdown = convert_positions_to_markdown_table(positions)
        return position_markdown


class PortfolioPerformanceAnalysisAct(Action):
    @property
    def name(self):
        return "get_portfolio_performance_analysis"

    @async_retry()
    async def arun(self, bot_id: str):
        timeline_values = await get_timeline_values(bot_id)
        analysis = action_utils.analyze_timeline_value(timeline_values)
        if analysis:
            return action_utils.create_performance_narrative(analysis)

        return "Insufficient data for analysis."


__all__ = ["ListPositionsAct", "PortfolioPerformanceAnalysisAct"]
```

src/tools_adaptors/research.py
```
import asyncio
import os
from google import genai
from google.genai import types

from src.tools_adaptors.base import Action
from src.services.utils import redis_cache
from src.utils import async_wrap, async_retry
from src.utils import get_current_date


class GoogleMarketResearchAct(Action):
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        self.config = types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            max_output_tokens=65535,
            tools=tools,
            thinking_config=types.ThinkingConfig(
                # include_thoughts=True,
                thinking_budget=-1,
            ),
        )

    @property
    def name(self):
        return "google_finance_market_research"

    @async_retry()  # pyright: ignore [reportArgumentType]
    @redis_cache(function_name="GoogleMarketResearch.arun", ttl=60 * 60 * 6)
    async def arun(self):
        return await self.run()

    @async_wrap()
    def run(self):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )

        with open(
            "./src/tools_adaptors/google_research.md", mode="r", encoding="utf-8"
        ) as f:
            prompt_template = f.read()

        datetime_str = get_current_date()
        prompt = prompt_template.format(datetime=datetime_str)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=self.config,
        )

        text = ""
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    if part.text:
                        text += part.text

        return text


class GoogleEquityResearchAct(Action):
    def __init__(self):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]
        self.config = types.GenerateContentConfig(
            temperature=1,
            top_p=1,
            max_output_tokens=65535,
            tools=tools,
            thinking_config=types.ThinkingConfig(
                # include_thoughts=True,
                thinking_budget=-1,
            ),
        )

    @property
    def name(self):
        return "google_equity_research"

    @async_retry()  # pyright: ignore [reportArgumentType]
    @redis_cache(function_name="GoogleEquityResearch.arun", ttl=60 * 60 * 24)
    async def arun(self, ticker: str):
        return await self.run(ticker)

    @async_wrap()
    def run(self, ticker: str):
        self.client = genai.Client(
            vertexai=True,
            api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
        )

        with open(
            "./src/tools_adaptors/google_equity_research.md", mode="r", encoding="utf-8"
        ) as f:
            prompt_template = f.read()

        datetime_str = get_current_date()
        prompt = prompt_template.format(TICKER=ticker, datetime=datetime_str)

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=self.config,
        )

        text = ""
        if response.candidates:
            for candidate in response.candidates:
                if not candidate.content or not candidate.content.parts:
                    continue
                for part in candidate.content.parts:
                    if part.text:
                        text += part.text

        return text


if __name__ == "__main__":
    # python -m src.tools.actions.research
    google_market_research_action = GoogleMarketResearchAct()
    result = asyncio.run(google_market_research_action.arun())  # type: ignore
    print(result)
```

src/tools_adaptors/risk.py
```
from datetime import date, timedelta

from src.services.alpaca import get_historical_price_bars
from src.services.yfinance.api_info import async_get_ticker_info
from src.tools_adaptors import utils
from src.tools_adaptors.base import Action
from src.utils import constants, async_retry


class FundamentalRiskDataAct(Action):
    @property
    def name(self):
        return "get_comprehensive_fundamental_risk_data"

    @async_retry()
    async def arun(self, ticker: str) -> str:
        info = await async_get_ticker_info(ticker)
        info = utils.preprocess_info_dict(info)
        categorized_data = utils.get_categorized_metrics(
            info, categories_map=constants.FUNDAMENTAL_RISK_CATEGORIES
        )
        md = utils.format_fundamentals_markdown(categorized_data, ticker)
        return md


class VolatilityRiskAct(Action):
    @property
    def name(self):
        return "get_volatility_risk_indicators"

    @async_retry()
    async def arun(self, ticker: str) -> str:
        """Get volatility risk indicators for a ticker"""
        start = (date.today() - timedelta(days=180 + 7)).isoformat()
        end = date.today().isoformat()
        price_bars = await get_historical_price_bars(
            symbols=[ticker], timeframe="1Day", start=start, end=end, sort="asc"
        )
        price_bars = price_bars[ticker]
        risk = utils.calculate_volatility_risk(price_bars)
        md = utils.format_volatility_risk_markdown(risk, ticker)
        return md


class PriceRiskAct(Action):
    @property
    def name(self):
        return "get_price_risk_indicators"

    @async_retry()
    async def arun(self, ticker: str) -> str:
        """Get price risk indicators for a ticker"""
        start = (date.today() - timedelta(days=180 + 7)).isoformat()
        end = date.today().isoformat()
        price_bars = await get_historical_price_bars(
            symbols=[ticker], timeframe="1Day", start=start, end=end, sort="asc"
        )
        price_bars = price_bars[ticker]
        risk = utils.calculate_price_risk(price_bars)
        md = utils.format_price_risk_markdown(risk, ticker)
        return md


__all__ = ["FundamentalRiskDataAct", "VolatilityRiskAct", "PriceRiskAct"]
```

src/tools_adaptors/stocks.py
```
import asyncio
from datetime import time, datetime, timedelta, timezone, date
from typing import TypedDict, List
from src.services.alpaca import (
    get_snapshots,
    get_historical_price_bars,
    get_most_active_stocks,
    get_latest_quotes,
)
from src.services.alpaca.typing import PriceBar
from src.tools_adaptors.base import Action
from src import utils
from src.utils.constants import ETF_TICKERS
from src.utils import async_retry


class StockRawSnapshotAct(Action):
    @property
    def name(self):
        return "Get Stock Snapshot"

    @async_retry()
    async def arun(self, tickers: list[str]) -> dict:
        """
        Fetch raw market snapshots for multiple tickers.

        Returns the latest trade, latest quote, minute bar, daily bar, and previous daily bar data
        for each ticker symbol.

        Args:
            tickers: A list of ticker symbols.
        Returns:
            A dictionary mapping each ticker to its complete raw snapshot data.
        """
        data = await get_snapshots(tickers)
        return data


class CurrentPriceAndIntradayChange(TypedDict):
    current_price: str
    current_intraday_percent: str


class StockCurrentPriceAndIntradayChangeAct(Action):
    @property
    def name(self):
        return "Stock Current Price and Intraday Change"

    @async_retry()
    async def arun(
        self, tickers: list[str]
    ) -> dict[str, CurrentPriceAndIntradayChange]:
        """
        Fetch the current price and intraday percent change for a list of tickers.

        The intraday change is calculated against the previous day's close:
        - Before 9:30 AM ET, the reference is the daily bar close.
        - At or after 9:30 AM ET, the reference is the previous daily bar close.

        Args:
            tickers: A list of ticker symbols.
        Returns:
            A dictionary mapping each ticker to its current price and intraday percent change.
        """

        ticker_price_changes = {}

        snapshots = await get_snapshots(tickers)

        for ticker in tickers:
            # Get current time in New York timezone
            current_time = utils.get_new_york_datetime().time()
            # Use dailyBar if before 9:30 AM, otherwise use prevDailyBar
            if current_time < time(9, 30):
                previous_close_price = snapshots[ticker]["dailyBar"]["c"]
            else:
                previous_close_price = snapshots[ticker]["prevDailyBar"]["c"]

            intraday_percent = (
                snapshots[ticker]["latestQuote"]["bp"] - previous_close_price
            ) / previous_close_price

            ticker_price_changes[ticker] = CurrentPriceAndIntradayChange(
                current_price=utils.format_currency(
                    snapshots[ticker]["latestQuote"]["bp"]
                ),
                current_intraday_percent=utils.format_percent(intraday_percent),
            )

        return ticker_price_changes


class HistoricalPriceChangePeriods(TypedDict):
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class StockHistoricalPriceChangesAct(Action):
    @property
    def name(self):
        return "Stock Historical Price Changes"

    # disable: pylint:disable=too-many-locals
    @async_retry()
    async def arun(self, tickers: list[str]) -> dict[str, HistoricalPriceChangePeriods]:
        """
        Compute percentage changes over standard periods using Alpaca daily bars.

        Periods: one_day, one_week, one_month, three_months, six_months, one_year, three_years.

        - Uses UTC timestamps (ISO 8601 with trailing 'Z') as required by Alpaca.
        - Selects the most recent close as the "current" reference.
        - For each period, finds the close on or before the target date.
        - Returns None when not enough history exists for a period.

        Args:
            tickers: List of ticker symbols.
        Returns:
            Mapping of ticker to period percent changes, e.g. {"AAPL": {"1w": 0.02, ...}}.
        """

        def _parse_ts(ts: str) -> datetime:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                )

        def _find_close_on_or_before(
            bars: List[PriceBar], target_dt: datetime
        ) -> float | None:
            for bar_ in bars:
                bar_dt = _parse_ts(bar_["timestamp"])
                if bar_dt <= target_dt:
                    return bar_["close_price"]
            return None

        period_deltas: dict[str, timedelta] = {
            "one_day": timedelta(days=1),
            "one_week": timedelta(days=7),
            "one_month": timedelta(days=30),
            "three_months": timedelta(days=90),
            "six_months": timedelta(days=180),
            # "one_year": timedelta(days=365),
            # "three_years": timedelta(days=365 * 3),
        }

        start = (date.today() - timedelta(days=180 + 7)).isoformat()
        end = date.today().isoformat()

        bars_by_symbol = await get_historical_price_bars(
            symbols=tickers,
            timeframe="1Day",
            start=start,
            end=end,
            sort="desc",  # IMPORTANT: sort by descending timestamp
        )

        results = {}
        for ticker in tickers:
            if not bars_by_symbol.get(ticker):
                continue
            bars = bars_by_symbol[ticker]
            latest_close = float(bars[0]["close_price"])
            latest_dt = _parse_ts(bars[0]["timestamp"])

            period_changes: dict[str, str | None] = {
                "one_day": None,
                "one_week": None,
                "one_month": None,
                "three_months": None,
                "six_months": None,
                "one_year": None,
                "three_years": None,
            }

            for key, delta in period_deltas.items():
                target_dt = latest_dt - delta
                prior_close = _find_close_on_or_before(bars, target_dt)
                if prior_close is not None and prior_close != 0:
                    period_changes[key] = utils.format_percent_change(
                        (latest_close - prior_close) / prior_close
                    )
                else:
                    period_changes[key] = None

            results[ticker] = period_changes

        return results


class StockPriceSnapshotWithHistory(TypedDict):
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class ETFPriceSnapshotWithHistory(TypedDict):
    ticker: str
    name: str
    description: str
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class StockLivePriceChangeAct(Action):
    @property
    def name(self):
        return "get_stock_live_price_and_change"

    # To need to retry because the sub-actions has retry decorator
    # @async_retry()
    async def arun(
        self, tickers: list[str]
    ) -> dict[str, StockPriceSnapshotWithHistory]:
        """
        Fetch a complete price snapshot for multiple tickers.

        Returns current price, intraday change, and historical percent changes
        for standard periods (1D, 1W, 1M, 3M, 6M, 1Y, 3Y) in a single call.
        """

        current_task = asyncio.create_task(
            StockCurrentPriceAndIntradayChangeAct().arun(tickers)
        )
        historical_task = asyncio.create_task(
            StockHistoricalPriceChangesAct().arun(tickers)
        )

        current, historical = await asyncio.gather(current_task, historical_task)

        results = {}
        if historical == "ERROR":
            return results

        if current == "ERROR":
            return results

        for ticker in tickers:
            history: HistoricalPriceChangePeriods = historical[ticker]
            currency: CurrentPriceAndIntradayChange = current[ticker]
            results[ticker] = StockPriceSnapshotWithHistory(
                current_price=currency["current_price"],
                current_intraday_percent=currency["current_intraday_percent"],
                one_day=history["one_day"],
                one_week=history["one_week"],
                one_month=history["one_month"],
                three_months=history["three_months"],
                six_months=history["six_months"],
                one_year=history["one_year"],
                three_years=history["three_years"],
            )

        return results


class ETFLivePriceChangeAct(Action):
    @property
    def name(self):
        return "get_major_etf_live_price_and_historical_change"

    # No need to retry because the sub-actions has retry decorator
    # @async_retry()
    async def arun(self) -> dict[str, ETFPriceSnapshotWithHistory]:
        """
        Fetch a complete price snapshot for multiple major ETF tickers.

        Returns current price, intraday change, and historical percent changes
        for standard periods (1D, 1W, 1M, 3M, 6M, 1Y, 3Y) in a single call.
        """
        tickers = [t["ticker"] for t in ETF_TICKERS]
        ticker_info_dict = {t["ticker"]: t for t in ETF_TICKERS}

        current_task = asyncio.create_task(
            StockCurrentPriceAndIntradayChangeAct().arun(tickers)
        )
        historical_task = asyncio.create_task(
            StockHistoricalPriceChangesAct().arun(tickers)
        )

        current, historical = await asyncio.gather(current_task, historical_task)

        results = {}
        if current == "ERROR":
            return results
        if historical == "ERROR":
            return results

        for ticker in tickers:
            history: HistoricalPriceChangePeriods = historical[ticker]
            currency: CurrentPriceAndIntradayChange = current[ticker]
            results[ticker] = ETFPriceSnapshotWithHistory(
                ticker=ticker,
                name=ticker_info_dict[ticker]["name"],
                description=ticker_info_dict[ticker]["description"],
                current_price=currency["current_price"],
                current_intraday_percent=currency["current_intraday_percent"],
                one_day=history["one_day"],
                one_week=history["one_week"],
                one_month=history["one_month"],
                three_months=history["three_months"],
                six_months=history["six_months"],
                one_year=history["one_year"],
                three_years=history["three_years"],
            )

        return results


class ActiveStockFullPriceMetrics(TypedDict):
    symbol: str
    trade_count: int
    volume: int
    current_price: str
    current_intraday_percent: str
    one_day: str | None
    one_week: str | None
    one_month: str | None
    three_months: str | None
    six_months: str | None
    one_year: str | None
    three_years: str | None


class MostActiveStockFullPriceMetrics(TypedDict):
    last_updated: str
    most_actives: list[ActiveStockFullPriceMetrics]


class MostActiveStockersAct(Action):
    @property
    def name(self):
        return "get_most_active_stockers_with_historical_price_changes"

    @async_retry()
    async def arun(self):
        """
        Get the most active stockers.
        """
        data = await get_most_active_stocks()
        tickers = [item["symbol"] for item in data["most_actives"]]
        most_active_stocks: list[ActiveStockFullPriceMetrics] = []

        results = await StockLivePriceChangeAct().arun(tickers)

        for item in data["most_actives"]:
            symbol = item["symbol"]
            price_metrics = results.get(symbol)
            if price_metrics:
                full_metrics = ActiveStockFullPriceMetrics(
                    symbol=symbol,
                    trade_count=item["trade_count"],
                    volume=item["volume"],
                    **price_metrics,
                )
                most_active_stocks.append(full_metrics)

        return MostActiveStockFullPriceMetrics(
            last_updated=data["last_updated"],
            most_actives=most_active_stocks,
        )


# src/tools_adaptors/stocks.py - Add this class
class MultiLatestQuotesAct(Action):
    @property
    def name(self):
        return "get_multi_symbols_latest_quotes"

    @async_retry()
    async def arun(self, symbols: list[str]) -> str:
        """
        Fetch latest quotes for multiple symbols.

        Returns:
            dict: Mapping of symbol to quote data including:
                - ask_price: Current ask price
                - ask_size: Ask size
                - bid_price: Current bid price
                - bid_size: Bid size
                - timestamp: Quote timestamp
        """

        quotesResponse = await get_latest_quotes(symbols)
        formatted_quotes = []
        quotes = quotesResponse["quotes"]
        for symbol in quotes:
            quote = quotes[symbol]
            formatted_quote = {
                "symbol": symbol,
                "bid_price": utils.format_currency(quote["bid_price"]),
                "bid_size": utils.human_format(quote["bid_size"]),
                "ask_price": utils.format_currency(quote["ask_price"]),
                "ask_size": utils.human_format(quote["ask_size"]),
                "spread": utils.format_currency(
                    quote["ask_price"] - quote["bid_price"]
                ),
                "spread_percent": utils.format_percent(
                    (quote["ask_price"] - quote["bid_price"]) / quote["bid_price"]
                ),
                "exchange": f"{quote['bid_exchange']}/{quote['ask_exchange']}",
                "timestamp": utils.format_datetime(quote["timestamp"]),
                # "conditions": ", ".join(quote["conditions"]) if quote["conditions"] else "Normal"
            }
            formatted_quotes.append(formatted_quote)

        if not formatted_quotes:
            return "No quote data available for the requested symbols."

        markdown_table = utils.dicts_to_markdown_table(formatted_quotes)
        heading = "## Latest Market Quotes"
        note = f"""
**Note:**
- Data fetched at {utils.get_current_timestamp()} New York time
- Bid/Ask prices are real-time consolidated quotes
- Spread = Ask Price - Bid Price
- Conditions: 'R' = Regular Market, 'O' = Opening Quote, 'C' = Closing Quote
"""

        return heading + "\n\n" + note + "\n\n" + markdown_table


class SingleLatestQuotesAct(Action):
    @property
    def name(self):
        return "get_single_symbol_latest_quotes"

    @async_retry()
    async def arun(self, symbol: str) -> str:
        """
        Fetch latest quotes for a single symbol.

        Returns:
            dict: Mapping of symbol to quote data including:
                - ask_price: Current ask price
                - ask_size: Ask size
                - bid_price: Current bid price
                - bid_size: Bid size
                - timestamp: Quote timestamp
        """

        quotes = await get_latest_quotes([symbol])
        quote = quotes.get(symbol)

        if not quote:
            return "No quote data available for the requested symbol."

        formatted_quote = {
            "symbol": symbol,
            "bid_price": utils.format_currency(quote["bid_price"]),
            "bid_size": utils.human_format(quote["bid_size"]),
            "ask_price": utils.format_currency(quote["ask_price"]),
            "ask_size": utils.human_format(quote["ask_size"]),
            "spread": utils.format_currency(quote["ask_price"] - quote["bid_price"]),
            "spread_percent": utils.format_percent(
                (quote["ask_price"] - quote["bid_price"]) / quote["bid_price"]
            ),
            "exchange": f"{quote['bid_exchange']}/{quote['ask_exchange']}",
            "timestamp": utils.format_datetime(quote["timestamp"]),
            # "conditions": ", ".join(quote["conditions"]) if quote["conditions"] else "Normal"
        }

        markdown_table = utils.dict_to_markdown_table(formatted_quote)
        heading = f"## Latest {symbol} Market Quotes"
        note = f"""
**Note:**
- Data fetched at {utils.get_current_timestamp()} New York time
- Bid/Ask prices are real-time consolidated quotes
- Spread = Ask Price - Bid Price
- Conditions: 'R' = Regular Market, 'O' = Opening Quote, 'C' = Closing Quote
"""

        return heading + "\n\n" + note + "\n\n" + markdown_table


class GetPriceTrendAct(Action):
    @property
    def name(self):
        return "get_price_trend"

    @async_retry()
    async def arun(self, symbols: List[str]) -> str:
        """
        Get the price trend of a stock.

        Returns:
            str: The price trend of the stock.
        """

        start = (date.today() - timedelta(days=31)).isoformat()
        end = date.today().isoformat()

        bars_by_symbol = await get_historical_price_bars(
            symbols=symbols,
            timeframe="1Day",
            start=start,
            end=end,
            sort="desc",  # IMPORTANT: sort by descending timestamp
        )

        markdown_tables = []
        for ticker in symbols:
            bars = bars_by_symbol[ticker]
            bars_ = []
            for index, bar in enumerate(bars):
                if index == len(bars) - 1:
                    break
                next_price = bars[index + 1]["close_price"]
                bars_.append(
                    {
                        "date": bar["timestamp"],
                        "price": bar["close_price"],
                        "change_percent": utils.format_percent(
                            (bar["close_price"] - next_price) / next_price
                        ),
                    }
                )

            markdown_table = utils.dicts_to_markdown_table(bars_)
            markdown_table = "**" + ticker + "**\n" + markdown_table
            markdown_tables.append(markdown_table)
        markdown_tables_str = "\n".join(markdown_tables)

        stock_price_changes = (
            (
                "Note: change_percent = (current_price - previous_day_price) / previous_day_price "
                "Measuring the change percent of the stock price relative to the previous day's price"
            )
            + "\n"
            + markdown_tables_str
        )
        return stock_price_changes


# only Act
__all__ = [
    "StockRawSnapshotAct",
    "StockCurrentPriceAndIntradayChangeAct",
    "StockHistoricalPriceChangesAct",
    "StockLivePriceChangeAct",
    "ETFLivePriceChangeAct",
    "MostActiveStockersAct",
    "SingleLatestQuotesAct",
    "MultiLatestQuotesAct",
]


if __name__ == "__main__":
    # python -m src.tools.actions.stocks
    async def main():
        changes = await StockLivePriceChangeAct().arun(["AAPL"])
        print(changes)

        etf_changes = await ETFLivePriceChangeAct().arun()
        print(etf_changes)

        most_active_stocks = await MostActiveStockersAct().arun()
        print(most_active_stocks)

    asyncio.run(main())
```

src/tools_adaptors/trading.py
```
from prisma.types import PositionCreateInput, PositionUpdateInput, TradeCreateInput
from src.services.alpaca import get_latest_quotes
from prisma.enums import TradeType
from prisma.enums import Role
from prisma.types import RecommendCreateInput
from src import utils, db
from src.utils import async_retry
from src.tools_adaptors.base import Action
from src.tools_adaptors.utils import format_recommendations_markdown
from src.services.alpaca.sdk_trading_client import client as alpaca_trading_client


class BuyAct(Action):
    @property
    def name(self):
        return "buy_stock"

    @async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot buy stock."
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("ask_price")
        if not price:
            return f"Cannot get price for {ticker}"
        price = float(price)
        total_cost = price * volume

        ticker = ticker.upper().strip()

        valid_ticker = await db.prisma.ticker.find_unique(
            where={"ticker": ticker.replace(".", "-")}
        )

        if valid_ticker is None:
            return f"Invalid ticker {ticker}"

        async with db.prisma.tx() as transaction:
            portfolio = await transaction.portfolio.find_unique(where={"botId": bot_id})
            if portfolio is None:
                raise ValueError("Portfolio not found")
            if portfolio.cash < total_cost:
                return f"Not enough cash to buy {volume} shares of {ticker} at {price} per share."
            portfolio.cash -= total_cost
            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": portfolio.cash}
            )
            existing = await transaction.position.find_unique(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                }
            )

            if existing is None:
                await transaction.position.create(
                    data=PositionCreateInput(
                        ticker=ticker,
                        volume=volume,
                        portfolioId=portfolio.id,
                        cost=price,
                    )
                )
                return (
                    f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                    f"Current volume is {volume} "
                    f"with average cost {utils.format_float(price)}"
                )
            else:
                await transaction.position.update(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    },
                    data=PositionUpdateInput(
                        volume=existing.volume + volume,
                        cost=(existing.cost * existing.volume + price * volume)
                        / (existing.volume + volume),
                    ),
                )

                await transaction.trade.create(
                    data=TradeCreateInput(
                        rationale=rationale,
                        confidence=confidence,
                        type=TradeType.BUY,
                        price=price,
                        ticker=ticker,
                        amount=volume,
                        runId=runId,
                        botId=bot_id,
                    )
                )

                return (
                    f"Successfully bought {volume} shares of {ticker} at {price} per share. "
                    f"Current volume is {existing.volume + volume} "
                    f"with average cost {utils.format_float(existing.cost)}"
                )


class SellAct(Action):
    @property
    def name(self):
        return "sell_stock"

    @utils.async_retry()
    async def arun(
        self,
        runId,
        bot_id: str,
        ticker: str,
        volume: float,
        rationale: str,
        confidence: float,
    ):
        clock = alpaca_trading_client.get_clock()
        if not clock.is_open:  # type: ignore
            return "Market is closed. Cannot sell stock."
        quotes = await get_latest_quotes([ticker])
        price = quotes["quotes"].get(ticker, {}).get("bid_price")
        if not price:
            return f"Cannot get price for {ticker}"

        # price = float(price)
        total_proceeds = price * volume

        ticker = ticker.upper().strip()

        valid_ticker = await db.prisma.ticker.find_unique(
            where={"ticker": ticker.replace(".", "-")}
        )

        if valid_ticker is None:
            return f"Invalid ticker {ticker}"

        async with db.prisma.tx() as transaction:
            portfolio = await transaction.portfolio.find_unique(where={"botId": bot_id})
            if portfolio is None:
                raise ValueError("Portfolio not found")

            existing = await transaction.position.find_unique(
                where={
                    "portfolioId_ticker": {
                        "portfolioId": portfolio.id,
                        "ticker": ticker,
                    }
                }
            )

            if existing is None:
                return f"No position found for {ticker}"

            if existing.volume < volume:
                return (
                    f"Not enough shares to sell {volume} shares of {ticker}. "
                    f"Current volume is {existing.volume}."
                )

            portfolio.cash += total_proceeds
            await transaction.portfolio.update(
                where={"botId": bot_id}, data={"cash": portfolio.cash}
            )

            new_volume = existing.volume - volume

            if new_volume == 0:
                await transaction.position.delete(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    }
                )
            else:
                await transaction.position.update(
                    where={
                        "portfolioId_ticker": {
                            "portfolioId": portfolio.id,
                            "ticker": ticker,
                        }
                    },
                    data=PositionUpdateInput(
                        volume=new_volume,
                    ),
                )

            await transaction.trade.create(
                data=TradeCreateInput(
                    rationale=rationale,
                    confidence=confidence,
                    type=TradeType.SELL,
                    price=price,
                    ticker=ticker,
                    amount=volume,
                    runId=runId,
                    botId=bot_id,
                    realizedPL=price * volume - existing.cost * volume,
                    realizedPLPercent=(price - existing.cost) / existing.cost,
                )
            )

            if new_volume == 0:
                return (
                    f"Successfully sold {volume} shares of {ticker} at {price} per share. "
                    f"Position closed."
                )

            return (
                f"Successfully sold {volume} shares of {ticker} at {price} per share. "
                f"Current volume is {new_volume} "
                f"with average cost {utils.format_float(existing.cost)}"
            )


class RecommendStockAct(Action):
    @property
    def name(self):
        return "recommend_stock"

    @utils.async_retry()
    async def arun(
        self,
        ticker: str,
        amount: float,
        rationale: str,
        confidence: float,
        trade_type: TradeType,
        role: Role,
        run_id: str,
        bot_id: str,
    ) -> str:
        """
        Recommend a stock to buy or sell.

        Args:
            ticker: Stock ticker, e.g. 'AAPL'
            amount: Amount of stock to buy or sell
            rationale: Rationale for the recommendation
            confidence: Confidence in the recommendation (0.0-1.0)

        Returns:
            A markdown-formatted table with the recommendation.
        """

        if not (0.0 <= confidence <= 1.0):
            return "Confidence must be between 0.0 and 1.0"

        await db.prisma.recommend.create(
            data=RecommendCreateInput(
                ticker=ticker,
                type=trade_type,
                amount=amount,
                rationale=rationale,
                confidence=confidence,
                role=role,
                runId=run_id,
                botId=bot_id,
            )
        )

        return (
            f"{role.value} recommended {trade_type.value} {amount} shares of {ticker}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Rationale: {rationale}"
        )


class GetAnalystsRecommendationsAct(Action):
    @property
    def name(self):
        return "get_analysts_recommendations"

    @utils.async_retry()
    async def arun(
        self,
        run_id: str,
    ) -> str:
        """
        Get analysts recommendations.

        Args:
            run_id: Run ID
            bot_id: Bot ID

        Returns:
            A markdown-formatted table with the recommendations.
        """

        recommendations = await db.prisma.recommend.find_many(
            where={
                "runId": run_id,
            },
            order={
                "role": "asc",
            },
        )

        return format_recommendations_markdown(recommendations)


class WriteDownTickersToReviewAct(Action):
    @property
    def name(self):
        return "write_down_tickers_to_review"

    @async_retry()
    async def arun(self, run_id: str, tickers: list[str]) -> str:
        runId = run_id

        run = await db.prisma.run.find_unique(
            where={
                "id": runId,
            }
        )
        if not run:
            return f"Run with id {runId} not found"

        if not run.tickers:
            await db.prisma.run.update(
                where={
                    "id": runId,
                },
                data={
                    "tickers": ",".join(tickers),
                },
            )
            return f"Tickers {tickers} written down to review"
        else:
            existing_tickers = set([t.strip().upper() for t in run.tickers.split(",")])
            new_tickers = set([t.strip().upper() for t in tickers])

            if existing_tickers != new_tickers:
                return (
                    "The tickers to review are not the same as the ones that user specified."
                    "Please check the tickers and try again. "
                    "The tickers that user specified are: "
                    f"{', '.join(new_tickers)}"
                    "The tickers that you are going to write down are: "
                    f"{', '.join(new_tickers)}"
                )
            else:
                return f"Tickers {tickers} written down to review"
```

src/agents/equity_research_analyst/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_equity_research_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.EQUITY_RESEARCH_ANALYST
    )
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.do_google_equity_research,
            tools.get_latest_equity_news,
            tools.get_price_trend,
            tools.get_recommend_stock_tool(Role.EQUITY_RESEARCH_ANALYST),
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.EQUITY_RESEARCH_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/agents/chief_investment_officer/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src.tools.handoff_tools import handoff_to_specialist
from src.middleware import (
    LoggingMiddleware,
    todo_list_middleware,
    # summarization_middleware,
)
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_chief_investment_officer_agent(context: Context):
    system_prompt = await build_agent_system_prompt(
        context, Role.CHIEF_INVESTMENT_OFFICER
    )
    langchain_model = get_model(context.llm_model)

    agent = create_agent(
        model=langchain_model,
        tools=[
            # Portfolio management tools
            # tools.list_current_positions,
            tools.get_portfolio_performance_analysis,
            tools.get_user_investment_strategy,
            # tools.buy_stock,
            # tools.sell_stock,
            # tools.get_latest_quotes,
            # tools.get_latest_quote,
            tools.get_analysts_recommendations,
            tools.get_recommend_stock_tool(Role.CHIEF_INVESTMENT_OFFICER),
            tools.send_summary_email_tool,
            tools.write_summary_report,
            tools.write_down_tickers_to_review,
            tools.get_market_status,
            tools.get_historical_reviewed_tickers,
            handoff_to_specialist,
        ],
        middleware=[
            todo_list_middleware,  # type: ignore
            # summarization_middleware,
            LoggingMiddleware("CHIEF_INVESTMENT_OFFICER"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/agents/market_analyst/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_market_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.MARKET_ANALYST)
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_latest_market_news,
            tools.do_google_market_research,
            tools.get_etf_live_historical_price_change,
            tools.get_stock_live_historical_price_change,
            tools.list_current_positions,
            tools.get_most_active_stockers,
            tools.get_portfolio_performance_analysis,
            tools.get_user_investment_strategy,
            tools.get_recommend_stock_tool(Role.MARKET_ANALYST),
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.MARKET_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/agents/trading_executive/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_trading_executor_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.TRADING_EXECUTOR)
    langchain_model = get_model(context.llm_model)

    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.list_current_positions,
            # tools.get_user_investment_strategy,
            tools.buy_stock,
            tools.sell_stock,
            # tools.get_latest_quotes,
            # tools.get_latest_quote,
            tools.get_market_status,
        ],
        middleware=[
            middleware.LoggingMiddleware("TRADING_EXECUTOR"),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/agents/fundamental_analyst/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_fundamental_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.FUNDAMENTAL_ANALYST)
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_fundamental_data,
            tools.get_stock_live_historical_price_change,
            tools.get_price_trend,
            tools.get_recommend_stock_tool(Role.FUNDAMENTAL_ANALYST),
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.FUNDAMENTAL_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/agents/risk_analyst/agent.py
```
from langchain.agents import create_agent
from prisma.enums import Role
from src import tools
from src import middleware
from src.models import get_model
from src.context import Context
from src.prompt import build_agent_system_prompt


async def build_risk_analyst_agent(context: Context):
    system_prompt = await build_agent_system_prompt(context, Role.RISK_ANALYST)
    langchain_model = get_model(context.llm_model)
    agent = create_agent(
        model=langchain_model,
        tools=[
            tools.get_price_trend,
            tools.get_fundamental_risk_data,
            tools.get_price_risk_indicators,
            tools.get_volatility_risk_indicators,
            tools.get_stock_live_historical_price_change,
            tools.get_recommend_stock_tool(Role.RISK_ANALYST),
        ],
        middleware=[
            # middleware.summarization_middleware,  # type: ignore
            # middleware.todo_list_middleware,
            middleware.LoggingMiddleware(Role.RISK_ANALYST.value),
        ],
        system_prompt=system_prompt,
        context_schema=Context,
    )

    return agent
```

src/services/alpaca/__init__.py
```
from src.services.alpaca.api_historical_bars import get_historical_price_bars
from src.services.alpaca.api_most_active_stockers import get_most_active_stocks
from src.services.alpaca.api_snapshots import get_snapshots
from src.services.alpaca.api_news import get_news
from src.services.alpaca.api_latest_quotes import get_latest_quotes

__all__ = [
    "get_historical_price_bars",
    "get_most_active_stocks",
    "get_snapshots",
    "get_news",
    "get_latest_quotes",
]
```

src/services/alpaca/api_client.py
```
from typing import Sequence, TypeVar, Generic, cast, Any
import httpx
import dotenv
from src.services.utils import async_retry_on_status_code
from src.utils import get_env

dotenv.load_dotenv()


BASE_URL = "https://data.alpaca.markets"

ALPACA_API_KEY_ID = get_env("ALPACA_API_KEY")
ALPACA_API_SECRET_KEY = get_env("ALPACA_API_SECRET")


T = TypeVar("T")


class AlpacaAPIClient(Generic[T]):  # pylint:disable=too-few-public-methods
    """Async Alpaca Data API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: https://data.alpaca.markets
    - Adds required Alpaca auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
    - Exponential backoff with jitter

    Usage:
        client = AlpacaAPIClient(endpoint="/v2/stocks/snapshots")
        data = await client.get(symbols=["AAPL", "MSFT"])
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    @async_retry_on_status_code(status_codes=[429, 500, 502, 503, 504])
    async def get(
        self,
        *,
        symbols: Sequence[str] | str | None = None,
        endpoint: str | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> T:
        """Perform a GET request with retries.

        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
            "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
            "accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)

        if symbols:
            params = params or {}
            params["symbols"] = _normalize_symbols(symbols)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=self.timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


def _normalize_symbols(symbols: Sequence[str] | str) -> str:
    if isinstance(symbols, str):
        s = symbols.strip()
        if not s:
            raise ValueError("symbols must not be empty")
        return s
    cleaned = [sym.strip() for sym in symbols if sym and sym.strip()]
    if not cleaned:
        raise ValueError("symbols must contain at least one non-empty symbol")
    return ",".join(cleaned)


__all__ = [
    "AlpacaAPIClient",
]
```

src/services/alpaca/api_historical_bars.py
```
import json
import os
from typing import Dict, TypedDict, Union

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError, redis_cache

load_dotenv()

Number = float | int

Bar = TypedDict(
    "Bar",
    {
        "c": Number,
        "h": Number,
        "l": Number,
        "n": int,
        "o": Number,
        "t": str,
        "v": int,
        "vw": Number,
    },
)


class HistoricalBars(TypedDict):
    bars: Dict[str, list[Bar]]
    next_page_token: str | None


historicalBarsAPI: AlpacaAPIClient[HistoricalBars] = AlpacaAPIClient(
    endpoint="/v2/stocks/bars"
)


class PriceBar(TypedDict):
    close_price: Union[int, float]
    high_price: Union[int, float]
    low_price: Union[int, float]
    trade_count: int
    open_price: Union[int, float]
    timestamp: str
    volume: int
    volume_weighted_average_price: Union[int, float]


example = {
    "bars": {
        "AAPL": [
            {
                "c": 272.185,
                "h": 275.24,
                "l": 272.185,
                "n": 66941,
                "o": 275,
                "t": "2025-11-12T05:00:00Z",
                "v": 8039893,
                "vw": 273.512065,
            }
        ],
        "TSLA": [
            {
                "c": 438.255,
                "h": 442.329,
                "l": 436.92,
                "n": 130571,
                "o": 442.15,
                "t": "2025-11-12T05:00:00Z",
                "v": 5538695,
                "vw": 438.975835,
            }
        ],
    },
    "next_page_token": None,
}


def _rename_keys(bars: Dict[str, list[Bar]]) -> Dict[str, list[PriceBar]]:
    new_bars = {}
    for key, bar_list in bars.items():
        new_bars[key] = []
        for _bar in bar_list:
            new_bar = PriceBar(
                close_price=_bar["c"],
                high_price=_bar["h"],
                low_price=_bar["l"],
                trade_count=_bar["n"],
                open_price=_bar["o"],
                timestamp=_bar["t"],
                volume=_bar["v"],
                volume_weighted_average_price=_bar["vw"],
            )
            new_bars[key].append(new_bar)
    return new_bars


@redis_cache(function_name="get_historical_price_bars", ttl=3600)
async def _get_price_bar(
    *,
    symbols: list[str],
    timeframe: str,
    start: str,
    end: str,
    page_token=None,
    limit=200,
    sort: str = "asc",
):
    params = {
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "sort": sort,
        "limit": limit,
    }
    if page_token:
        params["page_token"] = page_token

    api_response = await historicalBarsAPI.get(
        symbols=symbols,
        params=params,
    )
    return api_response


# @redis_cache(function_name="get_historical_price_bars", ttl=3600)
# @in_db_cache(function_name="get_historical_price_bars", ttl=3600)
async def get_historical_price_bars(
    *, symbols: list[str], timeframe: str, start: str, end: str, sort: str = "asc"
) -> Dict[str, list[PriceBar]]:
    """Get historical bars for a list of symbols.

    Args:
        symbols (list[str]): A list of symbols to get historical bars for.
        timeframe (str): The timeframe to get historical bars for.
        start (str): The start time to get historical bars for.
        end (str): The end time to get historical bars for.
        sort (str, optional): The sort order of the historical bars. Defaults to "asc".
        next_page_token (str | None, optional): The next page token to get historical bars for. Defaults to None.

    Returns:
        PriceHistoricalBars: A dictionary of historical bars for each symbol.
    """

    api_response = await _get_price_bar(
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        sort=sort,
    )

    next_page_token = api_response["next_page_token"]
    bars = api_response["bars"]

    while next_page_token is not None:
        next_api_response = await _get_price_bar(
            symbols=symbols,
            timeframe=timeframe,
            start=start,
            end=end,
            sort=sort,
            page_token=next_page_token,
        )

        if next_api_response["bars"]:
            for key, bar_list in next_api_response["bars"].items():
                if key not in bars:
                    bars[key] = []
                bars[key].extend(bar_list)
        next_page_token = next_api_response["next_page_token"]

    bars = _rename_keys(bars)
    return bars


__all__ = ["get_historical_price_bars", "PriceBar"]


async def _run() -> None:
    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_historical_price_bars(
            symbols=["AAPL", "MSFT"],
            timeframe="1Day",
            start="2024-11-12T05:00:00Z",
            end="2025-11-12T05:00:00Z",
            sort="desc",
        )
        print("Historical bars response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_historical_bars
    import asyncio

    asyncio.run(_run())
```

src/services/alpaca/api_latest_quotes.py
```
import json
import os
from typing import Dict, TypedDict, Union

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

load_dotenv()

example = {
    "quotes": {
        "AAPL": {
            "ap": 283.18,
            "as": 200,
            "ax": "P",
            "bp": 283.01,
            "bs": 21700,
            "bx": "P",
            "c": ["R"],
            "t": "2025-12-02T00:59:50.364749005Z",
            "z": "C",
        }
    }
}


Quote = TypedDict(
    "Quote",
    {
        "ap": Union[int, float],
        "as": int,
        "ax": str,
        "bp": Union[int, float],
        "bs": int,
        "bx": str,
        "c": list[str],
        "t": str,
        "z": str,
    },
)

QuotesResponse = TypedDict(
    "QuotesResponse",
    {
        "quotes": Dict[str, Quote],
    },
)

QuoteHuman = TypedDict(
    "QuoteHuman",
    {
        "ask_price": Union[int, float],
        "ask_size": int,
        "ask_exchange": str,
        "bid_price": Union[int, float],
        "bid_size": int,
        "bid_exchange": str,
        "conditions": list[str],
        "timestamp": str,
        "market_center": str,
    },
)

QuotesHumanResponse = TypedDict(
    "QuotesHumanResponse",
    {
        "quotes": Dict[str, QuoteHuman],
    },
)


def _rename_key(quote: Quote):
    quote_human: QuoteHuman = {
        "ask_price": quote["ap"],
        "ask_size": quote["as"],
        "ask_exchange": quote["ax"],
        "bid_price": quote["bp"],
        "bid_size": quote["bs"],
        "bid_exchange": quote["bx"],
        "conditions": quote["c"],
        "timestamp": quote["t"],
        "market_center": quote["z"],
    }
    return quote_human


latestQuotesAPI: AlpacaAPIClient[QuotesResponse] = AlpacaAPIClient(
    endpoint="/v2/stocks/quotes/latest"
)


async def get_latest_quotes(
    symbols: list[str],
):
    api_response = await latestQuotesAPI.get(
        symbols=symbols,
    )
    response: QuotesHumanResponse = {
        "quotes": {
            symbol: _rename_key(quote)
            for symbol, quote in api_response["quotes"].items()
        }
    }

    return response


__all__ = ["get_latest_quotes"]


async def _run() -> None:
    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_latest_quotes(
            symbols=["NVDA"],
        )
        print("Latest quotes response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_latest_quotes
    import asyncio

    asyncio.run(_run())
```

src/services/alpaca/api_most_active_stockers.py
```
from typing import Literal, TypedDict
from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

ActiveStock = TypedDict(
    "ActiveStock", {"symbol": str, "trade_count": int, "volume": int}
)

MostActiveStocksResponse = TypedDict(
    "MostActiveStocksResponse", {"last_updated": str, "most_actives": list[ActiveStock]}
)


mostActiveStocksAPI: AlpacaAPIClient[MostActiveStocksResponse] = AlpacaAPIClient(
    endpoint="/v1beta1/screener/stocks/most-actives"
)


async def get_most_active_stocks(
    by: Literal["trades", "volume"] = "trades", top: int = 20
):
    most_active_stocks = await mostActiveStocksAPI.get(params={"by": by, "top": top})
    return most_active_stocks


__all__ = [
    "get_most_active_stocks",
]


async def _run() -> None:
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_most_active_stocks(by="trades", top=30)
        print("Most active stocks response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
        print(
            f"Symbols returned: {', '.join([stock['symbol'] for stock in data['most_actives']])}"
        )
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_most_active_stockers
    import asyncio

    asyncio.run(_run())
```

src/services/alpaca/api_news.py
```
import json
import os
from typing import TypedDict

from dotenv import load_dotenv

from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

load_dotenv()

example = {
    "author": "Anusuya Lahiri",
    "content": "...",
    "created_at": "2025-11-28T18:25:15Z",
    "headline": "Intel Stock Soars 7% On Report It Could Soon Build Chips For Apple's Macs",
    "id": 49121202,
    "images": [
        {
            "size": "large",
            "url": "https://cdn.benzinga.com/files/imagecache/2048x1536xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
        {
            "size": "small",
            "url": "https://cdn.benzinga.com/files/imagecache/1024x768xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
        {
            "size": "thumb",
            "url": "https://cdn.benzinga.com/files/imagecache/250x187xUP/images/story/2025/11/28/Intel-Corp.jpeg",
        },
    ],
    "source": "benzinga",
    "summary": "Intel&#39;s stock gains momentum as Apple considers it as a supplier for their next-gen M chips, signaling a potential partnership.",
    "symbols": ["AAPL", "INTC", "TSM"],
    "updated_at": "2025-11-28T18:25:16Z",
    "url": "https://www.benzinga.com/analyst-stock-ratings/analyst-color/25/11/49121202/intel-stock-soars-7-on-report-it-could-soon-build-chips-for-apples-macs",
}

Image = TypedDict(
    "Image",
    {
        "size": str,
        "url": str,
    },
)

News = TypedDict(
    "News",
    {
        "id": int,
        "author": str,
        "content": str,
        "created_at": str,
        "headline": str,
        "images": list[Image],
        "source": str,
        "summary": str,
        "symbols": list[str],
        "updated_at": str,
        "url": str,
    },
)


class NewsResponse(TypedDict):
    news: list[News]


NewsAPI: AlpacaAPIClient[NewsResponse] = AlpacaAPIClient(endpoint="/v1beta1/news")

# @redis_cache(function_name="get_historical_price_bars", ttl=3600)
# @in_db_cache(function_name="get_historical_price_bars", ttl=3600)


async def get_news(
    symbols: list[str],
    start: str,
    end: str,
    sort: str = "desc",
    limit: int = 12,
) -> NewsResponse:
    response = await NewsAPI.get(
        params={
            "symbols": ",".join(symbols),
            "start": start,
            "end": end,
            "sort": sort,
            "limit": limit,
            "include_content": True,
        }
    )
    return response


__all__ = ["get_news", "News"]


async def _run() -> None:
    from datetime import date, timedelta

    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        start = (date.today() - timedelta(days=5)).isoformat()
        end = date.today().isoformat()
        data = await get_news(
            symbols=["AAPL", "MSFT"],
            start=start,
            end=end,
            sort="desc",
        )
        print("News response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_news
    import asyncio

    asyncio.run(_run())
```

src/services/alpaca/api_snapshots.py
```
from typing import Dict, TypedDict
from src.services.alpaca.api_client import AlpacaAPIClient
from src.services.utils import APIError

Number = float | int

DailyBar = TypedDict(
    "DailyBar",
    {
        "c": Number,
        "h": Number,
        "l": Number,
        "n": int,
        "o": Number,
        "t": str,
        "v": int,
        "vw": Number,
    },
)

MinuteBar = TypedDict(
    "MinuteBar",
    {
        "c": Number,
        "h": Number,
        "l": Number,
        "n": int,
        "o": Number,
        "t": str,
        "v": int,
        "vw": Number,
    },
)

LatestQuote = TypedDict(
    "LatestQuote",
    {
        "ap": Number,
        "as": int,
        "ax": str,
        "bp": Number,
        "bs": int,
        "bx": str,
        "c": list[str],
        "t": str,
        "z": str,
    },
)

LatestTrade = TypedDict(
    "LatestTrade",
    {
        "c": list[str],
        "i": int,
        "p": Number,
        "s": int,
        "t": str,
        "x": str,
        "z": str,
    },
)

Snapshot = TypedDict(
    "Snapshot",
    {
        "dailyBar": DailyBar,
        "latestQuote": LatestQuote,
        "latestTrade": LatestTrade,
        "minuteBar": MinuteBar,
        "prevDailyBar": DailyBar,
    },
)

SnapshotsResponse = Dict[str, Snapshot]


snapshotsAPI: AlpacaAPIClient[SnapshotsResponse] = AlpacaAPIClient(
    endpoint="/v2/stocks/snapshots"
)


async def get_snapshots(symbols: list[str]) -> SnapshotsResponse:
    return await snapshotsAPI.get(symbols=symbols)


__all__ = [
    "get_snapshots",
]


async def _run() -> None:
    import json
    import os
    from dotenv import load_dotenv

    load_dotenv()

    missing = [
        name
        for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this snapshots test."
        )
        return

    try:
        data = await get_snapshots(symbols=["AAPL", "MSFT"])
        print("Snapshots response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
        print(f"Symbols returned: {', '.join(list(data.keys()))}")
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify ALPACA_API_KEY and ALPACA_API_SECRET are correct "
            "and have access to the Alpaca Data API."
        )


if __name__ == "__main__":
    # python -m src.services.alpaca.api_snapshots
    import asyncio

    asyncio.run(_run())
```

src/services/alpaca/sdk_trading_client.py
```
import os
import dotenv
from alpaca.trading.client import TradingClient  # pylint: disable=import-error,no-name-in-module

dotenv.load_dotenv()

client = TradingClient(
    api_key=os.getenv("ALPACA_API_KEY"),
    secret_key=os.getenv("ALPACA_API_SECRET"),
    paper=True,
)

__all__ = ["client"]
```

src/services/alpaca/typing.py
```
from src.services.alpaca.api_historical_bars import PriceBar
from src.services.alpaca.api_news import News

__all__ = ["PriceBar", "News"]
```

src/services/sandx_ai/__init__.py
```
from src import db
from src.services.sandx_ai.api_position import list_positions
from src.services.sandx_ai.api_portfolio_timeline_value import get_timeline_values


async def get_cash_balance(bot_id: str):
    portfolio = await db.prisma.portfolio.find_unique(where={"botId": bot_id})
    if portfolio is None:
        raise ValueError("Portfolio not found")
    return portfolio.cash


__all__ = [
    "list_positions",
    "get_timeline_values",
    "get_cash_balance",
]
```

src/services/sandx_ai/api_client.py
```
import os
from typing import Any, Generic, Mapping, TypeVar, cast
import dotenv
import httpx
from src.services.utils import async_retry_on_status_code
from src.utils import get_env

dotenv.load_dotenv()


BASE_URL = os.getenv("SANDX_AI_URL", "http://localhost:3000/api")

API_KEY = get_env("API_KEY")


T = TypeVar("T")


class SandxAPIClient(Generic[T]):  # pylint: disable=too-few-public-methods
    """Async Sandx AI API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: http://localhost:3000
    - Adds required Sandx AI auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    @async_retry_on_status_code(status_codes=[429, 500, 502, 503, 504])
    async def get(
        self,
        *,
        endpoint: str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> T:
        """Perform a GET request with retries.

        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "Authorization": f"Bearer {API_KEY}",
            "accept": "application/json",
        }
        if headers:
            merged_headers.update(headers)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=self.timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


__all__ = [
    "SandxAPIClient",
]
```

src/services/sandx_ai/api_portfolio_timeline_value.py
```
import json
import os
from datetime import datetime, timedelta
from typing import List, TypedDict
from dotenv import load_dotenv
from src.services.sandx_ai.api_client import SandxAPIClient
from src.services.utils import APIError, redis_cache

load_dotenv()


TimelineValue = TypedDict(
    "TimelineValue",
    {
        "date": str,
        "value": float,
    },
)


api_client = SandxAPIClient[list[TimelineValue]]("/tools/portfolio/timeline-value")


@redis_cache(ttl=10, function_name="get_timeline_values")
async def get_timeline_values(bot_id: str) -> List[TimelineValue]:
    from_date = datetime.now() - timedelta(days=365) - timedelta(days=7)
    from_date_str = from_date.strftime("%Y-%m-%d %H:%M:%S")
    timeline_values = await api_client.get(
        params={"botId": bot_id, "from": from_date_str}
    )
    return timeline_values


__all__ = [
    "get_timeline_values",
]


async def _run() -> None:
    missing = [name for name in ["API_KEY"] if not os.environ.get(name)]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this positions test."
        )
        return

    try:
        data = await get_timeline_values(bot_id="7cf5cfb1-b30d-4d82-9363-af2096f2d926")
        print("Timeline values response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify API_KEY is correct "
            "and has access to the Sandx AI API."
        )


if __name__ == "__main__":
    # python -m src.services.sandx_ai.api_portfolio_timeline_value
    import asyncio

    asyncio.run(_run())
```

src/services/sandx_ai/api_position.py
```
import json
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from src.services.sandx_ai.api_client import SandxAPIClient
from src.services.utils import APIError, redis_cache

load_dotenv()


PositionItem = TypedDict(
    "PositionItem",
    {
        "allocation": float,
        "currentPrice": float,
        "ptcChangeInPrice": Annotated[
            float, "The percentage change in price relative to the open price"
        ],
        "currentValue": Annotated[
            float, "The total current value of the position in the portfolio"
        ],
        "id": str,
        "ticker": str,
        "volume": int,
        "cost": float,
        "pnl": float,
        "pnlPercent": float,
    },
)


class Position(TypedDict):
    allocation: Annotated[
        float, "The percentage allocation of the position in the portfolio"
    ]
    current_price: Annotated[float, "The current price of the stock position per share"]
    ptc_change_in_price: Annotated[
        float, "The percentage change in price relative to the open price"
    ]
    current_value: Annotated[
        float, "The total current value of the position in the portfolio"
    ]
    ticker: Annotated[str, "The stock ticker of the position"]
    volume: Annotated[int, "The total share of the position in the portfolio"]
    cost: Annotated[float, "The average cost of the position in the portfolio"]
    pnl: Annotated[float, "Profit and Loss of the position in the portfolio"]
    pnl_percent: Annotated[
        float, "Profit and Loss percentage of the position in the portfolio"
    ]


api_client = SandxAPIClient[list[PositionItem]]("/tools/portfolio/positions")


@redis_cache(ttl=10, function_name="list_positions")
async def list_positions(bot_id: str) -> Sequence[Position]:
    positions = await api_client.get(params={"botId": bot_id})

    readable_positions: list[Position] = []
    for position in positions:
        _dict: Position = {
            "allocation": position["allocation"],
            "current_price": position["currentPrice"],
            "ptc_change_in_price": position["ptcChangeInPrice"],
            "current_value": position["currentValue"],
            "ticker": position["ticker"],
            "volume": position["volume"],
            "cost": position["cost"],
            "pnl": position["pnl"],
            "pnl_percent": position["pnlPercent"],
        }
        readable_positions.append(_dict)
    return readable_positions


__all__ = [
    "list_positions",
]


async def _run() -> None:
    missing = [name for name in ["API_KEY"] if not os.environ.get(name)]
    if missing:
        print(
            f"Missing environment variables: {', '.join(missing)}. "
            "Set them to run this positions test."
        )
        return

    try:
        data = await list_positions(bot_id="7cf5cfb1-b30d-4d82-9363-af2096f2d926")
        print("Positions response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")
        print(
            "If you see 'Authorization Required', verify API_KEY is correct "
            "and has access to the Sandx AI API."
        )


if __name__ == "__main__":
    # python -m src.services.sandx_ai.api_position
    import asyncio

    asyncio.run(_run())
```

src/services/sandx_ai/typing.py
```
from src.services.sandx_ai.api_position import Position
from src.services.sandx_ai.api_portfolio_timeline_value import TimelineValue

__all__ = ["Position", "TimelineValue"]
```

src/services/yfinance/api_info.py
```
import yfinance as yf
from src.utils import async_wrap
from src.services.utils import redis_cache


@async_wrap()
def get_ticker_info(ticker: str) -> dict:
    yf_ticker = yf.Ticker(ticker=ticker)
    info = yf_ticker.info
    return info


@redis_cache(function_name="get_ticker_info", ttl=60 * 60 * 24)
async def async_get_ticker_info(ticker: str) -> dict:
    info = await get_ticker_info(ticker)  # type: ignore
    return info
```

src/services/tradingeconomics/__init__.py
```
from src.services.tradingeconomics.api_market_news import get_news

__all__ = ["get_news"]
```

src/services/tradingeconomics/api_client.py
```
import random
from typing import Any, Mapping, Sequence, TypeVar, Generic, cast
import httpx
import dotenv
from src.services.utils import async_retry_on_status_code

dotenv.load_dotenv()


BASE_URL = "https://tradingeconomics.com"


T = TypeVar("T")


class TradingEconomicsAPIClient(Generic[T]):  # pylint: disable=too-few-public-methods
    """Async TradingEconomics API client with exponential retry.

    Mirrors the provided TypeScript Axios client:
    - Base URL: https://tradingeconomics.com
    - Adds required auth headers
    - Retries on network errors and statuses in {429, 500, 502, 503, 504}
    - Exponential backoff with jitter

    Usage:
        client = TradingEconomicsAPIClient(endpoint="/ws/stream.ashx")
        # https://tradingeconomics.com/ws/stream.ashx?start=15&size=20&c=united states
        data = await client.get(params={"start": "15", "size": "20", "c": "united states"})
    """

    def __init__(
        self,
        endpoint: str,
        *,
        retries: int = 10,
        retry_statuses: Sequence[int] | None = None,
        base_delay_seconds: float = 0.3,
        max_delay_seconds: float = 30.0,
        jitter_seconds: float = 0.2,
    ) -> None:
        self.endpoint = endpoint
        self.retries = retries
        self.retry_statuses = set(retry_statuses or (429, 500, 502, 503, 504))
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_seconds = jitter_seconds

    def _retry_delay(self, attempt: int) -> float:
        # Exponential backoff with jitter, similar to axiosRetry.exponentialDelay
        delay = min(self.base_delay_seconds * (2**attempt), self.max_delay_seconds)
        jitter = random.uniform(0.0, self.jitter_seconds)
        return delay + jitter

    @async_retry_on_status_code()
    async def get(
        self,
        *,
        endpoint: str | None = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> T:
        """Perform a GET request with retries.
        - `endpoint`: optionally append to the base endpoint set at init
        - `params`: query parameters
        - `headers`: additional headers merged with auth headers
        - `timeout`: request timeout in seconds
        """

        path = self.endpoint + (endpoint or "")

        merged_headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:144.0) Gecko/20100101 Firefox/144.0",
            "accept": "application/json, text/javascript, */*; q=0.01",
        }

        if headers:
            merged_headers.update(headers)

        async with httpx.AsyncClient(base_url=BASE_URL, timeout=timeout) as client:
            resp = await client.get(path, params=params, headers=merged_headers)
            resp.raise_for_status()
            data = resp.json()
            return cast(T, data)


__all__ = [
    "TradingEconomicsAPIClient",
]
```

src/services/tradingeconomics/api_market_news.py
```
import json
from datetime import datetime, timezone
from typing import Sequence, TypedDict

from src.services.tradingeconomics.api_client import TradingEconomicsAPIClient
from src.services.utils import APIError, redis_cache

# https://tradingeconomics.com/ws/stream.ashx?start=15&size=20&c=united states

RawNews = TypedDict(
    "RawNews",
    {
        "ID": int,
        "title": str,
        "description": str,
        "url": str,
        "author": str,
        "country": str,
        "category": str,
        "image": str | None,
        "importance": int,
        "date": str,
        "expiration": str,
        "html": str | None,
        "type": str | None,
    },
)


class News(TypedDict):
    ID: int  # pylint: disable=invalid-name
    title: str
    description: str
    country: str
    category: str
    importance: int
    date: str
    expiration: str
    time_ago: str


api_client: TradingEconomicsAPIClient[Sequence[RawNews]] = TradingEconomicsAPIClient(
    endpoint="/ws/stream.ashx"
)


def _format_relative_time(date_str: str, *, now: datetime | None = None) -> str:  # pylint: disable=too-many-return-statements
    """Return human-friendly relative time (e.g., "4 hours ago").

    Expects ISO-8601 datetime string. If timezone is missing, assumes UTC.
    Handles 'Z' suffix by normalizing to '+00:00'. Falls back gracefully
    to a readable string if parsing fails.
    """
    s = date_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # Fallback to a readable format without relative calculation
        return date_str.replace("T", " ")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    ref = now if now is not None else datetime.now(timezone.utc)
    diff_seconds = int((ref - dt).total_seconds())
    future = diff_seconds < 0
    seconds = abs(diff_seconds)

    if seconds < 60:
        return "just now" if not future else "in less than a minute"

    minutes = seconds // 60
    if minutes < 60:
        unit = "minute" if minutes == 1 else "minutes"
        return f"{minutes} {unit} ago" if not future else f"in {minutes} {unit}"

    hours = minutes // 60
    if hours < 24:
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} ago" if not future else f"in {hours} {unit}"

    days = hours // 24
    if days < 7:
        unit = "day" if days == 1 else "days"
        return f"{days} {unit} ago" if not future else f"in {days} {unit}"

    weeks = days // 7
    if weeks < 5:
        unit = "week" if weeks == 1 else "weeks"
        return f"{weeks} {unit} ago" if not future else f"in {weeks} {unit}"

    months = days // 30
    if months < 12:
        unit = "month" if months == 1 else "months"
        return f"{months} {unit} ago" if not future else f"in {months} {unit}"

    years = days // 365
    unit = "year" if years == 1 else "years"
    return f"{years} {unit} ago" if not future else f"in {years} {unit}"


async def get_news() -> Sequence[News]:
    @redis_cache(function_name="tradingeconomics_news", ttl=60 * 5)
    async def _get():
        raw_news_list = await api_client.get(
            params={"c": "united states", "start": 0, "size": 30}, timeout=15.0
        )
        return raw_news_list

    raw_news_list = await _get()
    news_list: Sequence[News] = []
    for new in raw_news_list:
        if new["importance"] > 0:
            news_list.append(
                News(
                    ID=new["ID"],
                    title=new["title"],
                    description=new["description"],
                    country=new["country"],
                    category=new["category"],
                    importance=new["importance"],
                    date=new["date"],
                    expiration=new["expiration"],
                    time_ago=_format_relative_time(new["date"]),
                )
            )
    return news_list


__all__ = [
    "get_news",
]


async def _run() -> None:
    try:
        data = await get_news()
        print("News response (truncated):")
        print(json.dumps(data, indent=2)[:2000])
    except APIError as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    # python -m src.services.tradingeconomics.api_market_news
    import asyncio

    asyncio.run(_run())
```

src/tools_adaptors/utils/__init__.py
```
from src.tools_adaptors.utils.portfolio_timeline_value import (
    analyze_timeline_value,
    create_performance_narrative,
)
from src.tools_adaptors.utils.fundamental_data_utils import (
    categorize_fundamental_data,
    get_categorized_metrics,
    format_fundamentals_markdown,
    preprocess_info_dict,
)
from src.tools_adaptors.utils.risk_analysis import (
    calculate_volatility_risk,
    format_volatility_risk_markdown,
    calculate_price_risk,
    format_price_risk_markdown,
)
from src.tools_adaptors.utils.recommendations import format_recommendations_markdown


__all__ = [
    "analyze_timeline_value",
    "create_performance_narrative",
    "categorize_fundamental_data",
    "get_categorized_metrics",
    "format_fundamentals_markdown",
    "preprocess_info_dict",
    "calculate_volatility_risk",
    "format_volatility_risk_markdown",
    "calculate_price_risk",
    "format_price_risk_markdown",
    "format_recommendations_markdown",
]
```

src/tools_adaptors/utils/fundamental_data_utils.py
```
from src.utils.constants import FUNDAMENTAL_CATEGORIES


def preprocess_info_dict(info_dict):
    if "dividendYield" in info_dict:
        info_dict["dividendYield"] = float(info_dict["dividendYield"]) / 100
    if "fiveYearAvgDividendYield" in info_dict:
        info_dict["fiveYearAvgDividendYield"] = (
            float(info_dict["fiveYearAvgDividendYield"]) / 100
        )
    return info_dict


def categorize_fundamental_data(info_dict, categories_map=FUNDAMENTAL_CATEGORIES):
    """
    Categorize fundamental data from info dictionary into structured categories

    Args:
        info_dict (dict): The yfinance info dictionary

    Returns:
        dict: Structured data with categories containing available metrics
    """
    categorized_data = {}

    for category, metric_keys in categories_map.items():
        category_metrics = {}

        for key in metric_keys:
            if key in info_dict:
                category_metrics[key] = info_dict[key]

        # Only include categories that have data
        if category_metrics:
            categorized_data[category] = category_metrics

    return categorized_data


def get_categorized_metrics(
    info_dict,
    categories_map=FUNDAMENTAL_CATEGORIES,
    format_values=True,
    include_empty=False,
):
    """
    Get categorized metrics with advanced options

    Args:
        info_dict (dict): yfinance info dictionary
        categories_map (dict): Category mapping
        format_values (bool): Whether to format values for display
        include_empty (bool): Whether to include empty categories

    Returns:
        dict: Categorized metrics data
    """

    def format_metric_value(key, value):
        """Helper function to format metric values"""
        if value is None:
            return "N/A"

        # Large monetary values
        if isinstance(value, (int, float)) and abs(value) >= 1e9:
            return f"{value / 1e9:.2f}B"
        elif isinstance(value, (int, float)) and abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M"
        elif isinstance(value, (int, float)) and abs(value) >= 1e3:
            return f"{value / 1e3:.2f}K"

        # Percentages for ratios, margins, yields, returns
        percent_indicators = ["margin", "yield", "return", "growth"]
        if isinstance(value, float) and any(
            indicator in key.lower() for indicator in percent_indicators
        ):
            return f"{value:.2%}"

        # Simple floats
        elif isinstance(value, float):
            return f"{value:.4f}"

        # Integers
        elif isinstance(value, int):
            return f"{value:,}"

        return str(value)

    categorized_data = {}

    for category, metric_keys in categories_map.items():
        category_metrics = {}

        for key in metric_keys:
            if key in info_dict:
                value = info_dict[key]
                if format_values:
                    category_metrics[key] = format_metric_value(key, value)
                else:
                    category_metrics[key] = value

        # Decide whether to include empty categories
        if category_metrics or include_empty:
            categorized_data[category] = category_metrics

    return categorized_data


def format_fundamentals_markdown(categorized_data, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Fundamental Data", ""]

    for category, metrics in categorized_data.items():
        if metrics:
            md.append(f"## {category}")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            for key, value in metrics.items():
                md.append(f"| {key} | {value} |")
            md.append("")

    return "\n".join(md)


__all__ = [
    "categorize_fundamental_data",
    "get_categorized_metrics",
    "format_fundamentals_markdown",
    "preprocess_info_dict",
]
```

src/tools_adaptors/utils/portfolio_timeline_value.py
```
from collections import defaultdict
from datetime import timedelta, date, datetime
from typing import Dict, List, Optional, TypedDict
import numpy as np
from src.services.sandx_ai.typing import TimelineValue


class FormattedTimelineValue(TypedDict):
    date: date
    value: float


class PeriodMetrics(TypedDict):
    period: str
    start_date: str
    end_date: str
    days: int
    starting_value: float
    ending_value: float
    total_return: float
    total_return_percent: float
    average_daily_return: float
    volatility: float
    best_day_percent: float
    worst_day_percent: float
    max_drawdown_percent: float


class CurrentDetails(TypedDict):
    date: str
    portfolio_value: float
    positions_value: float
    cash: float
    cash_percentage: float


class AnalysisSummary(TypedDict):
    analysis_date: str
    total_days_analyzed: int
    date_range: str
    current_portfolio_value: float
    available_periods: List[str]
    total_return_full_period: float
    total_return_percent_full_period: float


class TimeSeriesData(TypedDict):
    dates: List[str]
    portfolio_values: List[float]
    daily_returns: List[float]


class AnalysisResult(TypedDict):
    overall_summary: AnalysisSummary
    period_performance: Dict[str, PeriodMetrics]
    time_series_data: TimeSeriesData


def deduplicate_timeline_by_date(timeline_values: List[TimelineValue]):
    """
    Groups timeline entries by calendar date and keeps only the latest entry per date.

    Args:
        timeline_values (list): List of dicts with keys 'date' (ISO string) and 'value'.

    Returns:
        list: Deduplicated list with one entry per calendar date, converted to date objects.
    """
    grouped: Dict[str, List[TimelineValue]] = defaultdict(list)
    for entry in timeline_values:
        date_key = entry["date"][:10]  # Extract YYYY-MM-DD
        grouped[date_key].append(entry)

    def convert_to_date(entry: TimelineValue) -> FormattedTimelineValue:
        return {
            "date": datetime.fromisoformat(entry["date"].replace("Z", "+00:00")).date(),
            "value": entry["value"],
        }

    transformed_timeline_values = [
        convert_to_date(
            max(
                group,
                key=lambda x: datetime.fromisoformat(x["date"].replace("Z", "+00:00")),
            )
        )
        for group in grouped.values()
    ]
    return transformed_timeline_values


def compute_daily_returns(values_slice: List[float]) -> np.ndarray:
    if len(values_slice) < 2:
        return np.array([], dtype=float)
    arr = np.array(values_slice, dtype=float)
    prev = arr[:-1]
    curr = arr[1:]
    return ((curr - prev) / prev) * 100


def compute_max_drawdown(values_slice: List[float]) -> float:
    if len(values_slice) < 2:
        return 0.0
    arr = np.array(values_slice, dtype=float)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (peaks - arr) / peaks * 100
    return float(np.max(drawdowns)) if drawdowns.size >= 2 else 0.0


def find_period_start_index(
    dates: List[date],  # asc ordering
    current_date: date,
    days_back: int,
) -> Optional[int]:
    target_date = current_date - timedelta(days=days_back)
    for i, snapshot_date in enumerate(dates):
        if snapshot_date >= target_date:
            return i
    return None


def period_metrics(  # pylint: disable=too-many-arguments
    dates: List[date],
    values: List[float],
    start_index: int,
    end_index: int,
    period_name: str,
) -> Optional[PeriodMetrics]:
    if start_index < 0 or start_index >= end_index:
        return None
    if end_index - start_index < 1:
        return None

    start_value = values[start_index]
    end_value = values[end_index]
    values_slice = values[start_index : end_index + 1]

    total_return = end_value - start_value
    total_return_pct = (total_return / start_value) * 100

    daily_returns = compute_daily_returns(values_slice)
    volatility = float(np.std(daily_returns)) if daily_returns.size >= 2 else 0.0
    best_day = float(np.max(daily_returns)) if daily_returns.size else 0.0
    worst_day = float(np.min(daily_returns)) if daily_returns.size else 0.0
    avg_daily_return = float(np.mean(daily_returns)) if daily_returns.size else 0.0
    max_drawdown = compute_max_drawdown(values_slice)

    return PeriodMetrics(
        period=period_name,
        start_date=dates[start_index].strftime("%Y-%m-%d"),
        end_date=dates[end_index].strftime("%Y-%m-%d"),
        days=end_index - start_index + 1,
        starting_value=start_value,
        ending_value=end_value,
        total_return=total_return,
        total_return_percent=total_return_pct,
        average_daily_return=avg_daily_return,
        volatility=volatility,
        best_day_percent=best_day,
        worst_day_percent=worst_day,
        max_drawdown_percent=max_drawdown,
    )


def build_periods(
    dates: List[date],  # asc ordering
    values: List[float],
    current_date: date,
    last_index: int,
) -> Dict[str, PeriodMetrics]:
    periods: Dict[str, PeriodMetrics] = {}

    full = period_metrics(dates, values, 0, last_index, "Full Period")
    if full:
        periods["full_period"] = full

    one_day = period_metrics(dates, values, max(0, last_index - 1), last_index, "1 Day")
    if one_day:
        periods["1_day"] = one_day

    specs = [
        ("1_week", 7, "1 Week"),
        ("1_month", 30, "1 Month"),
        ("3_month", 90, "3 Month"),
        ("1_year", 365, "1 Year"),
    ]
    for key, days_back, label in specs:
        start_idx = find_period_start_index(dates, current_date, days_back)
        if start_idx is None or start_idx >= last_index:
            continue
        m = period_metrics(dates, values, start_idx, last_index, label)
        if m and m["days"] >= 2:
            periods[key] = m
    return periods


def build_overall_summary(
    dates: List[date],
    current_date: date,
    current_value: float,
    periods: Dict[str, PeriodMetrics],
) -> AnalysisSummary:
    summary: AnalysisSummary = {
        "analysis_date": current_date.strftime("%Y-%m-%d"),
        "total_days_analyzed": len(dates),
        "date_range": f"{dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}",
        "current_portfolio_value": current_value,
        "available_periods": list(periods.keys()),
        "total_return_full_period": 0.0,
        "total_return_percent_full_period": 0.0,
    }
    if "full_period" in periods:
        fp = periods["full_period"]
        summary["total_return_full_period"] = fp["total_return"]
        summary["total_return_percent_full_period"] = fp["total_return_percent"]
    return summary


def build_time_series_data(
    dates: List[date], portfolio_values: List[float]
) -> TimeSeriesData:
    dates = dates[-365:]
    portfolio_values = portfolio_values[-365:]

    daily = [0.0] + [
        ((portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1])
        * 100
        for i in range(1, len(portfolio_values))
    ]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "portfolio_values": portfolio_values,
        "daily_returns": daily,
    }


def analyze_timeline_value(
    timeline_values: List[TimelineValue],
) -> Optional[AnalysisResult]:
    if len(timeline_values) < 2:
        return None

    snapshots = deduplicate_timeline_by_date(timeline_values)

    sorted_snapshots = sorted(snapshots, key=lambda x: x["date"])
    dates_local = [s["date"] for s in sorted_snapshots]
    values_local = [s["value"] for s in sorted_snapshots]

    current_date_local = sorted_snapshots[-1]["date"]
    current_value_local = values_local[-1]
    last_index_local = len(sorted_snapshots) - 1

    periods_local = build_periods(
        dates_local, values_local, current_date_local, last_index_local
    )
    overall_summary_local = build_overall_summary(
        dates_local, current_date_local, current_value_local, periods_local
    )
    time_series_data_local = build_time_series_data(dates_local, values_local)

    return {
        "overall_summary": overall_summary_local,
        "period_performance": periods_local,
        "time_series_data": time_series_data_local,
    }


__all__ = ["analyze_timeline_value", "create_performance_narrative"]


def create_performance_narrative(analysis: AnalysisResult) -> str:
    """
    Create a natural language narrative from the performance analysis
    Only includes periods that have sufficient data
    """

    summary = analysis["overall_summary"]
    periods = analysis["period_performance"]

    narrative = f"""
## User's PORTFOLIO PERFORMANCE ANALYSIS
As of {summary["analysis_date"]}

PERFORMANCE SUMMARY:
"""

    # Add performance for each available period
    period_order = ["1_day", "1_week", "1_month", "3_month", "1_year", "full_period"]
    period_names = {
        "1_day": "1 Day",
        "1_week": "1 Week",
        "1_month": "1 Month",
        "3_month": "3 Month",
        "1_year": "1 Year",
        "full_period": "Full Period",
    }

    available_periods_added = False
    for period_key in period_order:
        if period_key in periods:
            available_periods_added = True
            period = periods[period_key]
            arrow = "📈" if period["total_return_percent"] >= 0 else "📉"
            narrative += f"\n{period_names[period_key]} Performance {arrow}:"
            narrative += f"\n  Return: {period['total_return_percent']:+.2f}% ({period['total_return']:+,.2f})"
            narrative += f"\n  Volatility: {period['volatility']:.2f}%"

            # Only show best/worst day if we have multiple days
            if period["days"] > 1:
                narrative += f"\n  Best Day: {period['best_day_percent']:+.2f}%"
                narrative += f"\n  Worst Day: {period['worst_day_percent']:+.2f}%"
                narrative += f"\n  Max Drawdown: {period['max_drawdown_percent']:.2f}%"

            narrative += f"\n  Period: {period['start_date']} to {period['end_date']} ({period['days']} days)"

    if not available_periods_added:
        narrative += "\nNo sufficient data available for period analysis."

    # Add key insights only if we have multiple periods
    valid_periods = [p for p in periods.values() if p["days"] > 1]
    if len(valid_periods) >= 2:
        narrative += "\n\nKEY INSIGHTS:"

        best_period = max(valid_periods, key=lambda x: x["total_return_percent"])
        worst_period = min(valid_periods, key=lambda x: x["total_return_percent"])

        narrative += f"\n- Strongest period: {best_period['period']} ({best_period['total_return_percent']:+.2f}%)"
        narrative += f"\n- Weakest period: {worst_period['period']} ({worst_period['total_return_percent']:+.2f}%)"

        # Risk assessment
        if "full_period" in periods:
            volatility = periods["full_period"]["volatility"]
            if volatility < 2:
                risk_level = "Low"
            elif volatility < 5:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            narrative += f"\n- Risk Level: {risk_level} (Volatility: {volatility:.2f}%)"

    narrative += f"\n\nData Coverage: {summary['date_range']} ({summary['total_days_analyzed']} days of data)"

    return narrative
```

src/tools_adaptors/utils/recommendations.py
```
from prisma.models import Recommend


def format_recommendations_markdown(recommendations: list[Recommend]) -> str:
    """Format recommendations as a markdown table.

    Args:
        recommendations: List of recommendations

    Returns:
        A markdown-formatted table with the recommendations.
    """
    if not recommendations:
        return "No recommendations found."

    headers = ["Ticker", "Type", "Amount", "Confidence", "Analyst", "Rationale"]
    rows = [
        "| "
        + " | ".join(
            [
                rec.ticker,
                rec.type.value if hasattr(rec.type, "value") else str(rec.type),
                str(rec.amount),
                f"{rec.confidence:.2%}",
                rec.role,
                rec.rationale.replace("\n", " ").replace("|", "\\|"),
            ]
        )
        + " |"
        for rec in recommendations
    ]

    header_row = "| " + " | ".join(headers) + " |"
    separator = "|" + "|".join("-" * (len(h) + 2) for h in headers) + "|"

    return "\n".join([header_row, separator] + rows)


__all__ = ["format_recommendations_markdown"]
```

src/tools_adaptors/utils/risk_analysis.py
```
from typing import List, TypedDict, NotRequired

import numpy as np

from src import utils
from src.services.alpaca.typing import PriceBar


class VolatilityRisk(TypedDict, total=False):
    error: NotRequired[str]
    volatility_20d: NotRequired[float]
    volatility_60d: NotRequired[float]
    volatility_252d: NotRequired[float]
    garman_klass_volatility: NotRequired[float]
    parkinson_volatility: NotRequired[float]
    realized_volatility: NotRequired[float]
    volatility_clustering: NotRequired[float]
    max_drawdown: NotRequired[float]
    max_drawdown_duration: NotRequired[int]
    var_95: NotRequired[float]
    var_99: NotRequired[float]
    cvar_95: NotRequired[float]
    large_jumps_count: NotRequired[int]
    jump_intensity: NotRequired[float]


def calculate_volatility_risk(
    price_bars: List[PriceBar], lookback_periods: List[int] = [20, 60, 252]
) -> VolatilityRisk:
    """Volatility risk calculation with multiple timeframes and metrics
    price_bars: List of price bars ascending order. the index 0 is the oldest bar,
    and the index -1 is the latest bar.
    lookback_periods: List of lookback periods in days. Default is [20, 60, 252].
    """

    closes = [bar_["close_price"] for bar_ in price_bars]
    highs = [bar_["high_price"] for bar_ in price_bars]
    lows = [bar_["low_price"] for bar_ in price_bars]

    if len(closes) < 2:
        return {"error": "Insufficient data for volatility calculation"}

    # Calculate returns
    returns = np.array(
        [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    )

    results: VolatilityRisk = {}

    # 1. Multi-timeframe Historical Volatility
    for period in lookback_periods:
        if len(returns) >= period:
            period_returns = returns[-period:]
            hist_vol = np.std(period_returns) * np.sqrt(252)
            results[f"volatility_{period}d"] = hist_vol

    # 2. Garman-Klass Volatility (more efficient)
    if len(closes) > 1:
        log_hl = [np.log(high / low) for high, low in zip(highs[1:], lows[1:])]
        log_co = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        gk_vol = np.sqrt(
            np.mean(
                [
                    0.5 * (hl**2) - (2 * np.log(2) - 1) * (co**2)
                    for hl, co in zip(log_hl, log_co)
                ]
            )
        ) * np.sqrt(252)
        results["garman_klass_volatility"] = gk_vol

    # 3. Parkinson Volatility (uses high-low range)
    parkinson_vol = np.sqrt(
        (1 / (4 * np.log(2)))
        * np.mean([np.log(high / low) ** 2 for high, low in zip(highs, lows)])
    ) * np.sqrt(252)
    results["parkinson_volatility"] = parkinson_vol

    # 4. Realized Volatility (RMS of returns)
    realized_vol = np.sqrt(np.mean(returns**2)) * np.sqrt(252)
    results["realized_volatility"] = realized_vol

    # 5. Volatility Clustering (GARCH-like measure)
    squared_returns = returns**2
    volatility_clustering = (
        np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        if len(squared_returns) > 2
        else 0
    )
    results["volatility_clustering"] = volatility_clustering

    # 6. Maximum Drawdown with duration
    peak = closes[0]
    max_drawdown = 0
    # drawdown_start = 0
    max_drawdown_duration = 0
    current_drawdown_duration = 0

    for i, price in enumerate(closes):
        if price > peak:
            peak = price
            current_drawdown_duration = 0
        else:
            drawdown = (peak - price) / peak
            current_drawdown_duration += 1
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_duration = current_drawdown_duration

    results["max_drawdown"] = max_drawdown
    results["max_drawdown_duration"] = max_drawdown_duration

    # 7. Value at Risk (VaR) calculations
    var_95 = np.percentile(returns, 5).item()
    var_99 = np.percentile(returns, 1).item()
    results["var_95"] = var_95
    results["var_99"] = var_99

    # 8. Conditional VaR (Expected Shortfall)
    cvar_95 = (
        np.mean(returns[returns <= var_95]).item()
        if len(returns[returns <= var_95]) > 0
        else var_95
    )
    results["cvar_95"] = cvar_95

    # 10. Jump Detection
    returns_std = np.std(returns)
    large_jumps = len(returns[np.abs(returns) > returns_std * 2])
    results["large_jumps_count"] = large_jumps
    results["jump_intensity"] = large_jumps / len(returns) if len(returns) > 0 else 0

    return results


def calculate_price_risk(
    price_bars: List[PriceBar], lookback_periods: List[int] = [5, 10, 20, 50]
) -> dict:
    """Enhanced price action risk metrics with multiple lookbacks, momentum, and breakout signals
    price_bars: List of price bars ascending order. the index 0 is the oldest bar,
    and the index -1 is the latest bar.
    lookback_periods: List of lookback periods in days. Default is [5, 10, 20, 50].
    """
    if not price_bars:
        return {"error": "No price bars provided"}

    horizons = lookback_periods

    highs = [bar_["high_price"] for bar_ in price_bars]
    lows = [bar_["low_price"] for bar_ in price_bars]
    closes = [bar_["close_price"] for bar_ in price_bars]
    current_price = closes[-1]

    def support_resistance(lookback: int) -> tuple[float, float]:
        """Local support/resistance over lookback"""
        if len(highs) < lookback:
            return min(lows), max(highs)
        return min(lows[-lookback:]), max(highs[-lookback:])

    # Multi-horizon support/resistance
    sr_levels = {h: support_resistance(h) for h in horizons}

    # Distances to nearest levels
    distances = {}
    for h, (sup, res) in sr_levels.items():
        distances[f"support_{h}d"] = (current_price - sup) / current_price
        distances[f"resistance_{h}d"] = (res - current_price) / current_price

    # Momentum and trend strength (dynamically for each horizon)
    momentums = {}
    for h in horizons:
        if len(closes) >= h:
            momentums[f"momentum_{h}d"] = (current_price - closes[-h]) / closes[-h]
        else:
            momentums[f"momentum_{h}d"] = 0

    # Average True Range (proxy for intraday volatility)
    tr = [
        max(high, c_prev) - min(low, c_prev)
        for high, low, c_prev in zip(highs[1:], lows[1:], closes[:-1])
    ]

    atrs = {}
    for h in horizons:
        atrs[f"average_true_range_{h}d"] = (
            np.mean(tr[-h:]) if len(tr) >= h else np.mean(tr) if tr else 0
        )
        atrs[f"average_true_range_{h}d_percent"] = (
            atrs[f"average_true_range_{h}d"] / current_price
        )

    breakouts = {}
    breakdowns = {}
    for h in horizons:
        if h in sr_levels:
            breakouts[f"breakout_{h}d"] = 1 if current_price > sr_levels[h][1] else 0
            breakdowns[f"breakdown_{h}d"] = 1 if current_price < sr_levels[h][0] else 0
        else:
            breakouts[f"breakout_{h}d"] = 0
            breakdowns[f"breakdown_{h}d"] = 0

    # Return enriched dictionary
    return {
        "current_price": current_price,
        **{f"support_{h}d": sr[0] for h, sr in sr_levels.items()},
        **{f"resistance_{h}d": sr[1] for h, sr in sr_levels.items()},
        **distances,
        **momentums,
        **atrs,
        **breakouts,
        **breakdowns,
    }


def format_volatility_risk_markdown(risk, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Volatility Risk Indicators", ""]

    def describe_metric(key: str) -> str:
        if key.startswith("volatility_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Historical volatility over {horizon} days (annualized)"
        if key == "garman_klass_volatility":
            return "Range-based volatility using open/close and high/low (annualized)"
        if key == "parkinson_volatility":
            return "High–low range volatility estimator (annualized)"
        if key == "realized_volatility":
            return "RMS of daily returns (annualized)"
        if key == "volatility_clustering":
            return "Lag-1 correlation of squared returns (persistence)"
        if key == "max_drawdown":
            return "Largest peak-to-trough decline (fraction of peak)"
        if key == "max_drawdown_duration":
            return "Longest consecutive days spent in drawdown"
        if key == "var_95":
            return "5th percentile daily return (VaR 95%)"
        if key == "var_99":
            return "1st percentile daily return (VaR 99%)"
        if key == "cvar_95":
            return "Average return in worst 5% tail (CVaR 95%)"
        if key == "large_jumps_count":
            return "Days with |return| > 2× std"
        if key == "jump_intensity":
            return "Fraction of days with large jumps"
        return "Indicator"

    md.append("| Metric | Value | Definition |")
    md.append("|--------|-------|------------|")
    for key, value in risk.items():
        md.append(
            f"| {key} | {utils.format_float(value, 3)} | {describe_metric(key)} |"
        )
    md.append("")

    return "\n".join(md)


def format_price_risk_markdown(risk, ticker_symbol):
    """
    Simple markdown format with tables for each category
    """
    md = [f"# {ticker_symbol} Price Risk Indicators", ""]

    def describe_metric(key: str) -> str:
        if key == "current_price":
            return "Latest closing price"
        if key.startswith("momentum_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Return over {horizon} days relative to prior close"
        if (
            key.startswith("average_true_range_")
            and key.endswith("d")
            and not key.endswith("d_percent")
        ):
            horizon = key.split("_")[3].removesuffix("d")
            return f"Average True Range over {horizon} days (absolute)"
        if key.startswith("average_true_range_") and key.endswith("d_percent"):
            horizon = key.split("_")[3].removesuffix("d")
            return f"ATR over {horizon} days as a fraction of price"
        if key.startswith("support_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"Distance to {horizon}d support (lowest low) as fraction of price"
        if key.startswith("resistance_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return (
                f"Distance to {horizon}d resistance (highest high) as fraction of price"
            )
        if key.startswith("breakout_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"1 if price above {horizon}d resistance; else 0"
        if key.startswith("breakdown_") and key.endswith("d"):
            horizon = key.split("_")[1].removesuffix("d")
            return f"1 if price below {horizon}d support; else 0"
        return "Indicator"

    md.append("| Metric | Value | Definition |")
    md.append("|--------|-------|------------|")
    for key, value in risk.items():
        md.append(
            f"| {key} | {utils.format_float(value, 3)} | {describe_metric(key)} |"
        )
    md.append("")

    return "\n".join(md)


__all__ = [
    "calculate_volatility_risk",
    "format_volatility_risk_markdown",
    "calculate_price_risk",
    "format_price_risk_markdown",
]
```

</source_code>