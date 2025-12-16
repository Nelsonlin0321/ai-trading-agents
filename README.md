# SandX AI Trading Agents

A multi-agent trading research and execution system built on LangChain. It orchestrates specialized agents (Market Analyst, Equity Research Analyst, Fundamental Analyst, Risk Analyst, Trading Executor) under a Chief Investment Officer (CIO) to research markets, generate recommendations, and execute trades.

## Overview
- Roles: `MARKET_ANALYST`, `EQUITY_RESEARCH_ANALYST`, `FUNDAMENTAL_ANALYST`, `RISK_ANALYST`, `TRADING_EXECUTOR`, coordinated by `CHIEF_INVESTMENT_OFFICER`
- Data: Alpaca market data, Yahoo Finance fundamentals, curated news, Google market research
- Persistence: PostgreSQL via Prisma; caching on Upstash Redis
- Models: OpenAI-compatible chat models via `OPENAI_API_URL` (e.g., DeepSeek)

## Architecture
- `src/agents/*/agent.py`: Agent builders with tools and middleware
- `src/tools/*`: LangChain tools for news, fundamentals, risk, trading
- `src/tools_adaptors/*`: Async actions wrapping services and formatting markdown
- `src/context.py`: Builds run context from DB and restores previous messages
- `src/middleware/__init__.py`: Logging, summarization, todo list middleware
- `src/services/alpaca/*`: Alpaca data APIs (quotes, bars, news)
- `src/services/sandx_ai/*`: SandX AI portfolio/positions and timeline
- `src/models.py`: Model factory using `ChatOpenAI`
- `schema.prisma`: DB schema for `Run`, `Bot`, `Portfolio`, `Trade`, `AgentMessage`, `Recommend`, etc.

## Requirements
- Python 3.12
- PostgreSQL reachable via `DATABASE_URL`
- Redis (Upstash REST) for caching
- Alpaca API credentials
- OpenAI-compatible API endpoint and key

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
uv sync
```
Generate Prisma client:
```bash
source .venv/bin/activate
python -m prisma generate
```
Enable git hooks (optional):
```bash
uv add pre-commit
pre-commit install
```
Type checking and linting:
```bash
source .venv/bin/activate
pyright
ruff --fix
```

## Environment Variables
Set via `.env` or AWS Secrets Manager (`sandx.ai`). Required:
- `DATABASE_URL` (PostgreSQL) or `DEV_DATABASE_URL`/`PROD_DATABASE_URL` with `ENV`
- `ENV` (e.g., `dev` or `prod`)
- `OPENAI_API_KEY`, `OPENAI_API_URL`
- `ALPACA_API_KEY`, `ALPACA_API_SECRET`
- `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`
- `API_KEY` (SandX AI API), `SANDX_AI_URL` (default `http://localhost:3000/api`)
- `BASE_URL` (used in emails and links)

## Running
The main entry consumes a `runId` created in the DB.
```bash
python src/main.py <runId>
```
This:
- Restores unfinished CIO messages for the run
- Builds context from `Bot`, `Portfolio`, `Watchlist`, and `Trades`
- Streams CIO agent events and persists summaries
- Updates `Run.status` to `SUCCESS`/`FAILURE`

To build and run with Docker:
```bash
docker build -t ai-trading-agents:latest .
docker run --rm \
  -e ENV=dev \
  -e OPENAI_API_KEY=... -e OPENAI_API_URL=... \
  -e ALPACA_API_KEY=... -e ALPACA_API_SECRET=... \
  -e UPSTASH_REDIS_REST_URL=... -e UPSTASH_REDIS_REST_TOKEN=... \
  -e API_KEY=... -e SANDX_AI_URL=... -e BASE_URL=... \
  ai-trading-agents:latest <runId>
```

## Testing
- Alpaca client quick test:
```bash
python -m test.src.services.alpaca.test_api_client
```

## Key Tools
- Market research: `get_latest_market_news`, `do_google_market_research`
- Fundamentals: `get_fundamental_data`
- Risk: `get_price_risk_indicators`, `get_volatility_risk_indicators`
- Trading: `buy_stock`, `sell_stock`, `get_market_status`
- Portfolio: `list_current_positions`, `get_portfolio_performance_analysis`
- Recommendations: `get_recommend_stock_tool`, `get_analysts_recommendations`

## Development Notes
- Notebooks are cleared on commit via pre-commit
- Use `pylint` to generate config: `pylint --generate-rcfile > .pylintrc`
- CIO default instruction lives in `src/main.py` and can be overridden by `Run.instruction`

## Troubleshooting
- Missing env: the app raises descriptive errors (`get_env`) when variables are absent
- Prisma errors: ensure `DATABASE_URL` is set and `python -m prisma generate` ran
- API auth errors: verify `API_KEY` for SandX AI and Alpaca keys
- Docker entrypoint expects a `<runId>` argument; pass it as the container command