chat: https://chat.deepseek.com/a/chat/s/0aa783f0-51fc-4e79-bd43-6221268b26a5

# Agents:

## Agent Architecture:
- MiddleWare:
  - SummarizationMiddleware: Done
  - TodoListMiddleware: Done
- Portfolio Manager: WIP


## Context:
- Construct the context to be LLM-friendly content: WIP
- Make tool result LLM-friendly in markdown format: WIP
  - Us Market News: Done
  - Major ETF Price Statistics: Done
  - List Positions: Done
  

## Technical Implementation:
- in-database cache the results of the tools to avoid redundant calls: Done
- Async Cache Decorator: Done
- Refactor in-database cache to be decorator: Done
- Sync Agent Message to Sandx AI Monitoring Dashboard: Done in PoC
- Pylint: Done
- Yahoo Finance Fundamental Data API: Done

## Market Research Agent:
### Tools:
- Get the latest news from : https://tradingeconomics.com/united-states/news: Done
- Google Market Research: Done
- Optimize Google Market Research Prompt: Done
- Get the major ETF price statistics Done
- Get the latest stock price statistics: Done
- Get the most active stocks: Done
- Get Jina DeepSearch Tool: Not Impressive and Slow - Avoid using it
- BUY
- SELL


## Common Tools:
- Get positions with latest price and change: Done

## Deep Agent
- Learn About Langchain Deep Agent: Done

## Resources:
- https://github.com/The-Swarm-Corporation/AutoHedge: Less Useful