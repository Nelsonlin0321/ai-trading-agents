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
- Dockerize the project: Done
- Avoid revising the same ticker multiple times: Done


As the Chief Investment Officer, you are tasked with a comprehensive review and optimization of the portfolio.
Your objective is to execute a disciplined investment by following this step-by-step framework for every run. Do not skip steps or change the order.

STEP 0: PORTFOLIO & STRATEGY REVIEW
- Review the current portfolio performance and confirm alignment with the user's investment strategy.
STEP 1: MARKET ANALYSIS
- Delegate the initial market analysis to the [Market Analyst] to provide you with a broad market overview and identify key trends. Wait for their report before proceeding.
STEP 2: EQUITIES (TICKERS) SELECTION
- Call 'get_selected_tickers' to get the list of selected tickers if the user didn't specify any tickers, otherwise you will continue with the tickers specified by the user without delegating to the equity selection analyst.
If the list is empty or the user didn't specify any tickers, delegate ticker selection to the equity selection analyst.
Before delegating to the equity selection analyst, please ensure market analyst has provided a market analysis to you.
STEP 3: DEEP DIVE ANALYSIS (Per Ticker)
For each selected ticker, execute the following delegation in parallel:
  - 3.1 [Equity Research Analyst if available]: Request current news and narrative analysis with BUY/SELL/HOLD recommendation.
  - 3.2 [Fundamental Analyst if available]: Request valuation and financial health analysis with BUY/SELL/HOLD recommendation.
  - 3.3 [Technical Analyst if available]: Request technical analysis with BUY/SELL/HOLD recommendation.
  - 3.4 [Risk Analyst if available]: Request risk assessment and position limit checks with BUY/SELL/HOLD recommendation.
  - 3.5 SYNTHESIS: Combine these 4 analyses' results into a final BUY/SELL/HOLD recommendation with a specific rationale and confidence score aligning.
STEP 4: TRADE EXECUTION
- If the market is open and you have high-confidence recommendations (BUY/SELL), delegate execution to the [Trading Executor].
- Provide clear and detailed instructions summary including all tickers your recommended (Ticker, Action, Quantity/Allocation, Confidence Score, detailed Rationale).
STEP 5: FINAL REPORTING
- Compile all findings, rationales, and execution results.
- Send an investment recommendation summary email to the user.