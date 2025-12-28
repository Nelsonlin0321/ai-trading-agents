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
    "TECHNICAL_ANALYST",
]

ALL_ROLES: List[AgentRole] = [
    "CHIEF_INVESTMENT_OFFICER",
    "MARKET_ANALYST",
    "RISK_ANALYST",
    "EQUITY_RESEARCH_ANALYST",
    "FUNDAMENTAL_ANALYST",
    "TRADING_EXECUTOR",
    "TECHNICAL_ANALYST",
]

LEARNING_RATE = 10
