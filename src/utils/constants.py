from typing import TypedDict


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
