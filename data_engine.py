"""
Second Derivative Data Engine v3
Uses FRED as primary data source (works from any cloud server).
Yahoo Finance as optional fallback for individual stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io
import time
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 200
ROC_WINDOW = 10
ACCEL_WINDOW = 5
ZSCORE_WINDOW = 60

# ---------------------------------------------------------------------------
# FRED series — primary data source (works from cloud servers)
# ---------------------------------------------------------------------------

FRED_MARKET_DATA = {
    # Equities
    'SP500': 'S&P 500',
    'NASDAQCOM': 'Nasdaq',
    'DJIA': 'Dow Jones',
    # Volatility
    'VIXCLS': 'VIX',
    # Yields
    'DGS2': '2Y Yield',
    'DGS5': '5Y Yield',
    'DGS10': '10Y Yield',
    'DTB3': '3M T-Bill',
    # Inflation
    'T5YIE': '5Y Breakeven',
    'T10YIE': '10Y Breakeven',
    # Credit
    'BAMLH0A0HYM2': 'HY OAS',
    'BAMLC0A4CBBB': 'BBB OAS',
    # Commodities
    'DCOILWTICO': 'WTI Oil',
    'DCOILBRENTEU': 'Brent Oil',
    # FX
    'DEXJPUS': 'USD/JPY',
    'DEXUSEU': 'EUR/USD',
    'DEXCHUS': 'USD/CNY',
    'DTWEXBGS': 'Trade-Wt Dollar',
}

# Yahoo Finance tickers — for individual stocks only (works locally, may fail on cloud)
YAHOO_STOCKS = [
    'OWL', 'ARCC', 'FSK', 'OBDC', 'KKR',          # Private credit
    'HYG', 'LQD', 'JNK', 'BKLN', 'EMB',            # Bond ETFs
    'TLT', 'IEF', 'SHY',                             # Treasury ETFs
    'CRWV', 'SMCI', 'NVDA', 'AVGO', 'TSM',          # AI infra
    'MSFT', 'GOOGL', 'META', 'AMZN',                 # Hyperscalers
    'XLE', 'QQQ', 'SPY', 'RSP', 'IWM', 'BRK-B',    # ETFs/indices
    'GC=F', 'SI=F', 'HG=F', 'CL=F', 'DBA', 'UGA',  # Commodities
]

# Polymarket
POLYMARKET_MARKETS = [
    {'slug': 'will-no-fed-rate-cuts-happen-in-2026', 'name': 'No Fed Cuts 2026', 'category': 'fed'},
    {'slug': 'will-1-fed-rate-cut-happen-in-2026', 'name': '1 Fed Cut 2026', 'category': 'fed'},
    {'slug': 'will-2-fed-rate-cuts-happen-in-2026', 'name': '2 Fed Cuts 2026', 'category': 'fed'},
    {'slug': 'will-3-fed-rate-cuts-happen-in-2026', 'name': '3 Fed Cuts 2026', 'category': 'fed'},
    {'slug': 'us-recession-by-end-of-2026', 'name': 'US Recession 2026', 'category': 'economy'},
    {'slug': 'will-china-invade-taiwan-before-2027', 'name': 'China Invades Taiwan', 'category': 'geopolitical'},
    {'slug': 'us-x-russia-military-clash-by-june-30-2026-249', 'name': 'US-Russia Clash', 'category': 'geopolitical'},
    {'slug': 'russia-x-ukraine-ceasefire-before-2027', 'name': 'Ukraine Ceasefire', 'category': 'geopolitical'},
    {'slug': 'putin-out-before-2027', 'name': 'Putin Out by 2027', 'category': 'geopolitical'},
    {'slug': 'nvda-above-170-on-april-30-2026', 'name': 'NVDA > $170 April', 'category': 'markets'},
    {'slug': 'will-meta-acquire-tiktok-745-612-641', 'name': 'Meta Buys TikTok', 'category': 'markets'},
    {'slug': 'will-tesla-release-optimus-by-june-30-2026', 'name': 'Tesla Optimus Jun', 'category': 'markets'},
    {'slug': 'will-trump-be-impeached-by-december-31-2026', 'name': 'Trump Impeached', 'category': 'politics'},
    {'slug': 'will-trump-resign-by-december-31-2026', 'name': 'Trump Resigns', 'category': 'politics'},
]

POLYMARKET_AUTO_DISCOVER = True
POLYMARKET_DISCOVER_KEYWORDS = [
    'recession', 'fed', 'rate cut', 'tariff', 'iran', 'crude oil', 'inflation',
    'china', 'taiwan', 'russia', 'war', 'military', 'ceasefire', 'sanctions',
    'bitcoin', 'crash', 'bear market', 'treasury', 'debt ceiling', 'default',
    'nvda', 'tesla', 'openai', 'stock market', 'nato', 'nuclear',
]


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 30, retries: int = 2) -> str:
    """HTTP GET with retries for slow cloud servers."""
    import requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/json,text/csv,*/*',
    }
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
            else:
                print(f"HTTP GET failed after {retries+1} attempts: {url[:60]}... - {e}")
    return None


# ---------------------------------------------------------------------------
# FRED fetching (primary — works everywhere)
# ---------------------------------------------------------------------------

def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a FRED series via public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        text = _http_get(url)
        if text:
            df = pd.read_csv(io.StringIO(text))
            date_col = df.columns[0]
            val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.set_index(date_col)
            s = pd.to_numeric(df[val_col], errors='coerce').dropna()
            s.name = series_id
            return s
    except Exception as e:
        print(f"FRED error {series_id}: {e}")
    return pd.Series(dtype=float, name=series_id)


def fetch_all_fred() -> pd.DataFrame:
    """Fetch all FRED series, trimmed to lookback window."""
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS + 60)
    frames = {}
    for sid in FRED_MARKET_DATA:
        s = fetch_fred_series(sid)
        if not s.empty:
            s = s[s.index >= pd.Timestamp(cutoff)]
            if not s.empty:
                frames[sid] = s
    print(f"FRED: fetched {len(frames)}/{len(FRED_MARKET_DATA)} series")
    return pd.DataFrame(frames) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Yahoo fetching (optional — for individual stocks)
# ---------------------------------------------------------------------------

def fetch_yahoo_stocks() -> pd.DataFrame:
    """Try to fetch individual stock prices via yfinance. May fail on cloud."""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS + 60)
    try:
        tickers_str = ' '.join(YAHOO_STOCKS)
        raw = yf.download(tickers_str, start=start, end=end, auto_adjust=True, progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw['Close']
            else:
                prices = raw[['Close']].copy()
                prices.columns = YAHOO_STOCKS[:1]
            ok_count = prices.dropna(axis=1, how='all').shape[1]
            if ok_count > 3:
                print(f"Yahoo: fetched {ok_count}/{len(YAHOO_STOCKS)} stocks")
                return prices
    except Exception as e:
        print(f"Yahoo stocks unavailable: {e}")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------

def _fetch_single_polymarket(slug: str, name: str, category: str) -> dict:
    try:
        text = _http_get(f'https://gamma-api.polymarket.com/markets?slug={slug}', timeout=8)
        if text:
            data = json.loads(text)
            if data:
                m = data[0]
                prices = m.get('outcomePrices', '[]')
                if isinstance(prices, str):
                    prices = json.loads(prices)
                yes_price = float(prices[0]) * 100 if prices else None
                if yes_price is not None:
                    return {
                        'yes_pct': yes_price,
                        'category': category,
                        'slug': slug,
                        'question': m.get('question', name),
                    }
    except Exception:
        pass
    return None


def fetch_polymarket_prices() -> dict:
    results = {}
    for market in POLYMARKET_MARKETS:
        info = _fetch_single_polymarket(market['slug'], market['name'], market['category'])
        if info:
            results[market['name']] = info

    if POLYMARKET_AUTO_DISCOVER:
        try:
            all_markets = []
            for offset in [0, 100, 200]:
                text = _http_get(
                    f'https://gamma-api.polymarket.com/markets?limit=100&offset={offset}&active=true&closed=false',
                    timeout=12)
                if text:
                    all_markets.extend(json.loads(text))
            for m in all_markets:
                q = m.get('question', '').lower()
                slug = m.get('slug', '')
                if not any(k in q for k in POLYMARKET_DISCOVER_KEYWORDS):
                    continue
                if any(slug == v.get('slug') for v in results.values()):
                    continue
                prices = m.get('outcomePrices', '[]')
                if isinstance(prices, str):
                    try: prices = json.loads(prices)
                    except: continue
                if not prices:
                    continue
                yes_pct = float(prices[0]) * 100
                if 3 < yes_pct < 97:
                    cat = 'discovered'
                    for kw, c in [('fed', 'fed'), ('rate', 'fed'), ('recession', 'economy'),
                                  ('china', 'geopolitical'), ('russia', 'geopolitical'),
                                  ('iran', 'geopolitical'), ('war', 'geopolitical'),
                                  ('military', 'geopolitical'), ('ceasefire', 'geopolitical'),
                                  ('bitcoin', 'crypto'), ('btc', 'crypto'),
                                  ('oil', 'commodities'), ('crude', 'commodities'),
                                  ('tariff', 'policy'), ('trump', 'politics'),
                                  ('nvda', 'markets'), ('tesla', 'markets'),
                                  ('stock', 'markets'), ('crash', 'markets')]:
                        if kw in q:
                            cat = c
                            break
                    name = m.get('question', '')[:50]
                    results[name] = {
                        'yes_pct': yes_pct, 'category': cat,
                        'slug': slug, 'question': m.get('question', '')[:80],
                    }
        except Exception as e:
            print(f"Polymarket discover error: {e}")
    return results


# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def rate_of_change(series: pd.Series, window: int = ROC_WINDOW) -> pd.Series:
    return series.pct_change(periods=window) * 100

def second_derivative(series: pd.Series, roc_window: int = ROC_WINDOW,
                      accel_window: int = ACCEL_WINDOW) -> pd.Series:
    roc = rate_of_change(series, roc_window)
    return roc.diff(periods=accel_window)

def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = series.rolling(window=window, min_periods=20).mean()
    std = series.rolling(window=window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Dashboard Data
# ---------------------------------------------------------------------------

class DashboardData:
    def __init__(self):
        self.fred = pd.DataFrame()
        self.stocks = pd.DataFrame()
        self.polymarket = {}
        self.alerts = []
        self.last_updated = None

    def _s(self, col: str) -> pd.Series:
        """Get a series from FRED or Yahoo, whichever has it."""
        if col in self.fred.columns:
            return self.fred[col].dropna()
        if col in self.stocks.columns:
            return self.stocks[col].dropna()
        return pd.Series(dtype=float, name=col)

    def refresh(self):
        self.fred = fetch_all_fred()
        self.stocks = fetch_yahoo_stocks()
        self.polymarket = fetch_polymarket_prices()
        self.alerts = []
        conv = self.convergence_data()
        for name, series in conv.items():
            s = series.dropna()
            if not s.empty and abs(s.iloc[-1]) > 2.0:
                self.alerts.append({
                    'name': name, 'value': round(s.iloc[-1], 2),
                    'direction': 'BEARISH' if s.iloc[-1] > 0 else 'BULLISH',
                    'severity': 'high' if abs(s.iloc[-1]) > 2.5 else 'medium',
                })
        self.last_updated = datetime.now()

    # --- Panel 1: Convergence Monitor ---
    def convergence_data(self) -> dict:
        result = {}

        # Breakeven inflation acceleration
        bie = self._s('T5YIE')
        if not bie.empty:
            result['Inflation Expectations'] = rolling_zscore(second_derivative(bie))

        # Credit stress: HY OAS acceleration (rising = stress)
        oas = self._s('BAMLH0A0HYM2')
        if not oas.empty:
            result['Credit Stress (HY OAS)'] = rolling_zscore(second_derivative(oas))

        # Private credit proxy (OWL if available, else BBB OAS)
        owl = self._s('OWL')
        if not owl.empty:
            result['Private Credit (OWL)'] = rolling_zscore(second_derivative(owl)) * -1
        else:
            bbb = self._s('BAMLC0A4CBBB')
            if not bbb.empty:
                result['Credit Quality (BBB)'] = rolling_zscore(second_derivative(bbb))

        # VIX acceleration
        vix = self._s('VIXCLS')
        if not vix.empty:
            result['VIX Acceleration'] = rolling_zscore(second_derivative(vix))

        # Yield curve: 2/10 spread acceleration
        gs2 = self._s('DGS2')
        gs10 = self._s('DGS10')
        if not gs2.empty and not gs10.empty:
            spread = gs10 - gs2
            result['Yield Curve (2/10)'] = rolling_zscore(second_derivative(spread))

        return result

    def convergence_score(self) -> tuple:
        data = self.convergence_data()
        if not data:
            return 0, 0
        bearish = sum(1 for s in data.values() if not s.dropna().empty and s.dropna().iloc[-1] > 1.0)
        return bearish, len(data)

    # --- Panel 2: Yield Curve ---
    def yield_curve_data(self) -> dict:
        result = {}
        for sid, name in [('DGS10', '10Y Yield'), ('DGS2', '2Y Yield'), ('DGS5', '5Y Yield')]:
            s = self._s(sid)
            if not s.empty:
                result[name] = second_derivative(s)

        gs2, gs10 = self._s('DGS2'), self._s('DGS10')
        if not gs2.empty and not gs10.empty:
            result['2/10 Spread'] = second_derivative(gs10 - gs2)

        # Real yield: 10Y minus breakeven
        bie = self._s('T10YIE')
        if not gs10.empty and not bie.empty:
            aligned = pd.concat([gs10, bie], axis=1).dropna()
            if len(aligned) > 20:
                result['Real Yield'] = second_derivative(aligned.iloc[:, 0] - aligned.iloc[:, 1])

        oas = self._s('BAMLH0A0HYM2')
        if not oas.empty:
            result['HY Credit Spread'] = second_derivative(oas)

        return result

    # --- Panel 3: Private Credit Stress ---
    def private_credit_data(self) -> dict:
        result = {}
        for ticker in ['OWL', 'ARCC', 'FSK', 'OBDC', 'KKR']:
            s = self._s(ticker)
            if not s.empty:
                result[ticker] = second_derivative(s) * -1
        for t, n in [('HYG', 'HY Bonds'), ('JNK', 'Junk Bonds'), ('BKLN', 'Sr Loans')]:
            s = self._s(t)
            if not s.empty:
                result[n] = second_derivative(s) * -1
        # FRED fallbacks if Yahoo unavailable
        if len(result) < 2:
            oas = self._s('BAMLH0A0HYM2')
            if not oas.empty:
                result['HY OAS (FRED)'] = second_derivative(oas)
            bbb = self._s('BAMLC0A4CBBB')
            if not bbb.empty:
                result['BBB OAS (FRED)'] = second_derivative(bbb)
        return result

    # --- Panel 4: Oil / Inflation ---
    def oil_inflation_data(self) -> dict:
        result = {}
        for sid, name in [('DCOILWTICO', 'WTI Crude'), ('DCOILBRENTEU', 'Brent Crude')]:
            s = self._s(sid)
            if not s.empty:
                result[name] = second_derivative(s)
        for t in ['GC=F', 'SI=F', 'HG=F', 'DBA', 'UGA']:
            s = self._s(t)
            if not s.empty:
                names = {'GC=F': 'Gold', 'SI=F': 'Silver', 'HG=F': 'Copper', 'DBA': 'Agriculture', 'UGA': 'Gasoline'}
                result[names.get(t, t)] = second_derivative(s)

        # Copper/Gold ratio (growth vs safety signal)
        copper = self._s('HG=F')
        gold = self._s('GC=F')
        if not copper.empty and not gold.empty:
            result['Copper/Gold Ratio'] = second_derivative(copper / gold)

        # Energy vs Tech
        xle = self._s('XLE')
        qqq = self._s('QQQ')
        if not xle.empty and not qqq.empty:
            result['Energy vs Tech'] = second_derivative(xle / qqq)

        # Breakeven inflation for context
        bie = self._s('T5YIE')
        if not bie.empty:
            result['Breakeven Inflation'] = second_derivative(bie)

        return result

    # --- Panel 5: AI Infrastructure ---
    def ai_infra_data(self) -> dict:
        result = {}
        for t, n in [('CRWV', 'CoreWeave'), ('SMCI', 'Super Micro'), ('NVDA', 'Nvidia'),
                      ('AVGO', 'Broadcom'), ('TSM', 'TSMC')]:
            s = self._s(t)
            if not s.empty:
                result[n] = second_derivative(s)
        hypers = ['MSFT', 'GOOGL', 'META', 'AMZN']
        hyper_data = [rate_of_change(self._s(t)) for t in hypers if not self._s(t).empty]
        if hyper_data:
            avg = pd.concat(hyper_data, axis=1).mean(axis=1)
            result['Hyperscaler Avg'] = avg.diff(periods=ACCEL_WINDOW)
        # FRED Nasdaq as fallback
        if len(result) < 2:
            nas = self._s('NASDAQCOM')
            if not nas.empty:
                result['Nasdaq (FRED)'] = second_derivative(nas)
            sp = self._s('SP500')
            if not sp.empty:
                result['S&P 500 (FRED)'] = second_derivative(sp)
        return result

    # --- Panel 6: FX & Dollar ---
    def fx_dollar_data(self) -> dict:
        result = {}
        for sid, name in [('DEXJPUS', 'USD/JPY'), ('DEXUSEU', 'EUR/USD'),
                          ('DEXCHUS', 'USD/CNY'), ('DTWEXBGS', 'Trade-Wt Dollar')]:
            s = self._s(sid)
            if not s.empty:
                result[name] = second_derivative(s)
        # Oil for JPY overlay
        oil = self._s('DCOILWTICO')
        if not oil.empty:
            result['Oil (JPY overlay)'] = second_derivative(oil)
        return result

    # --- Panel 7: Market Structure ---
    def market_structure_data(self) -> dict:
        result = {}
        vix = self._s('VIXCLS')
        if not vix.empty:
            result['VIX Level'] = vix
            result['VIX Acceleration'] = second_derivative(vix)

        brk = self._s('BRK-B')
        spy = self._s('SPY')
        if not brk.empty and not spy.empty:
            result['BRK-B vs SPY'] = second_derivative(brk / spy)

        rsp = self._s('RSP')
        if not rsp.empty and not spy.empty:
            result['Equal vs Cap Wt'] = second_derivative(rsp / spy)

        iwm = self._s('IWM')
        if not iwm.empty and not spy.empty:
            result['Small vs Large'] = second_derivative(iwm / spy)

        # FRED fallbacks
        if len(result) < 3:
            sp = self._s('SP500')
            nas = self._s('NASDAQCOM')
            if not sp.empty and not nas.empty:
                result['Nasdaq vs S&P'] = second_derivative(nas / sp)
            dj = self._s('DJIA')
            if not dj.empty and not sp.empty:
                result['Dow vs S&P'] = second_derivative(dj / sp)

        return result

    # --- Prediction markets ---
    def prediction_market_data(self) -> dict:
        return self.polymarket

    # --- Alerts ---
    def get_all_alerts(self) -> list:
        return list(self.alerts)
