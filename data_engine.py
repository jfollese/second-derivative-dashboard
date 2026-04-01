"""
Second Derivative Data Engine v2
Fetches market data and computes rate-of-change acceleration metrics.
Now includes: yield curve, prediction markets, copper/gold, credit spreads, and alerts.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import json
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOOKBACK_DAYS = 200          # ~8 months of history for rolling calcs
ROC_WINDOW = 10              # First derivative: 10-day rate of change
ACCEL_WINDOW = 5             # Second derivative: 5-day ROC of the ROC
ZSCORE_WINDOW = 60           # Rolling z-score normalization window

# Alert thresholds
ALERT_ZSCORE = 2.0           # z-score above this triggers alert
PREDICTION_ALERT_PCT = 10    # Prediction market move > this % in a day triggers alert

# --- Yahoo Finance tickers organized by theme ---

YIELD_CURVE = ['^TNX', '^TYX', '^FVX', '^IRX', 'TLT', 'IEF', 'SHY']
# ^TNX = 10Y yield, ^TYX = 30Y, ^FVX = 5Y, ^IRX = 13-week T-bill
# TLT = 20+ yr bond ETF, IEF = 7-10 yr, SHY = 1-3 yr

PRIVATE_CREDIT = ['OWL', 'ARCC', 'FSK', 'OBDC', 'KKR', 'HYG', 'JNK', 'BKLN']

OIL_INFLATION = ['CL=F', 'GC=F', 'UGA', 'XLE', 'QQQ', 'SI=F', 'HG=F', 'DBA']
# SI=F = silver, HG=F = copper, DBA = agriculture ETF

AI_INFRA = ['CRWV', 'SMCI', 'NVDA', 'AVGO', 'TSM', 'MSFT', 'GOOGL', 'META', 'AMZN']

FX_DOLLAR = ['JPY=X', 'DX-Y.NYB', 'CL=F', 'EURUSD=X', 'CNY=X']

MARKET_STRUCTURE = ['^VIX', '^VIX3M', 'BRK-B', 'SPY', 'RSP', 'IWM']
# RSP = equal weight S&P, IWM = Russell 2000

CREDIT_SPREADS = ['HYG', 'LQD', 'EMB', 'AGG', 'BKLN']
# EMB = emerging market bonds, AGG = total bond, BKLN = senior loans

ALL_TICKERS = list(set(
    YIELD_CURVE + PRIVATE_CREDIT + OIL_INFLATION + AI_INFRA +
    FX_DOLLAR + MARKET_STRUCTURE + CREDIT_SPREADS + ['OWL', 'LQD']
))

# FRED series
FRED_SERIES = {
    'T5YIE': 'T5YIE',      # 5-Year Breakeven Inflation Rate (daily)
    'T10YIE': 'T10YIE',    # 10-Year Breakeven Inflation Rate (daily)
    'BAMLH0A0HYM2': 'BAMLH0A0HYM2',  # ICE BofA US High Yield OAS (daily)
    'DTWEXBGS': 'DTWEXBGS',  # Trade Weighted Dollar Index (daily)
}

# Polymarket prediction markets to track — organized by theme
# These are picked for fast movement and macro relevance
POLYMARKET_MARKETS = [
    # --- Fed & Rates ---
    {'slug': 'will-no-fed-rate-cuts-happen-in-2026', 'name': 'No Fed Cuts in 2026', 'category': 'fed'},
    {'slug': 'will-1-fed-rate-cut-happen-in-2026', 'name': '1 Fed Cut in 2026', 'category': 'fed'},
    {'slug': 'will-2-fed-rate-cuts-happen-in-2026', 'name': '2 Fed Cuts in 2026', 'category': 'fed'},
    {'slug': 'will-3-fed-rate-cuts-happen-in-2026', 'name': '3 Fed Cuts in 2026', 'category': 'fed'},
    # --- Recession & Economy ---
    {'slug': 'us-recession-by-end-of-2026', 'name': 'US Recession by End 2026', 'category': 'economy'},
    # --- Geopolitical ---
    {'slug': 'will-china-invade-taiwan-before-2027', 'name': 'China Invades Taiwan', 'category': 'geopolitical'},
    {'slug': 'us-x-russia-military-clash-by-june-30-2026-249', 'name': 'US-Russia Clash by Jun', 'category': 'geopolitical'},
    {'slug': 'russia-x-ukraine-ceasefire-before-2027', 'name': 'Ukraine Ceasefire by 2027', 'category': 'geopolitical'},
    {'slug': 'putin-out-before-2027', 'name': 'Putin Out by 2027', 'category': 'geopolitical'},
    # --- Markets & Tech ---
    {'slug': 'nvda-above-170-on-april-30-2026', 'name': 'NVDA > $170 End April', 'category': 'markets'},
    {'slug': 'will-meta-acquire-tiktok-745-612-641', 'name': 'Meta Acquires TikTok', 'category': 'markets'},
    {'slug': 'will-tesla-release-optimus-by-june-30-2026', 'name': 'Tesla Optimus by Jun', 'category': 'markets'},
    # --- Trump / Policy ---
    {'slug': 'will-trump-be-impeached-by-december-31-2026', 'name': 'Trump Impeached by 2027', 'category': 'politics'},
    {'slug': 'will-trump-resign-by-december-31-2026', 'name': 'Trump Resigns by 2027', 'category': 'politics'},
]

# How often to auto-search for new trending markets
POLYMARKET_AUTO_DISCOVER = True
POLYMARKET_DISCOVER_KEYWORDS = [
    'recession', 'fed', 'rate cut', 'tariff', 'iran', 'crude oil', 'inflation',
    'china', 'taiwan', 'russia', 'war', 'military', 'ceasefire', 'sanctions',
    'bitcoin', 'crash', 'bear market', 'treasury', 'debt ceiling', 'default',
    'nvda', 'tesla', 'openai', 'stock market', 'nato', 'nuclear',
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_yahoo_v8(ticker: str, period: str = '9mo') -> pd.Series:
    """Fetch a single ticker via Yahoo Finance v8 chart API (cloud-friendly)."""
    import requests
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {'range': period, 'interval': '1d', 'includePrePost': 'false'}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        'Accept': 'application/json',
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code != 200:
            return pd.Series(dtype=float, name=ticker)
        data = r.json()
        result = data.get('chart', {}).get('result', [])
        if not result:
            return pd.Series(dtype=float, name=ticker)
        timestamps = result[0].get('timestamp', [])
        closes = result[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])
        if not timestamps or not closes:
            return pd.Series(dtype=float, name=ticker)
        dates = pd.to_datetime(timestamps, unit='s').normalize()
        s = pd.Series(closes, index=dates, name=ticker, dtype=float)
        return s.dropna()
    except Exception as e:
        print(f"Yahoo v8 error for {ticker}: {e}")
        return pd.Series(dtype=float, name=ticker)


def fetch_yahoo_data() -> pd.DataFrame:
    """Download closing prices for all tickers. Tries yfinance first, falls back to v8 API."""
    end = datetime.now()
    start = end - timedelta(days=LOOKBACK_DAYS + 60)

    # Try yfinance bulk download first (fast when it works)
    try:
        tickers_str = ' '.join(ALL_TICKERS)
        raw = yf.download(tickers_str, start=start, end=end, auto_adjust=True, progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw['Close']
            else:
                prices = raw[['Close']].copy()
                prices.columns = ALL_TICKERS[:1]
            if len(prices.columns) > 5:  # Got meaningful data
                return prices
    except Exception as e:
        print(f"yfinance bulk failed ({e}), falling back to v8 API...")

    # Fallback: fetch each ticker individually via v8 API
    print("Using Yahoo v8 chart API fallback...")
    frames = {}
    for ticker in ALL_TICKERS:
        s = _fetch_yahoo_v8(ticker)
        if not s.empty:
            frames[ticker] = s
    if frames:
        return pd.DataFrame(frames)
    return pd.DataFrame()


def _http_get(url: str, timeout: int = 15) -> str:
    """HTTP GET with curl_cffi if available, else plain requests."""
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(url, impersonate="chrome", timeout=timeout)
        if r.status_code == 200:
            return r.text
    except ImportError:
        pass
    except Exception:
        pass
    import requests
    r = requests.get(url, timeout=timeout,
                     headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
    if r.status_code == 200:
        return r.text
    return None


def fetch_fred_series(series_id: str) -> pd.Series:
    """Fetch a FRED series via the public CSV endpoint (no API key)."""
    import io
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        text = _http_get(url)
    except Exception as e:
        print(f"FRED fetch error for {series_id}: {e}")
        return pd.Series(dtype=float, name=series_id)
    if text:
        try:
            df = pd.read_csv(io.StringIO(text))
            date_col = df.columns[0]
            val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.set_index(date_col)
            s = pd.to_numeric(df[val_col], errors='coerce').dropna()
            s.name = series_id
            return s
        except Exception as e:
            print(f"FRED parse error for {series_id}: {e}")
    return pd.Series(dtype=float, name=series_id)


def fetch_fred_data() -> pd.DataFrame:
    """Fetch all FRED series and combine into a DataFrame."""
    cutoff = datetime.now() - timedelta(days=LOOKBACK_DAYS + 60)
    frames = {}
    for sid in FRED_SERIES:
        s = fetch_fred_series(sid)
        if not s.empty:
            s = s[s.index >= pd.Timestamp(cutoff)]
            if not s.empty:
                frames[sid] = s
    if frames:
        return pd.DataFrame(frames)
    return pd.DataFrame()


def _fetch_single_polymarket(slug: str, name: str, category: str) -> dict:
    """Fetch a single Polymarket market by slug."""
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
    """Fetch current prices from configured + auto-discovered Polymarket markets."""
    results = {}

    # 1. Fetch configured markets
    for market in POLYMARKET_MARKETS:
        info = _fetch_single_polymarket(market['slug'], market['name'],
                                         market['category'])
        if info:
            results[market['name']] = info

    # 2. Auto-discover trending macro markets
    if POLYMARKET_AUTO_DISCOVER:
        try:
            all_markets = []
            for offset in [0, 100, 200]:
                text = _http_get(
                    f'https://gamma-api.polymarket.com/markets?limit=100&offset={offset}&active=true&closed=false',
                    timeout=12
                )
                if text:
                    all_markets.extend(json.loads(text))

            for m in all_markets:
                q = m.get('question', '').lower()
                slug = m.get('slug', '')
                if not any(k in q for k in POLYMARKET_DISCOVER_KEYWORDS):
                    continue
                # Skip if already in results or if it's a resolved/near-resolved market
                if any(slug == v.get('slug') for v in results.values()):
                    continue
                prices = m.get('outcomePrices', '[]')
                if isinstance(prices, str):
                    try:
                        prices = json.loads(prices)
                    except:
                        continue
                if not prices:
                    continue
                yes_pct = float(prices[0]) * 100
                if 3 < yes_pct < 97:  # Skip near-resolved markets
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
                        'yes_pct': yes_pct,
                        'category': cat,
                        'slug': slug,
                        'question': m.get('question', '')[:80],
                    }
        except Exception as e:
            print(f"Polymarket auto-discover error: {e}")

    return results


# ---------------------------------------------------------------------------
# Derivative calculations
# ---------------------------------------------------------------------------

def rate_of_change(series: pd.Series, window: int = ROC_WINDOW) -> pd.Series:
    """First derivative: percentage rate of change over N days."""
    return series.pct_change(periods=window) * 100


def second_derivative(series: pd.Series, roc_window: int = ROC_WINDOW,
                      accel_window: int = ACCEL_WINDOW) -> pd.Series:
    """Second derivative: rate of change of the rate of change."""
    roc = rate_of_change(series, roc_window)
    return roc.diff(periods=accel_window)


def rolling_zscore(series: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    """Normalize a series to rolling z-scores for cross-asset comparison."""
    mean = series.rolling(window=window, min_periods=20).mean()
    std = series.rolling(window=window, min_periods=20).std()
    return (series - mean) / std.replace(0, np.nan)


# ---------------------------------------------------------------------------
# Alert system
# ---------------------------------------------------------------------------

def check_alerts(data_dict: dict, threshold: float = ALERT_ZSCORE) -> list:
    """Check if any series has a z-score above the alert threshold."""
    alerts = []
    for name, series in data_dict.items():
        s = series.dropna()
        if s.empty:
            continue
        val = s.iloc[-1]
        if abs(val) > threshold:
            direction = "BEARISH" if val > 0 else "BULLISH"
            alerts.append({
                'name': name,
                'value': round(val, 2),
                'direction': direction,
                'severity': 'high' if abs(val) > 2.5 else 'medium',
            })
    return alerts


# ---------------------------------------------------------------------------
# Panel data builders
# ---------------------------------------------------------------------------

class DashboardData:
    """Fetches all data and computes second-derivative metrics for each panel."""

    def __init__(self):
        self.prices = pd.DataFrame()
        self.fred = pd.DataFrame()
        self.polymarket = {}
        self.polymarket_history = {}  # Track changes over time
        self.alerts = []
        self.last_updated = None

    def refresh(self):
        """Pull fresh data from all sources."""
        self.prices = fetch_yahoo_data()
        self.fred = fetch_fred_data()

        # Track polymarket changes
        old_prices = {k: v.get('yes_pct', 0) for k, v in self.polymarket.items()}
        self.polymarket = fetch_polymarket_prices()

        # Compute prediction market alerts
        for name, data in self.polymarket.items():
            old = old_prices.get(name)
            if old is not None and data['yes_pct'] is not None:
                change = data['yes_pct'] - old
                if abs(change) > PREDICTION_ALERT_PCT:
                    self.alerts.append({
                        'name': f"PREDICTION: {name}",
                        'value': round(change, 1),
                        'direction': f"{'UP' if change > 0 else 'DOWN'} {abs(change):.1f}pp",
                        'severity': 'high',
                    })

        # Check all convergence signals for alerts
        conv_data = self.convergence_data()
        self.alerts = check_alerts(conv_data, ALERT_ZSCORE)

        self.last_updated = datetime.now()

    def _safe_col(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Safely get a column, return empty series if missing."""
        if col in df.columns:
            return df[col].dropna()
        return pd.Series(dtype=float, name=col)

    # --- Panel 1: Convergence Monitor ---
    def convergence_data(self) -> dict:
        """The 'one chart' — 5 z-scored second-derivative series."""
        result = {}

        # 1. Breakeven inflation ROC
        bie = self._safe_col(self.fred, 'T5YIE')
        if not bie.empty:
            result['Inflation Expectations'] = rolling_zscore(second_derivative(bie))

        # 2. HYG/LQD ratio ROC (credit stress)
        hyg = self._safe_col(self.prices, 'HYG')
        lqd = self._safe_col(self.prices, 'LQD')
        if not hyg.empty and not lqd.empty:
            ratio = hyg / lqd
            result['Credit Stress (HYG/LQD)'] = rolling_zscore(second_derivative(ratio)) * -1

        # 3. OWL momentum (private credit stress proxy)
        owl = self._safe_col(self.prices, 'OWL')
        if not owl.empty:
            result['Private Credit (OWL)'] = rolling_zscore(second_derivative(owl)) * -1

        # 4. VIX/VIX3M ratio ROC (term structure inversion speed)
        vix = self._safe_col(self.prices, '^VIX')
        vix3m = self._safe_col(self.prices, '^VIX3M')
        if not vix.empty and not vix3m.empty:
            vterm = vix / vix3m
            result['VIX Term Structure'] = rolling_zscore(second_derivative(vterm))

        # 5. 2/10 spread acceleration (via bond ETF proxy: IEF/SHY ratio)
        ief = self._safe_col(self.prices, 'IEF')  # 7-10 yr
        shy = self._safe_col(self.prices, 'SHY')  # 1-3 yr
        if not ief.empty and not shy.empty:
            spread_proxy = ief / shy  # Rising = curve steepening
            result['Yield Curve (2/10)'] = rolling_zscore(second_derivative(spread_proxy))

        return result

    def convergence_score(self) -> tuple:
        """How many convergence metrics are simultaneously bearish (z > 1)."""
        data = self.convergence_data()
        if not data:
            return 0, 0
        bearish = 0
        total = len(data)
        for name, series in data.items():
            latest = series.dropna()
            if not latest.empty and latest.iloc[-1] > 1.0:
                bearish += 1
        return bearish, total

    # --- Panel 2: Yield Curve & Rates ---
    def yield_curve_data(self) -> dict:
        """2/10 spread, real rates, and curve shape acceleration."""
        result = {}

        # 10Y yield acceleration
        tnx = self._safe_col(self.prices, '^TNX')
        if not tnx.empty:
            result['10Y Yield'] = second_derivative(tnx)

        # 2Y proxy via SHY (inverse relationship: SHY down = 2Y yield up)
        shy = self._safe_col(self.prices, 'SHY')
        if not shy.empty:
            result['2Y Proxy (inv SHY)'] = second_derivative(shy) * -1

        # 2/10 spread proxy: TLT/SHY ratio (long duration vs short duration)
        tlt = self._safe_col(self.prices, 'TLT')
        if not tlt.empty and not shy.empty:
            spread = tlt / shy
            result['2/10 Spread (TLT/SHY)'] = second_derivative(spread)

        # 5Y yield
        fvx = self._safe_col(self.prices, '^FVX')
        if not fvx.empty:
            result['5Y Yield'] = second_derivative(fvx)

        # Real yield proxy: 10Y minus breakeven
        tnx_s = self._safe_col(self.prices, '^TNX')
        bie = self._safe_col(self.fred, 'T5YIE')
        if not tnx_s.empty and not bie.empty:
            # Align the indices
            combined = pd.concat([tnx_s, bie], axis=1).dropna()
            if len(combined) > 20:
                real_yield = combined.iloc[:, 0] - combined.iloc[:, 1]
                result['Real Yield (10Y-BEI)'] = second_derivative(real_yield)

        # HY OAS from FRED (credit spread)
        oas = self._safe_col(self.fred, 'BAMLH0A0HYM2')
        if not oas.empty:
            result['HY Credit Spread (OAS)'] = second_derivative(oas)

        return result

    # --- Panel 3: Private Credit Stress Velocity ---
    def private_credit_data(self) -> dict:
        result = {}
        names = {'OWL': 'Blue Owl', 'ARCC': 'Ares Capital', 'FSK': 'FS KKR',
                 'OBDC': 'Owl Rock', 'KKR': 'KKR', 'BKLN': 'Sr Loans (BKLN)'}
        for ticker, name in names.items():
            s = self._safe_col(self.prices, ticker)
            if not s.empty:
                result[name] = second_derivative(s) * -1  # Invert: down = stress

        # HYG and JNK for junk bond stress
        for t, n in [('HYG', 'HY Bonds (HYG)'), ('JNK', 'Junk Bonds (JNK)')]:
            s = self._safe_col(self.prices, t)
            if not s.empty:
                result[n] = second_derivative(s) * -1

        return result

    # --- Panel 4: Oil / Inflation Transmission ---
    def oil_inflation_data(self) -> dict:
        result = {}

        oil = self._safe_col(self.prices, 'CL=F')
        if not oil.empty:
            result['WTI Crude'] = second_derivative(oil)

        uga = self._safe_col(self.prices, 'UGA')
        if not uga.empty:
            result['Gasoline (UGA)'] = second_derivative(uga)

        gold = self._safe_col(self.prices, 'GC=F')
        if not gold.empty:
            result['Gold'] = second_derivative(gold)

        # Copper/Gold ratio — leading indicator of growth vs safety
        copper = self._safe_col(self.prices, 'HG=F')
        if not copper.empty and not gold.empty:
            ratio = copper / gold
            result['Copper/Gold Ratio'] = second_derivative(ratio)

        # Energy vs Tech relative
        xle = self._safe_col(self.prices, 'XLE')
        qqq = self._safe_col(self.prices, 'QQQ')
        if not xle.empty and not qqq.empty:
            ratio = xle / qqq
            result['Energy vs Tech (XLE/QQQ)'] = second_derivative(ratio)

        # Agriculture
        dba = self._safe_col(self.prices, 'DBA')
        if not dba.empty:
            result['Agriculture (DBA)'] = second_derivative(dba)

        # Silver
        silver = self._safe_col(self.prices, 'SI=F')
        if not silver.empty:
            result['Silver'] = second_derivative(silver)

        return result

    # --- Panel 5: AI Infrastructure ---
    def ai_infra_data(self) -> dict:
        result = {}
        ai_names = {'CRWV': 'CoreWeave', 'SMCI': 'Super Micro', 'NVDA': 'Nvidia',
                     'AVGO': 'Broadcom', 'TSM': 'TSMC'}
        for ticker, name in ai_names.items():
            s = self._safe_col(self.prices, ticker)
            if not s.empty:
                result[name] = second_derivative(s)

        # Hyperscaler average
        hypers = ['MSFT', 'GOOGL', 'META', 'AMZN']
        hyper_data = []
        for t in hypers:
            s = self._safe_col(self.prices, t)
            if not s.empty:
                hyper_data.append(rate_of_change(s))
        if hyper_data:
            avg = pd.concat(hyper_data, axis=1).mean(axis=1)
            result['Hyperscaler Avg'] = avg.diff(periods=ACCEL_WINDOW)

        # NVDA vs supply chain relative
        nvda = self._safe_col(self.prices, 'NVDA')
        supply = []
        for t in ['AVGO', 'TSM', 'SMCI']:
            s = self._safe_col(self.prices, t)
            if not s.empty:
                supply.append(s)
        if not nvda.empty and supply:
            supply_avg = pd.concat(supply, axis=1).mean(axis=1)
            ratio = nvda / supply_avg
            result['NVDA vs Supply Chain'] = second_derivative(ratio)

        return result

    # --- Panel 6: FX & Dollar ---
    def fx_dollar_data(self) -> dict:
        result = {}

        jpy = self._safe_col(self.prices, 'JPY=X')
        if not jpy.empty:
            result['USD/JPY'] = second_derivative(jpy)

        dxy = self._safe_col(self.prices, 'DX-Y.NYB')
        if not dxy.empty:
            result['Dollar Index (DXY)'] = second_derivative(dxy)

        eur = self._safe_col(self.prices, 'EURUSD=X')
        if not eur.empty:
            result['EUR/USD'] = second_derivative(eur)

        cny = self._safe_col(self.prices, 'CNY=X')
        if not cny.empty:
            result['USD/CNY'] = second_derivative(cny)

        # FRED trade-weighted dollar
        tw = self._safe_col(self.fred, 'DTWEXBGS')
        if not tw.empty:
            result['Trade-Weighted $'] = second_derivative(tw)

        return result

    # --- Panel 7: Market Structure ---
    def market_structure_data(self) -> dict:
        result = {}

        vix = self._safe_col(self.prices, '^VIX')
        vix3m = self._safe_col(self.prices, '^VIX3M')
        if not vix.empty and not vix3m.empty:
            spread = vix - vix3m
            result['VIX Term Spread'] = spread
            result['VIX Accel'] = second_derivative(vix)

        # BRK-B vs SPY (defensive rotation)
        brk = self._safe_col(self.prices, 'BRK-B')
        spy = self._safe_col(self.prices, 'SPY')
        if not brk.empty and not spy.empty:
            ratio = brk / spy
            result['BRK-B vs SPY (defense)'] = second_derivative(ratio)

        # Equal weight vs cap weight (breadth)
        rsp = self._safe_col(self.prices, 'RSP')
        if not rsp.empty and not spy.empty:
            ratio = rsp / spy
            result['Equal vs Cap Wt (breadth)'] = second_derivative(ratio)

        # Small caps vs large (risk appetite)
        iwm = self._safe_col(self.prices, 'IWM')
        if not iwm.empty and not spy.empty:
            ratio = iwm / spy
            result['Small vs Large (IWM/SPY)'] = second_derivative(ratio)

        return result

    # --- Panel 8: Prediction Markets ---
    def prediction_market_data(self) -> dict:
        """Return current Polymarket probabilities with metadata."""
        return self.polymarket

    # --- All alerts ---
    def get_all_alerts(self) -> list:
        """Collect alerts from all panels."""
        alerts = list(self.alerts)  # Start with convergence alerts

        # Check each panel for extreme moves
        panels = {
            'Yield Curve': self.yield_curve_data(),
            'Private Credit': self.private_credit_data(),
            'Oil/Inflation': self.oil_inflation_data(),
            'AI Infra': self.ai_infra_data(),
            'FX': self.fx_dollar_data(),
            'Market Structure': self.market_structure_data(),
        }
        for panel_name, panel_data in panels.items():
            for name, series in panel_data.items():
                s = series.dropna()
                if s.empty or len(s) < 2:
                    continue
                # Check if latest move is extreme (>3 std dev from recent mean)
                recent = s.tail(60)
                if len(recent) < 20:
                    continue
                mean = recent.mean()
                std = recent.std()
                if std == 0:
                    continue
                z = (s.iloc[-1] - mean) / std
                if abs(z) > 2.5:
                    alerts.append({
                        'name': f"{panel_name}: {name}",
                        'value': round(s.iloc[-1], 2),
                        'direction': 'EXTREME HIGH' if z > 0 else 'EXTREME LOW',
                        'severity': 'high' if abs(z) > 3 else 'medium',
                    })

        return alerts
