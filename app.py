# Streamlit MVP: daily/weekly charts + basic pattern detection
# Run: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

st.set_page_config(page_title="Chart Patterns MVP", layout="wide")

# -------------------- Utilities --------------------

def load_data(ticker: str, tf: str = "1D", period: str = "10y") -> pd.DataFrame:
    if tf == "1D":
        interval = "1d"
    elif tf == "1W":
        interval = "1wk"
    else:
        raise ValueError("Unsupported timeframe")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = df.rename(columns=str.title)
    df = df.dropna()
    return df

def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    # simple smoothing for peak detection
    return series.rolling(window=window, min_periods=1, center=True).mean()

def rolling_max_breakout(df: pd.DataFrame, lookback: int) -> Optional[pd.Timestamp]:
    # Returns the date of a breakout above rolling max (prior N bars)
    if len(df) < lookback + 2:
        return None
    roll = df['Close'].rolling(lookback).max().shift(1)  # prior max
    cond = df['Close'] > roll
    if cond.any():
        idx = cond[cond].index[-1]
        return idx
    return None

def compute_sma(df: pd.DataFrame, length: int) -> pd.Series:
    return df['Close'].rolling(length).mean()

def circ(ax, x, y, rad=18):
    from matplotlib.patches import Circle
    c = Circle((x, y), rad, fill=False, linestyle='--', linewidth=2, edgecolor='red', alpha=0.9)
    ax.add_patch(c)

# -------------------- Pattern Detection --------------------

@dataclass
class HSDetection:
    left_shoulder: int
    head: int
    right_shoulder: int
    trough_left: int
    trough_right: int
    neckline_pts: Tuple[int, int]
    confirmed: bool

def detect_head_shoulders(df: pd.DataFrame, inverted: bool = False) -> Optional[HSDetection]:
    """
    Heuristic H&S detection using peaks/troughs on smoothed closes.
    This is intentionally simple for MVP; expect false positives/negatives.
    """
    close = df['Close'].copy()
    if inverted:
        close = -close

    s = smooth(close, 7)
    # find local maxima (peaks) for H&S, minima for inverse (because we inverted, it's still peaks)
    peaks, _ = find_peaks(s.values, distance=max(5, len(df)//100))
    troughs, _ = find_peaks((-s).values, distance=max(5, len(df)//100))
    if len(peaks) < 3 or len(troughs) < 2:
        return None

    # Consider last ~400 bars to keep it recent
    start_idx = max(0, len(df) - 450)
    peaks = [p for p in peaks if p >= start_idx]
    troughs = [t for t in troughs if t >= start_idx]
    if len(peaks) < 3 or len(troughs) < 2:
        return None

    # Find sequence i<j<k peaks s.t. j is head (highest), i and k similar height
    px = s.values
    best = None
    tol_shoulder = 0.12  # shoulders within 12%
    min_head_gap = 0.03  # head at least 3% higher than shoulders
    for a in range(len(peaks)-2):
        i, j, k = peaks[a], peaks[a+1], peaks[a+2]
        p_i, p_j, p_k = px[i], px[j], px[k]
        # head highest
        if not (p_j > p_i*(1+min_head_gap) and p_j > p_k*(1+min_head_gap)):
            continue
        # shoulders similar
        avg_sh = (p_i + p_k) / 2.0
        if avg_sh == 0: 
            continue
        if abs(p_i - p_k)/avg_sh > tol_shoulder:
            continue
        # locate troughs between i-j and j-k
        left_tr = [t for t in troughs if i < t < j]
        right_tr = [t for t in troughs if j < t < k]
        if not left_tr or not right_tr:
            continue
        tl = min(left_tr, key=lambda t: px[t])
        tr = min(right_tr, key=lambda t: px[t])
        # neckline coords (tl, tr). Confirmation if price breaks below (or above for inverted)
        # Estimate neckline level at last bar by linear interpolation
        x1, x2 = tl, tr
        y1, y2 = px[tl], px[tr]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        xN = len(df)-1
        yN = y1 + slope*(xN - x1)

        last = px[-1]
        confirmed = last < yN if not inverted else last > yN

        cand = (j, HSDetection(i, j, k, tl, tr, (tl, tr), confirmed))
        if best is None or cand[0] > best[0]:
            best = cand

    if best:
        return best[1]
    return None

@dataclass
class CupHandle:
    left_rim: int
    bottom: int
    right_rim: int
    handle_low: Optional[int]
    breakout: Optional[int]

def detect_cup_handle(df: pd.DataFrame) -> Optional[CupHandle]:
    """
    Very heuristic cup&handle:
      - Find U-shape: two rims similar height, deep bottom between.
      - Handle: small pullback after right rim (< 1/3 cup depth).
      - Breakout: close above right rim high.
    """
    close = df['Close'].copy()
    s = smooth(close, 7).values
    n = len(df)
    # Work on last ~600 bars
    start_idx = max(0, n-600)
    x = np.arange(n)

    # Find candidate rims via local maxima
    peaks, _ = find_peaks(s, distance=10, prominence=np.nanmax(s)*0.005)
    peaks = [p for p in peaks if p >= start_idx]
    if len(peaks) < 2:
        return None

    # Try pairs of peaks (left<right)
    tol_rim = 0.12
    for a in range(len(peaks)-1):
        L = peaks[a]
        for b in range(a+1, min(a+30, len(peaks))):
            R = peaks[b]
            pL, pR = s[L], s[R]
            if abs(pL - pR)/((pL+pR)/2) > tol_rim:
                continue
            # bottom between
            mid = np.argmin(s[L:R+1]) + L
            depth = (max(pL, pR) - s[mid]) / max(1e-9, max(pL, pR))
            if depth < 0.12:  # at least 12% deep
                continue
            # handle: small pullback after R
            handle_low = None
            if R+5 < n-1:
                post = s[R+1: min(R+60, n)]
                if len(post) > 0:
                    h_idx_rel = np.argmin(post)
                    h_val = post[h_idx_rel]
                    h_idx = R+1+h_idx_rel
                    handle_depth = (pR - h_val)/max(1e-9, pR)
                    if 0.02 <= handle_depth <= min(0.35, depth*0.7):
                        handle_low = h_idx
            # breakout
            breakout = None
            rim_level = max(pL, pR)
            # last close above rim?
            last_break = np.where(close.values > rim_level)[0]
            last_break = last_break[last_break > R]
            if last_break.size > 0:
                breakout = int(last_break[-1])
            return CupHandle(L, mid, R, handle_low, breakout)
    return None

# -------------------- Plotting --------------------

def plot_chart(df: pd.DataFrame, ticker: str, tf: str, hs: Optional[HSDetection], ihs: Optional[HSDetection],
               breakout_idx: Optional[pd.Timestamp], cup: Optional[CupHandle]) -> plt.Figure:
    df = df.copy()
    df["SMA200"] = compute_sma(df, 200)
    style = mpf.make_mpf_style(marketcolors=mpf.make_marketcolors(up='k', down='k', edge='k', wick='k'),
                               gridstyle='', facecolor='white', figcolor='white')
    addp = []
    if df["SMA200"].notna().any():
        addp.append(mpf.make_addplot(df["SMA200"], color="#23D5D5", width=1.5))
    fig, axlist = mpf.plot(df.tail(800), type='candle', addplot=addp, style=style,
                           returnfig=True, volume=False, figsize=(12,5),
                           title=f"{ticker} — {'Diario' if tf=='1D' else 'Semanal'}")
    ax = axlist[0]

    # map index to integer x for annotation
    xmap = {d:i for i,d in enumerate(df.tail(800).index)}
    # helper to draw circle at index
    def draw_at(i, color='red'):
        idx = df.index[i]
        if idx in xmap:
            x = xmap[idx]; y = df['Close'].iloc[i]
            circ(ax, x, y, rad=max(8, len(xmap)*0.02))

    # H&S
    if hs:
        for i in [hs.left_shoulder, hs.head, hs.right_shoulder, hs.trough_left, hs.trough_right]:
            if 0 <= i < len(df):
                draw_at(i)
        # neckline
        x1, x2 = hs.neckline_pts
        if 0 <= x1 < len(df) and 0 <= x2 < len(df):
            ax.plot([xmap[df.index[x1]], xmap[df.index[x2]]],
                    [df['Close'].iloc[x1], df['Close'].iloc[x2]],
                    linestyle='--', linewidth=2, color='red', alpha=0.6)
    # Inverse H&S
    if ihs:
        for i in [ihs.left_shoulder, ihs.head, ihs.right_shoulder, ihs.trough_left, ihs.trough_right]:
            if 0 <= i < len(df):
                draw_at(i)
        x1, x2 = ihs.neckline_pts
        if 0 <= x1 < len(df) and 0 <= x2 < len(df):
            ax.plot([xmap[df.index[x1]], xmap[df.index[x2]]],
                    [df['Close'].iloc[x1], df['Close'].iloc[x2]],
                    linestyle='--', linewidth=2, color='green', alpha=0.6)

    # Breakout 52w (or N)
    if breakout_idx is not None and breakout_idx in xmap:
        circ(ax, xmap[breakout_idx], df.loc[breakout_idx,'Close'], rad=max(8, len(xmap)*0.02))

    # Cup & Handle
    if cup:
        for i in [cup.left_rim, cup.bottom, cup.right_rim, cup.handle_low or -1, cup.breakout or -1]:
            if i is None or i < 0: 
                continue
            if 0 <= i < len(df): draw_at(i)
        # draw cup rims line
        if 0 <= cup.left_rim < len(df) and 0 <= cup.right_rim < len(df):
            ax.plot([xmap[df.index[cup.left_rim]], xmap[df.index[cup.right_rim]]],
                    [df['Close'].iloc[cup.left_rim], df['Close'].iloc[cup.right_rim]],
                    linestyle=':', linewidth=1.5, color='gray', alpha=0.8)

    return fig

# -------------------- UI --------------------

st.title("🧪 MVP — Patrones: Diario & Semanal")
st.caption("Introduce un ticker (ej. AAPL, MSFT, BTC-USD). Este MVP detecta H&S, H&S invertido, breakout de máximos (N barras) y Cup & Handle (heurístico).")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip()
with col2:
    lookback_breakout = st.number_input("Breakout: lookback barras", min_value=20, max_value=400, value=252, step=10)
with col3:
    period = st.selectbox("Histórico", options=["5y","10y","max"], index=1)

if ticker:
    for tf in ["1D","1W"]:
        df = load_data(ticker, tf=tf, period=period)
        if df.empty:
            st.warning(f"No hay datos para {ticker} en {tf}.")
            continue

        # detecciones
        hs = detect_head_shoulders(df, inverted=False)
        ihs = detect_head_shoulders(df, inverted=True)
        breakout_date = rolling_max_breakout(df, lookback_breakout)
        cup = detect_cup_handle(df)

        # panel de resultados
        flags = []
        if hs: flags.append("Head & Shoulders"+(" ✅ Confirmado" if hs.confirmed else " (setup)"))
        if ihs: flags.append("Inv. Head & Shoulders"+(" ✅ Confirmado" if ihs.confirmed else " (setup)"))
        if breakout_date: flags.append(f"Breakout > max {lookback_breakout} barras")
        if cup: flags.append("Cup & Handle (heurístico)")
        st.subheader(f"{ticker} — {'Diario' if tf=='1D' else 'Semanal'}")
        st.write("Patrones:", ", ".join(flags) if flags else "—")

        fig = plot_chart(df, ticker, tf, hs, ihs, breakout_date, cup)
        st.pyplot(fig, clear_figure=True)
else:
    st.info("Escribe un ticker y pulsa Enter.")