# Streamlit MVP: daily/weekly charts + basic pattern detection (fixed for SciPy 1-D arrays)
# Run: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Chart Patterns MVP", layout="wide")

# -------------------- Utilities --------------------

def load_data(ticker: str, tf: str = "1D", period: str = "10y") -> pd.DataFrame:
    try:
        if tf == "1D":
            interval = "1d"
        elif tf == "1W":
            interval = "1wk"
        else:
            raise ValueError("Unsupported timeframe")
        
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if df.empty:
            return df
        
        # Fix column names - handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df.rename(columns={col: col.title() for col in df.columns})
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        return pd.DataFrame()

def smooth(series: pd.Series, window: int = 5) -> pd.Series:
    # simple smoothing for peak detection
    return series.rolling(window=window, min_periods=1, center=True).mean()

def series_1d(x) -> np.ndarray:
    """Return a clean 1-D float64 numpy array (no NaNs) from Series/DataFrame/ndarray."""
    # Accept pandas objects or numpy arrays
    if isinstance(x, pd.DataFrame):
        # take first column
        arr = x.iloc[:, 0].to_numpy()
    elif isinstance(x, pd.Series):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    # Squeeze any singleton dimensions, e.g., (N,1) -> (N,)
    arr = np.asarray(arr, dtype="float64").squeeze()
    if arr.ndim != 1:
        # Flatten as last resort
        arr = arr.reshape(-1)

    # Replace NaNs/Infs via forward/back fill using pandas (simplest)
    s = pd.Series(arr, copy=True)
    if s.empty:
        return np.array([], dtype="float64")
    s = s.replace([np.inf, -np.inf], np.nan)
    # Use modern pandas syntax
    s = s.ffill().bfill().fillna(0.0)
    return s.to_numpy()

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
    # Add validation for coordinates
    if not (np.isfinite(x) and np.isfinite(y)):
        return
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
    neckline_pts: tuple
    confirmed: bool

def detect_head_shoulders(df: pd.DataFrame, inverted: bool = False) -> Optional[HSDetection]:
    """
    Heuristic H&S detection using peaks/troughs on smoothed closes.
    This is intentionally simple for MVP; expect false positives/negatives.
    """
    try:
        if 'Close' not in df.columns or len(df) < 20:
            return None
        close = df['Close'].copy()
        if inverted:
            close = -close

        s = smooth(close, 7)
        x = series_1d(s)
        if x.size < 20:
            return None

        dist = max(5, len(df)//100) or 5
        peaks, _ = find_peaks(x, distance=dist)
        troughs, _ = find_peaks(-x, distance=dist)
        if len(peaks) < 3 or len(troughs) < 2:
            return None

        # Consider last ~400 bars to keep it recent
        start_idx = max(0, len(df) - 450)
        peaks = [p for p in peaks if p >= start_idx]
        troughs = [t for t in troughs if t >= start_idx]
        if len(peaks) < 3 or len(troughs) < 2:
            return None

        # Find sequence i<j<k peaks s.t. j is head (highest), i and k similar height
        px = x
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
            # troughs between i-j and j-k
            left_tr = [t for t in troughs if i < t < j]
            right_tr = [t for t in troughs if j < t < k]
            if not left_tr or not right_tr:
                continue
            tl = min(left_tr, key=lambda t: px[t])
            tr = min(right_tr, key=lambda t: px[t])
            # neckline
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
    except Exception as e:
        st.warning(f"Error in H&S detection: {str(e)}")
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
    try:
        if 'Close' not in df.columns or len(df) < 50:
            return None
        close = df['Close'].copy()
        s = series_1d(smooth(close, 7))
        n = len(df)
        if n < 50 or not np.isfinite(s).any():
            return None
        # Work on last ~600 bars
        start_idx = max(0, n-600)

        # Find candidate rims via local maxima
        try:
            prom = float(np.nanmax(s) * 0.005) if np.isfinite(np.nanmax(s)) else 0.0
        except ValueError:
            prom = 0.0
        peaks, _ = find_peaks(s, distance=10, prominence=prom)
        peaks = [p for p in peaks if p >= start_idx]
        if len(peaks) < 2:
            return None

        tol_rim = 0.12
        for a in range(len(peaks)-1):
            L = peaks[a]
            for b in range(a+1, min(a+30, len(peaks))):
                R = peaks[b]
                pL, pR = s[L], s[R]
                if (pL + pR) == 0: 
                    continue
                if abs(pL - pR)/((pL+pR)/2) > tol_rim:
                    continue
                # bottom between
                mid = int(np.argmin(s[L:R+1]) + L)
                depth = (max(pL, pR) - s[mid]) / max(1e-9, max(pL, pR))
                if depth < 0.12:
                    continue
                # handle: small pullback after R
                handle_low = None
                if R+5 < n-1:
                    post = s[R+1: min(R+60, n)]
                    if post.size > 0:
                        h_idx_rel = int(np.argmin(post))
                        h_val = float(post[h_idx_rel])
                        h_idx = R+1+h_idx_rel
                        handle_depth = (pR - h_val)/max(1e-9, pR)
                        if 0.02 <= handle_depth <= min(0.35, depth*0.7):
                            handle_low = h_idx
                # breakout
                breakout = None
                rim_level = max(pL, pR)
                idxs = np.where(df['Close'].values > rim_level)[0]
                idxs = idxs[idxs > R]
                if idxs.size > 0:
                    breakout = int(idxs[-1])
                return CupHandle(L, mid, R, handle_low, breakout)
    except Exception as e:
        st.warning(f"Error in Cup & Handle detection: {str(e)}")
    return None

# -------------------- Plotting --------------------

def plot_chart(df: pd.DataFrame, ticker: str, tf: str, hs: Optional[HSDetection], ihs: Optional[HSDetection],
               breakout_idx: Optional[pd.Timestamp], cup: Optional[CupHandle]) -> plt.Figure:
    try:
        df = df.copy()
        df["SMA200"] = compute_sma(df, 200)
        
        # Create style for mplfinance
        style = mpf.make_mpf_style(
            marketcolors=mpf.make_marketcolors(up='k', down='k', edge='k', wick='k'),
            gridstyle='', 
            facecolor='white', 
            figcolor='white'
        )
        
        addp = []
        if df["SMA200"].notna().any():
            addp.append(mpf.make_addplot(df["SMA200"], color="#23D5D5", width=1.5))
        
        # Use a more compatible approach for plotting
        plot_data = df.tail(800)
        
        fig, axlist = mpf.plot(
            plot_data, 
            type='candle', 
            addplot=addp if addp else None,
            style=style,
            returnfig=True, 
            volume=False, 
            figsize=(12,5),
            title=f"{ticker} — {'Diario' if tf=='1D' else 'Semanal'}"
        )
        ax = axlist[0]

        # map index to integer x for annotation
        xmap = {d:i for i,d in enumerate(plot_data.index)}
        
        def draw_at(i):
            if i < 0 or i >= len(df): 
                return
            idx = df.index[i]
            if idx in xmap:
                x = xmap[idx]
                y = df['Close'].iloc[i]
                if np.isfinite(x) and np.isfinite(y):
                    circ(ax, x, y, rad=max(8, len(xmap)*0.02))

        # H&S
        if hs:
            for i in [hs.left_shoulder, hs.head, hs.right_shoulder, hs.trough_left, hs.trough_right]:
                draw_at(i)
            x1, x2 = hs.neckline_pts
            if 0 <= x1 < len(df) and 0 <= x2 < len(df):
                x1_pos = xmap.get(df.index[x1], None)
                x2_pos = xmap.get(df.index[x2], None)
                if x1_pos is not None and x2_pos is not None:
                    ax.plot([x1_pos, x2_pos],
                            [df['Close'].iloc[x1], df['Close'].iloc[x2]],
                            linestyle='--', linewidth=2, color='red', alpha=0.6)

        # Inverse H&S
        if ihs:
            for i in [ihs.left_shoulder, ihs.head, ihs.right_shoulder, ihs.trough_left, ihs.trough_right]:
                draw_at(i)
            x1, x2 = ihs.neckline_pts
            if 0 <= x1 < len(df) and 0 <= x2 < len(df):
                x1_pos = xmap.get(df.index[x1], None)
                x2_pos = xmap.get(df.index[x2], None)
                if x1_pos is not None and x2_pos is not None:
                    ax.plot([x1_pos, x2_pos],
                            [df['Close'].iloc[x1], df['Close'].iloc[x2]],
                            linestyle='--', linewidth=2, color='green', alpha=0.6)

        # Breakout
        if breakout_idx is not None and breakout_idx in xmap:
            circ(ax, xmap[breakout_idx], df.loc[breakout_idx,'Close'], rad=max(8, len(xmap)*0.02))

        # Cup & Handle
        if cup:
            for i in [cup.left_rim, cup.bottom, cup.right_rim, cup.handle_low or -1, cup.breakout or -1]:
                if i is None or i < 0: 
                    continue
                draw_at(i)
            if 0 <= cup.left_rim < len(df) and 0 <= cup.right_rim < len(df):
                x1_pos = xmap.get(df.index[cup.left_rim], None)
                x2_pos = xmap.get(df.index[cup.right_rim], None)
                if x1_pos is not None and x2_pos is not None:
                    ax.plot([x1_pos, x2_pos],
                            [df['Close'].iloc[cup.left_rim], df['Close'].iloc[cup.right_rim]],
                            linestyle=':', linewidth=1.5, color='gray', alpha=0.8)

        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        # Return a simple fallback plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df['Close'].tail(100))
        ax.set_title(f"{ticker} - Error in advanced plotting")
        return fig

# -------------------- UI --------------------

st.title("🧪 MVP — Patrones: Diario & Semanal")
st.caption("Introduce un ticker (ej. AAPL, MSFT, BTC-USD). MVP: H&S, H&S invertido, breakout de máximos (N barras), Cup & Handle (heurístico).")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
with col2:
    lookback_breakout = st.number_input("Breakout: lookback barras", min_value=20, max_value=400, value=252, step=10)
with col3:
    period = st.selectbox("Histórico", options=["5y","10y","max"], index=1)

if ticker:
    for tf in ["1D","1W"]:
        with st.spinner(f"Cargando datos para {ticker} ({tf})..."):
            df = load_data(ticker, tf=tf, period=period)
            
        if df.empty:
            st.warning(f"No hay datos para {ticker} en {tf}.")
            continue

        # detecciones
        try:
            # Validar que tenemos datos suficientes
            if len(df) < 50:
                st.warning(f"Datos insuficientes para {ticker} ({tf}): solo {len(df)} barras")
                continue
                
            hs = detect_head_shoulders(df, inverted=False)
            ihs = detect_head_shoulders(df, inverted=True)
            breakout_date = rolling_max_breakout(df, lookback_breakout)
            cup = detect_cup_handle(df)
        except Exception as e:
            st.error(f"Error en detección de patrones para {ticker} ({tf}): {str(e)}")
            hs = ihs = cup = None
            breakout_date = None

        # panel de resultados
        flags = []
        if hs: flags.append("Head & Shoulders"+(" ✅ Confirmado" if hs.confirmed else " (setup)"))
        if ihs: flags.append("Inv. Head & Shoulders"+(" ✅ Confirmado" if ihs.confirmed else " (setup)"))
        if breakout_date: flags.append(f"Breakout > max {lookback_breakout} barras")
        if cup: flags.append("Cup & Handle (heurístico)")
        
        st.subheader(f"{ticker} — {'Diario' if tf=='1D' else 'Semanal'}")
        st.write("**Patrones:**", ", ".join(flags) if flags else "—")

        with st.spinner("Generando gráfico..."):
            fig = plot_chart(df, ticker, tf, hs, ihs, breakout_date, cup)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)  # Explicitly close figure to free memory
else:
    st.info("Escribe un ticker y pulsa Enter.")
