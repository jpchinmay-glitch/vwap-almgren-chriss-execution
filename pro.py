import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from scipy.stats import linregress
from scipy.interpolate import interp1d

try:
    import yfinance as yf
    HAS_YF=True
except Exception:
    yf=None
    HAS_YF=False

# SECTION 1 — DATA CLASSES      
@dataclass
class MarketParams:
    """Market microstructure parameters."""
    sigma:float   # Daily return volatility
    eta:float     # Temporary impact coefficient
    gamma:float    # Permanent impact coefficient
    adv:float      # Average daily volume (shares)
    spread_bps:float  # Full spread in basis points
    price:float = 100.0

@dataclass
class OrderParams:
    """Order and strategy parameters."""
    X:float              # Total shares to execute
    T:int            # Execution horizon in days
    lam:float    # Risk aversion / urgency
    alpha_annual:float   # Annualized alpha
    alpha_halflife:float # Signal half-life in days
    side: int=-1        # -1 sell, +1 buy

@dataclass
class CalibrationResult:
    """Output from empirical parameter calibration."""
    eta:float
    gamma:float
    sigma:float
    adv:float
    spread_bps:float
    price:float
    r2_temp:float
    r2_perm:float
    notes:List[str]=field(default_factory=list)

 ## MARKET DATA
def fetch_market_data(ticker:str="SPY",period:str="1y")->Optional[pd.DataFrame]:
    """Download daily OHLCV data via yfinance."""
    if not HAS_YF:
        print("yfinance not available. Using synthetic data.")
        return None

    try:
        df=yf.download(ticker,period=period,progress=False,auto_adjust=True)
        if df is None or df.empty or len(df)<40:
            return None

        df=df[["Open","High","Low","Close","Volume"]].copy()
        df.columns=["Open","High","Low","Close","Volume"]
        df.dropna(inplace=True)  
        df["Returns"]=df["Close"].pct_change()
        df["LogReturns"]=np.log(df["Close"]/df["Close"].shift(1))
        df.dropna(inplace=True)
        print(f"Fetched{len(df)} days of {ticker} data from yfinance.")
        return df
    except Exception as e:
        print(f"yfinance fetch failed ({e}).Using synthetic data.")
        return None

def synthetic_market_data(
    n_days:int=252,
    S0:float=100.0,
    mu:float=0.08,
    sigma:float=0.015,
    adv:int=1000000,
    seed:int=42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data using GBM-like returns."""
    rng=np.random.default_rng(seed)
    dt=1/252
    ret=rng.normal(mu*dt,sigma,n_days)
    close=S0*np.cumprod(1+ret)

    noise=rng.uniform(0.002,0.008,n_days)
    high=close*(1+noise)
    low=close*(1-noise)
    open_=np.roll(close,1)
    open_[0]=S0
    base_vol=adv*(1+0.6*np.abs(ret)/max(sigma,1e-8))
    volume=(base_vol*rng.lognormal(0,0.25,n_days)).astype(int)

    dates=pd.bdate_range(end=pd.Timestamp.today(),periods=n_days)
    df=pd.DataFrame(
        {
            "Open":open_,
            "High":high,
            "Low":low,
            "Close":close,
            "Volume":volume,
        },
        index=dates,
    )
    df["Returns"]=df["Close"].pct_change()
    df["LogReturns"]=np.log(df["Close"] / df["Close"].shift(1))
    df.dropna(inplace=True)
    print(f"Generated {len(df)} days of synthetic data (σ={sigma:.2%}, ADV={adv:,.0f}).")
    return df

# SECTION 3 — VOLUME CURVE ESTIMATION
def estimate_intraday_volume_curve(df:pd.DataFrame,n_buckets:int=13)->np.ndarray:
    """
    Estimate a proxy U-shaped intraday volume curve from daily bars.
    This is not a true empirical intraday curve because minute data is not available.
    """
    tmp=df.copy()
    tmp["HL"]=tmp["High"]-tmp["Low"]
    tmp["OC"]=(tmp["Close"]-tmp["Open"]).abs()
    tmp["skew"]=np.clip(tmp["OC"]/(tmp["HL"]+1e-12),0,1)
    mean_skew=float(tmp["skew"].mean())
    open_weight=0.45+0.30*mean_skew
    close_weight=0.28+0.18*mean_skew
    mid_weight=max(1.0-open_weight-close_weight,0.05)
    t=np.linspace(0,1,n_buckets)
    curve=(
        open_weight*np.exp(-10 * t**2)
        +close_weight*np.exp(-10 * (t - 1.0) ** 2)
        +mid_weight * np.ones(n_buckets)
    )
    curve=np.maximum(curve, 1e-6)
    return curve/curve.sum()

# SECTION 4:PARAMETER CALIBRATION
def calibrate_impact_parameters(df: pd.DataFrame) -> CalibrationResult:
    """
    Heuristically calibrate impact parameters from daily OHLCV.
    Permanent impact gamma:
        Regress next-day return on signed participation proxy.
    Temporary impact eta:
        Estimate from high-low volatility scaled by liquidity.
    Spread:
        Daily-bar proxy via high-low estimator, clipped to a sensible range.
    """
    notes:List[str] = []
    tmp=df.copy().dropna()
    n=len(tmp)
    sigma=float(tmp["LogReturns"].std())
    adv=float(tmp["Volume"].mean())
    price=float(tmp["Close"].iloc[-1])

    # Permanent impact proxy
    signed_vol=tmp["Volume"]*np.sign(tmp["Returns"].fillna(0.0))
    participation=signed_vol / max(adv, 1e-8)
    X_reg=participation.values[:-1]
    y_reg=tmp["Returns"].values[1:]
    slope_perm, _, r_perm, p_perm, _ = linregress(X_reg, y_reg)
    gamma=max(float(abs(slope_perm)), 1e-6)
    r2_perm=float(r_perm**2)
    if p_perm>0.10:
        notes.append(f"γ weakly identified on daily bars (p={p_perm:.2f}); conservative floor applied.")
        gamma=max(gamma,0.005)

    # Temporary impact proxy
    hl_sigma=float((np.log(tmp["High"] / tmp["Low"]) / (2 * np.sqrt(np.log(2)))).mean())
    eta=float(np.clip(hl_sigma / np.sqrt(max(adv / 1e6, 1e-8)), 0.01, 2.0))
    impact_proxy=tmp["High"] / tmp["Low"] - 1
    vol_proxy=tmp["Volume"] / max(adv, 1e-8)
    _, _, r_temp, _, _ = linregress(vol_proxy.values, impact_proxy.values)
    r2_temp=float(r_temp**2)

    # Spread proxy
    log_hl=np.log(tmp["High"] / tmp["Low"])
    beta=log_hl.rolling(2).sum().dropna()
    alpha_cs=(np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
    spread_f=2 * (np.exp(alpha_cs) - 1) / (1 + np.exp(alpha_cs))
    spread_bps = float(np.clip(spread_f.mean() * 10000, 1.0, 50.0))

    notes.append(f"Calibrated on {n} trading days.")
    notes.append(f"σ={sigma:.4f}/day  ADV={adv:,.0f}  price={price:.2f}")
    notes.append(f"η={eta:.4f}  γ={gamma:.5f}  spread={spread_bps:.1f}bps")
    notes.append(f"R²_perm={r2_perm:.3f}  R²_temp={r2_temp:.3f}")
    notes.append("Calibration uses daily-bar proxies; intraday trade/quote data would materially improve realism.")
    return CalibrationResult(
        eta=eta,
        gamma=gamma,
        sigma=sigma,
        adv=adv,
        spread_bps=spread_bps,
        price=price,
        r2_temp=r2_temp,
        r2_perm=r2_perm,
        notes=notes,
    )
# SECTION 5 — VWAP SCHEDULE
def vwap_schedule(X: float, T: int, volume_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute a VWAP schedule by allocating shares in proportion to expected volume."""
    if volume_curve is None:
        return np.full(T, X / T)
    if len(volume_curve)== T:
        weights=volume_curve.copy()
    else:
        orig=np.linspace(0, 1, len(volume_curve))
        new=np.linspace(0, 1, T)
        f=interp1d(orig, volume_curve, kind="linear")
        weights=f(new)
    weights=np.maximum(weights, 0)
    weights/=weights.sum()
    return X*weights

# SECTION 6 — ALMGREN-CHRISS MODEL
class AlmgrenChriss:
    """Almgren-Chriss schedule and expected-cost model with linear impact."""
    def __init__(self, market: MarketParams, order: OrderParams):
        self.m=market
        self.o=order
        self.tau=1.0
        sigma_dollar=self.m.price*self.m.sigma
        a_temp=self.m.price*self.m.eta/max(self.m.adv,1.0)
        self.kappa = np.sqrt(self.o.lam * sigma_dollar**2 / max(a_temp, 1e-12)) / self.tau

    def optimal_trajectory(self) -> np.ndarray:
        T, X = self.o.T, self.o.X
        k = np.arange(T + 1, dtype=float)
        z = self.kappa * T * self.tau
    # If urgency is extremely high, use near-immediate liquidation fallback
        if not np.isfinite(z) or z > 50:
            traj = np.zeros(T + 1)
            traj[0] = X
            return traj
        denom = np.sinh(z)
        if denom < 1e-12 or not np.isfinite(denom):
            return X * (1 - k / T)
        traj = X * np.sinh(self.kappa * (T - k) * self.tau) / denom
        return np.maximum(traj, 0.0)
    
    def optimal_trades(self) -> np.ndarray:# Shares traded in each period
        trades = np.maximum(-np.diff(self.optimal_trajectory()), 0.0)
        total = trades.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full(self.o.T, self.o.X / self.o.T)
        trades *= self.o.X / total
        return trades

    def expected_cost(self,trades: np.ndarray) -> float:
        
        impact_scale = self.m.price / max(self.m.adv, 1.0) ## expected cost in dollars,AC-style
        perm = 0.5 * self.m.gamma * impact_scale * self.o.X**2
        temp = (self.m.eta * impact_scale / self.tau) * float(np.sum(trades**2))
        return perm + temp

    def variance_cost(self, trajectory: np.ndarray) -> float:
        """Variance term in dollars^2."""
        sigma_dollar = self.m.price * self.m.sigma
        x_mid=trajectory[1:]
        return float(sigma_dollar**2 * self.tau * np.sum(x_mid**2))

    def execution_shortfall_bps(self, trades: np.ndarray) -> Dict[str, float]:
        """Analytical expected-cost breakdown in basis points."""
        X, S0, tau = self.o.X, self.m.price, self.tau
        traj = np.r_[X, X - np.cumsum(trades)]
        traj = np.maximum(traj, 0.0)

        impact_scale = self.m.price / max(self.m.adv, 1.0)
        cost_temp = (self.m.eta * impact_scale / tau) * float(np.sum(trades**2))
        cost_perm = 0.5 * self.m.gamma * impact_scale * X**2
        half_spread = (self.m.spread_bps / 10000) * S0 / 2
        cost_spread = half_spread * float(np.sum(np.abs(trades)))
        risk_dollar = self.o.lam * self.variance_cost(traj)
        total = cost_temp + cost_perm + cost_spread

        def bps(d: float) -> float:
            return round(d / (X * S0) * 10000, 3)

        return {
            "temporary_impact_bps": bps(cost_temp),
            "permanent_impact_bps": bps(cost_perm),
            "spread_cost_bps": bps(cost_spread),
            "risk_penalty_bps": bps(risk_dollar),
            "total_cost_bps": bps(total),
            "total_with_risk_bps": bps(total + risk_dollar),
        }
# SECTION 7 — PATHWISE SLIPPAGE SIMULATION
def simulate_slippage(
    trades:np.ndarray,
    price_path:np.ndarray,
    market:MarketParams,
    side:int=-1,
    label:str="Strategy",
) -> Dict:
    """
    Simulate realized execution prices on a concrete price path.
    For buys, higher execution prices are worse.
    For sells, lower execution prices are worse.
    Returns both realized shortfall vs arrival price and vs schedule-weighted VWAP benchmark.
    """
    T=len(trades)
    prices=np.asarray(price_path[:T], dtype=float)
    S0=float(prices[0])
    exec_prices=np.zeros(T)
    perm_shift=0.0
    spread_frac=(market.spread_bps / 10000) / 2.0

    for k in range(T):
        n_k=float(trades[k])
        participation=n_k / max(market.adv, 1e-12)
        temp_frac=market.eta*participation
        perm_shift+=market.gamma*participation
        if side>0:# buy
            exec_prices[k]=prices[k]*(1 + temp_frac + spread_frac + perm_shift)
        else:         # sell
            exec_prices[k]=prices[k]*(1-temp_frac-spread_frac-perm_shift)

    total_shares=float(np.sum(trades))
    avg_exec=float(np.dot(trades, exec_prices) / total_shares)
    sched_vwap=float(np.dot(trades, prices) / total_shares)
    # Positive numbers are worse costs regardless of side.
    if side>0:
        is_arrival_bps=(avg_exec-S0)/S0*10000
        is_sched_bps=(avg_exec-sched_vwap)/S0*10000
    else:
        is_arrival_bps=(S0-avg_exec)/S0*10000
        is_sched_bps=(sched_vwap-avg_exec)/S0*10000
    return {
        "label": label,
        "avg_exec_price": round(avg_exec,4),
        "arrival_price": round(S0,4),
        "schedule_vwap": round(sched_vwap, 4),
        "arrival_shortfall_bps": round(is_arrival_bps, 2),
        "schedule_shortfall_bps": round(is_sched_bps, 2),
        "exec_prices": exec_prices,
    }
def realized_cost_from_slippage(slippage: Dict) -> float:
    """Extract realized cost in basis points from pathwise slippage output."""
    return float(slippage["arrival_shortfall_bps"])

# SECTION 8 — MONTE CARLO ANALYSIS
def monte_carlo_execution(
    market: MarketParams,
    order: OrderParams,
    n_paths: int = 1000,
    seed: int = 0,
) -> Dict:
    """Simulate realized execution outcomes across many price paths."""
    rng = np.random.default_rng(seed)
    T, S0 = order.T, market.price
    ac = AlmgrenChriss(market, order)
    vol_curve = estimate_intraday_volume_curve(
        synthetic_market_data(126, S0=S0, sigma=market.sigma, adv=int(market.adv), seed=seed + 1)
    )
    schedules = {
        "TWAP": np.full(T, order.X / T),
        "VWAP": vwap_schedule(order.X, T, vol_curve),
        "AC-Optimal": ac.optimal_trades(),
    }
    results = {
        name: {"realized_cost_bps": [], "schedule_shortfall_bps": [], "net_alpha": []}
        for name in schedules
    }
    kappa_a = np.log(2) / order.alpha_halflife
    retained = float(np.mean(np.exp(-kappa_a * np.arange(1, T + 1))))
    base_alpha_pct = order.alpha_annual / 252 * T * retained * 100
    for _ in range(n_paths):
        daily_ret = rng.normal(0, market.sigma, T)
        prices = S0 * np.cumprod(1 + daily_ret)

        for name, trades in schedules.items():
            slip = simulate_slippage(trades, prices, market, side=order.side, label=name)
            realized_cost_bps = realized_cost_from_slippage(slip)
            net_alpha = base_alpha_pct - realized_cost_bps / 100

            results[name]["realized_cost_bps"].append(realized_cost_bps)
            results[name]["schedule_shortfall_bps"].append(float(slip["schedule_shortfall_bps"]))
            results[name]["net_alpha"].append(net_alpha)
    for name in results:
        for key in results[name]:
            results[name][key] = np.array(results[name][key], dtype=float)
    return results

# SECTION 9 — SENSITIVITY STUDY
def sensitivity_study(market: MarketParams, order: OrderParams) -> Dict:
    """One-at-a-time sensitivity of analytical AC total cost."""

    def cost_for(m: MarketParams, o: OrderParams) -> float:
        ac_ = AlmgrenChriss(m, o)
        return ac_.execution_shortfall_bps(ac_.optimal_trades())["total_cost_bps"]
    out: Dict[str, tuple] = {}
    lam_grid = np.linspace(0.5, 30, 40)
    out["lambda"] = (
        lam_grid,
        np.array([
            cost_for(market, OrderParams(order.X, order.T, float(l), order.alpha_annual, order.alpha_halflife, order.side))
            for l in lam_grid
        ]),
    )
    eta_grid = np.linspace(0.005, 0.5, 40)
    out["eta"] = (
        eta_grid,
        np.array([
            cost_for(MarketParams(market.sigma, float(e), market.gamma, market.adv, market.spread_bps, market.price), order)
            for e in eta_grid
        ]),
    )
    T_grid = np.arange(1, 21)
    out["horizon"] = (
        T_grid.astype(float),
        np.array([
            cost_for(market, OrderParams(order.X, int(t), order.lam, order.alpha_annual, order.alpha_halflife, order.side))
            for t in T_grid
        ]),
    )
    sig_grid = np.linspace(0.003, 0.04, 40)
    out["sigma"] = (
        sig_grid * 100,
        np.array([
            cost_for(MarketParams(float(s), market.eta, market.gamma, market.adv, market.spread_bps, market.price), order)
            for s in sig_grid
        ]),
    )
    adv_pct = np.linspace(0.01, 0.30, 40)
    out["adv_pct"] = (
        adv_pct * 100,
        np.array([
            cost_for(market, OrderParams(market.adv * float(p), order.T, order.lam, order.alpha_annual, order.alpha_halflife, order.side))
            for p in adv_pct
        ]),
    )
    return out

# SECTION 10 — WALK-FORWARD HISTORICAL BACKTEST
def historical_backtest(
    df: pd.DataFrame,
    market: MarketParams,
    order: OrderParams,
    rebalance_freq: int = 21,
) -> pd.DataFrame:
    """
    Walk-forward backtest using realized fills on historical close paths.
    For each rebalance date:
      - re-estimate local sigma and ADV from trailing data only
      - schedule TWAP, VWAP, AC execution over next T days
      - simulate realized fills on the subsequent path
      - compute net alpha proxy = decayed alpha capture - realized cost
    """
    closes = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    n = len(closes)
    T = order.T
    vol_curve = estimate_intraday_volume_curve(df)
    records: List[Dict] = []
    i = max(21, T)

    while i + T < n:
        price_window = closes[i:i + T]
        S0 = float(price_window[0])
        lookback = max(0, i - 60)
        hist_prices = closes[lookback:i]
        hist_volumes = volumes[lookback:i]
        past_ret = np.diff(np.log(hist_prices + 1e-12))

        local_sig = float(np.std(past_ret)) if len(past_ret) > 5 else market.sigma
        local_sig = max(local_sig, 0.003)
        local_adv = float(np.mean(hist_volumes)) if len(hist_volumes) > 0 else market.adv
        local_adv = max(local_adv, 1.0)

        m_local = MarketParams(
            sigma=local_sig,
            eta=market.eta,
            gamma=market.gamma,
            adv=local_adv,
            spread_bps=market.spread_bps,
            price=S0,
        )
        o_local = OrderParams(
            X=order.X,
            T=T,
            lam=order.lam,
            alpha_annual=order.alpha_annual,
            alpha_halflife=order.alpha_halflife,
            side=order.side,
        )
        ac = AlmgrenChriss(m_local, o_local)
        schedules = {
            "TWAP": np.full(T, order.X / T),
            "VWAP": vwap_schedule(order.X, T, vol_curve),
            "AC-Optimal": ac.optimal_trades(),
        }
        kappa_a = np.log(2) / order.alpha_halflife
        retained = float(np.mean(np.exp(-kappa_a * np.arange(1, T + 1))))
        captured_alpha_pct = order.alpha_annual / 252 * T * retained * 100

        for name, trades in schedules.items():
            slip = simulate_slippage(trades, price_window, m_local, side=order.side, label=name)
            realized_cost_bps = realized_cost_from_slippage(slip)
            net_alpha_pct = captured_alpha_pct - realized_cost_bps / 100
            records.append(
                {
                    "date": df.index[i],
                    "strategy": name,
                    "S0": round(S0, 2),
                    "sigma_local": round(local_sig * 100, 3),
                    "adv_local": int(local_adv),
                    "arrival_shortfall_bps": round(slip["arrival_shortfall_bps"], 2),
                    "schedule_shortfall_bps": round(slip["schedule_shortfall_bps"], 2),
                    "realized_cost_bps": round(realized_cost_bps, 2),
                    "captured_alpha_pct": round(captured_alpha_pct, 3),
                    "net_alpha_pct": round(net_alpha_pct, 3),
                }
            )
        i += rebalance_freq
    return pd.DataFrame(records)

# SECTION 11 — ALPHA DECAY
def alpha_decay_curve(t: np.ndarray, alpha_annual: float, halflife: float, T_horizon: int) -> np.ndarray:
    """Exponential alpha decay curve."""
    kappa = np.log(2) / halflife
    alpha0 = alpha_annual / 252 * T_horizon
    return alpha0 * np.exp(-kappa * t)

def net_realised_alpha(order: OrderParams, total_cost_bps: float) -> Dict[str, float]:
    """Compute net alpha after decay and costs."""
    kappa = np.log(2) / order.alpha_halflife
    t = np.arange(1, order.T + 1)
    retained = float(np.mean(np.exp(-kappa * t)))
    daily_alpha = order.alpha_annual / 252
    gross_captured = daily_alpha * order.T * retained
    cost_frac = total_cost_bps / 10000

    return {
        "gross_alpha_pct": round(order.alpha_annual * 100, 2),
        "captured_alpha_pct": round(gross_captured * 100, 3),
        "total_cost_pct": round(cost_frac * 100, 3),
        "net_alpha_pct": round((gross_captured - cost_frac) * 100, 2),
        "alpha_retained_pct": round(retained * 100, 1),
    }
# SECTION 12 — STRATEGY COMPARISON

def compare_strategies(market: MarketParams, order: OrderParams) -> pd.DataFrame:
    """Compare TWAP, VWAP, and AC using consistent analytical AC cost formulas."""
    ac = AlmgrenChriss(market, order)
    vol_curve = estimate_intraday_volume_curve(
        synthetic_market_data(126, S0=market.price, sigma=market.sigma, adv=int(market.adv), seed=11)
    )
    rows = []
    for name, trades in [
        ("TWAP", np.full(order.T, order.X / order.T)),
        ("VWAP", vwap_schedule(order.X, order.T, vol_curve)),
        ("AC-Optimal", ac.optimal_trades()),
    ]:
        costs = ac.execution_shortfall_bps(trades)
        net = net_realised_alpha(order, costs["total_cost_bps"])
        rows.append(
            {
                "Strategy": name,
                "Temp Impact (bps)": costs["temporary_impact_bps"],
                "Perm Impact (bps)": costs["permanent_impact_bps"],
                "Spread (bps)": costs["spread_cost_bps"],
                "Risk Penalty (bps)": costs["risk_penalty_bps"],
                "Total Cost (bps)": costs["total_cost_bps"],
                "Captured Alpha (%)": net["captured_alpha_pct"],
                "Net Alpha (%)": net["net_alpha_pct"],
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")

# SECTION 13 — VISUALIZATION
BG = "#0d0d14"
PANEL = "#13131e"
GRIDC = "#1e1e2e"
TEXT = "#dcd9d0"
MUTED = "#7a7888"
BLUE = "#4a9eff"
ORANGE = "#ff6b35"
GREEN = "#3ecf8e"
AMBER = "#ffb74d"
PINK = "#e879a0"
PURPLE = "#9d7fe8"

def _style(ax, title: str):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRIDC)
    ax.grid(color=GRIDC, linestyle="--", linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT, pad=8)

def plot_full_analysis(
    market: MarketParams,
    order: OrderParams,
    df_hist: pd.DataFrame,
    mc_results: Dict,
    sens: Dict,
    bt_df: pd.DataFrame,
):
    """Build 9-panel summary figure."""
    ac = AlmgrenChriss(market, order)
    vol_curve = estimate_intraday_volume_curve(df_hist)
    days = np.arange(1, order.T + 1)

    twap_t = np.full(order.T, order.X / order.T)
    vwap_t = vwap_schedule(order.X, order.T, vol_curve)
    ac_t = ac.optimal_trades()
    ac_traj = ac.optimal_trajectory()

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.46, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    twap_rem = order.X - np.cumsum(twap_t)
    vwap_rem = order.X - np.cumsum(vwap_t)
    ac_rem = ac_traj[1:]
    ax1.plot(days, twap_rem / order.X * 100, color=GREEN, lw=1.8, ls="--", label="TWAP")
    ax1.plot(days, vwap_rem / order.X * 100, color=BLUE, lw=1.8, ls="-.", label="VWAP")
    ax1.plot(days, ac_rem / order.X * 100, color=ORANGE, lw=2.2, label="AC-Optimal")
    ax1.set_xlabel("Trading day")
    ax1.set_ylabel("Remaining position (%)")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax1, "A  Execution Trajectories")

    ax2 = fig.add_subplot(gs[0, 1])
    n_b = len(vol_curve)
    ax2.bar(range(n_b), vol_curve * 100, color=BLUE, alpha=0.75, width=0.8)
    tick_idx = range(0, n_b, max(1, n_b // 5))
    hour_labels = [f"{9 + int(i * 6.5 / n_b)}:{int((i * 390 / n_b) % 60):02d}" for i in tick_idx]
    ax2.set_xticks(list(tick_idx))
    ax2.set_xticklabels(hour_labels, fontsize=7.5)
    ax2.set_ylabel("Volume share (%)")
    _style(ax2, "B  Proxy Intraday Volume Curve")

    ax3 = fig.add_subplot(gs[0, 2])
    impact_scale = market.price / max(market.adv, 1.0)
    temp_c = market.eta * impact_scale * ac_t**2 / (order.X * market.price) * 10000
    perm_c = 0.5 * market.gamma * impact_scale * ac_t**2 / (order.X * market.price) * 10000
    w = 0.35
    ax3.bar(days - w / 2, temp_c, width=w, color=BLUE, alpha=0.85, label="Temporary")
    ax3.bar(days + w / 2, perm_c, width=w, color=ORANGE, alpha=0.85, label="Permanent")
    ax3.set_xlabel("Trading day")
    ax3.set_ylabel("Cost (bps)")
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax3, "C  Analytical Impact per Interval")

    ax4 = fig.add_subplot(gs[1, 0])
    for name, color in [("TWAP", GREEN), ("VWAP", BLUE), ("AC-Optimal", ORANGE)]:
        data = mc_results[name]["realized_cost_bps"]
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
    ax4.hist(data, bins=40, alpha=0.55, color=color, label=f"{name}  μ={data.mean():.1f}bps")
    ax4.set_xlabel("Realized cost (bps)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax4, "D  Monte Carlo — Realized Cost")

    ax5 = fig.add_subplot(gs[1, 1])
    for name, color in [("TWAP", GREEN), ("VWAP", BLUE), ("AC-Optimal", ORANGE)]:
        data = mc_results[name]["net_alpha"]
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
    ax5.hist(data, bins=40, alpha=0.55, color=color, label=f"{name}  μ={data.mean():.2f}%")
    ax5.axvline(0, color=MUTED, lw=1, ls="--")
    ax5.set_xlabel("Net alpha (%)")
    ax5.set_ylabel("Frequency")
    ax5.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax5, "E  Monte Carlo — Net Alpha")

    ax6 = fig.add_subplot(gs[1, 2])
    param_styles = {
        "lambda": ("λ (urgency)", ORANGE),
        "eta": ("η (temp impact)", BLUE),
        "horizon": ("T (days)", GREEN),
        "sigma": ("σ% (vol)", AMBER),
        "adv_pct": ("Order size %ADV", PINK),
    }
    for key, (label, color) in param_styles.items():
        xv, yv = sens[key]
        xn = (xv - xv.min()) / (xv.max() - xv.min() + 1e-12)
        ax6.plot(xn, yv, color=color, lw=1.7, label=label)
    ax6.set_xlabel("Parameter (normalized 0→1)")
    ax6.set_ylabel("Analytical total cost (bps)")
    ax6.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax6, "F  Parameter Sensitivity")

    ax7 = fig.add_subplot(gs[2, 0])
    t_c = np.linspace(0, order.T * 2.0, 300)
    a_d = alpha_decay_curve(t_c, order.alpha_annual, order.alpha_halflife, order.T)
    c_val = ac.execution_shortfall_bps(ac_t)["total_cost_bps"] / 100
    ax7.fill_between(t_c, a_d * 100, alpha=0.12, color=BLUE)
    ax7.plot(t_c, a_d * 100, color=BLUE, lw=2, label=f"Alpha (T½={order.alpha_halflife}d)")
    ax7.axhline(c_val, color=ORANGE, lw=1.5, ls="--", label="AC total cost")
    ax7.axvline(order.T, color=AMBER, lw=1.2, ls=":", label=f"Horizon T={order.T}d")
    cross_i = int(np.argmin(np.abs(a_d * 100 - c_val)))
    ax7.axvline(t_c[cross_i], color=GREEN, lw=1, ls=":", alpha=0.7, label=f"Crossover ≈ d{t_c[cross_i]:.0f}")
    ax7.set_xlabel("Days")
    ax7.set_ylabel("Alpha / cost (%)")
    ax7.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax7, "G  Alpha Decay vs Cost")

    ax8 = fig.add_subplot(gs[2, 1])
    for name, color in [("TWAP", GREEN), ("VWAP", BLUE), ("AC-Optimal", ORANGE)]:
        sub = bt_df[bt_df["strategy"] == name].sort_values("date")
        if sub.empty:
            continue
        cum = sub["net_alpha_pct"].cumsum().values
        ax8.plot(range(len(cum)), cum, color=color, lw=1.9, label=name)
    ax8.axhline(0, color=MUTED, lw=0.8, ls="--")
    ax8.set_xlabel("Trade number")
    ax8.set_ylabel("Cumulative net alpha (%)")
    ax8.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax8, "H  Walk-Forward Backtest")

    ax9 = fig.add_subplot(gs[2, 2])
    df_cmp = compare_strategies(market, order)
    strats = df_cmp.index.tolist()
    costs_p = df_cmp["Total Cost (bps)"].values / 100
    nets = df_cmp["Net Alpha (%)"].values
    x = np.arange(len(strats))
    ax9.bar(x, order.alpha_annual * 100, color=GREEN, alpha=0.22, width=0.55, label="Gross alpha")
    ax9.bar(x, costs_p, color=ORANGE, alpha=0.82, width=0.55, label="Total cost")
    ax9.bar(x, nets, bottom=costs_p, color=BLUE, alpha=0.88, width=0.55, label="Net alpha")
    for xi, (c, n) in enumerate(zip(costs_p, nets)):
        ax9.text(xi, c + n + 0.04, f"{n:.2f}%", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")
    ax9.set_xticks(x)
    ax9.set_xticklabels(strats, fontsize=9)
    ax9.set_ylabel("Return (%)")
    ax9.legend(fontsize=8, facecolor=PANEL, labelcolor=TEXT, framealpha=0.9)
    _style(ax9, "I  Net Alpha by Strategy")

    fig.suptitle(
        f"VWAP & Almgren-Chriss Execution Research Framework  ·  "
        f"Order: {order.X / market.adv * 100:.0f}%ADV  ·  "
        f"T={order.T}d  ·  σ={market.sigma*100:.2f}%/day  ·  "
        f"λ={order.lam}  ·  η={market.eta:.3f}  ·  γ={market.gamma:.4f}",
        fontsize=11,
        color=TEXT,
        y=0.998,
        fontweight="bold",
    )
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/vwap_ac_analysis.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print("\nFigure saved → images/vwap_ac_analysis.png")

# SECTION 14 — MAIN
def main():
    print("=" * 68)
    print("  VWAP Execution & Almgren-Chriss Market Impact Model")
    print("=" * 68)

    ticker = "SPY"

    print(f"\n[1/7] Fetching market data ({ticker})...")
    df = fetch_market_data(ticker, period="1y")
    if df is None:
        df = synthetic_market_data(252, S0=480.0, mu=0.08, sigma=0.012, adv=1_000_000)

    print("\n[2/7] Calibrating impact parameters from historical data...")
    cal = calibrate_impact_parameters(df)
    for note in cal.notes:
        print(f"  {note}")

    market = MarketParams(
        sigma=cal.sigma,
        eta=cal.eta,
        gamma=cal.gamma,
        adv=cal.adv,
        spread_bps=cal.spread_bps,
        price=cal.price,
    )
    order = OrderParams(
        X=market.adv * 0.10,
        T=5,
        lam=8.0,
        alpha_annual=0.15,
        alpha_halflife=10.0,
        side=-1,
    )
    print("\n[3/7] Almgren-Chriss analytical cost breakdown...")
    ac = AlmgrenChriss(market, order)
    ac_t = ac.optimal_trades()
    costs = ac.execution_shortfall_bps(ac_t)
    print("\n  Cost breakdown:")
    for k, v in costs.items():
        print(f"    {k:<30} {v:>10.3f} bps")
    net = net_realised_alpha(order, costs["total_cost_bps"])
    print("\n  Alpha P&L:")
    for k, v in net.items():
        print(f"    {k:<30} {v:>10.2f}%")

    print("\n[4/7] Strategy comparison...")
    df_cmp = compare_strategies(market, order)
    print("\n" + df_cmp.to_string())

    print("\n[5/7] Monte Carlo simulation (1000 paths)...")
    mc = monte_carlo_execution(market, order, n_paths=1000)
    print(f"\n  {'Strategy':<14}  {'Realized Cost μ ± σ (bps)':<30}  Net Alpha μ ± σ (%)")
    print(f"  {'-'*72}")
    for name in ["TWAP", "VWAP", "AC-Optimal"]:
        c = mc[name]["realized_cost_bps"]
        na = mc[name]["net_alpha"]
        print(f"  {name:<14}  {c.mean():6.1f} ± {c.std():5.1f}                 {na.mean():+.2f} ± {na.std():.2f}%")

    print("\n[6/7] Parameter sensitivity study...")
    sens = sensitivity_study(market, order)
    print("  Sweeps complete: λ, η, T, σ, order-size (%ADV)")

    print("\n[7/7] Walk-forward historical backtest...")
    bt_df = historical_backtest(df, market, order, rebalance_freq=21)
    summary = (
        bt_df.groupby("strategy")
        .agg(
            trades=("net_alpha_pct", "count"),
            avg_realized_cost_bps=("realized_cost_bps", "mean"),
            avg_schedule_shortfall_bps=("schedule_shortfall_bps", "mean"),
            avg_captured_alpha_pct=("captured_alpha_pct", "mean"),
            total_net_alpha_pct=("net_alpha_pct", "sum"),
            sharpe_like=("net_alpha_pct", lambda x: x.mean() / (x.std() + 1e-10)),
        )
        .round(3)
    )
    print("\n" + summary.to_string())
    os.makedirs("images", exist_ok=True)

    summary.to_csv("images/backtest_summary.csv")
    fig, ax = plt.subplots(figsize=(16,4.8))
    ax.axis('off')
    table = ax.table(
        cellText=summary.round(3).values,
        colLabels=summary.columns,
        rowLabels=summary.index,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(1.2)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eaeaea")
        if col == -1:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig("images/summary_table.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("\nGenerating 9-panel figure...")
    plot_full_analysis(market,order, df, mc, sens, bt_df)
    print("\nDone.")

if __name__ == "__main__":
    main()
