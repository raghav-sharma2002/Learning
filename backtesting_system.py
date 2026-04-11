from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


TRADING_DAYS = 252


@dataclass
class BacktestArtifacts:
    data: pd.DataFrame
    feature_columns: List[str]
    train_results: pd.DataFrame
    test_results: pd.DataFrame
    metrics: Dict[str, Dict[str, float]]
    stress_summary: pd.DataFrame
    factor_importance: pd.DataFrame


def fetch_price_data(
    symbols: Iterable[str],
    start: str = "2018-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = list(symbols)
    if not tickers:
        raise ValueError("At least one symbol is required.")

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw.empty:
        raise ValueError("No market data was returned. Check the ticker list or date range.")

    frames: List[pd.DataFrame] = []
    if len(tickers) == 1:
        ticker = tickers[0]
        frame = raw.copy()
        frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
        frame["symbol"] = ticker
        frames.append(frame.reset_index())
    else:
        for ticker in tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker].copy()
            frame.columns = [str(col).lower().replace(" ", "_") for col in frame.columns]
            frame["symbol"] = ticker
            frames.append(frame.reset_index())

    data = pd.concat(frames, ignore_index=True)
    data = data.rename(columns={"Date": "date", "Datetime": "date"})
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data = data.dropna(subset=["close"])
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    return data


def build_factor_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise ValueError("Cannot engineer factors from an empty dataset.")

    frames: List[pd.DataFrame] = []
    for symbol, frame in data.groupby("symbol", sort=False):
        item = frame.copy().sort_values("date")
        close = item["close"]
        volume = item["volume"].replace(0, np.nan)

        item["return_1d"] = close.pct_change()
        item["return_5d"] = close.pct_change(5)
        item["return_21d"] = close.pct_change(21)
        item["sma_10"] = close.rolling(10).mean()
        item["sma_50"] = close.rolling(50).mean()
        item["trend_gap"] = item["sma_10"] / item["sma_50"] - 1
        item["vol_20"] = item["return_1d"].rolling(20).std() * np.sqrt(TRADING_DAYS)
        item["volume_z"] = (np.log(volume).replace([np.inf, -np.inf], np.nan) - np.log(volume).rolling(20).mean()) / np.log(volume).rolling(20).std()

        delta = close.diff()
        gains = delta.clip(lower=0).rolling(14).mean()
        losses = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gains / losses.replace(0, np.nan)
        item["rsi_14"] = 100 - (100 / (1 + rs))

        item["forward_return_5d"] = close.shift(-5) / close - 1
        item["symbol"] = symbol
        frames.append(item)

    factors = pd.concat(frames, ignore_index=True)
    factors = factors.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return factors


class AlgorithmicTradingSystem:
    """A reusable multi-factor strategy wrapper for ranking and position sizing."""

    def __init__(
        self,
        feature_columns: List[str],
        buy_threshold: float = 0.55,
        short_threshold: float = 0.45,
        max_leverage: float = 1.0,
    ) -> None:
        self.feature_columns = feature_columns
        self.buy_threshold = buy_threshold
        self.short_threshold = short_threshold
        self.max_leverage = max_leverage
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=8,
            random_state=42,
        )

    def fit(self, train_df: pd.DataFrame) -> None:
        self.model.fit(train_df[self.feature_columns], train_df["forward_return_5d"])

    def score(self, df: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(df[self.feature_columns])
        ranked = pd.Series(preds, index=df.index)
        return ranked.groupby(df["date"]).rank(pct=True)

    def generate_positions(self, df: pd.DataFrame) -> pd.Series:
        score_rank = self.score(df)
        long_signal = (score_rank >= self.buy_threshold).astype(float)
        short_signal = -(score_rank <= self.short_threshold).astype(float)
        raw = long_signal + short_signal

        def normalize(day_positions: pd.Series) -> pd.Series:
            gross = day_positions.abs().sum()
            if gross == 0:
                return day_positions
            return day_positions / gross * self.max_leverage

        return raw.groupby(df["date"]).transform(normalize)


def attach_strategy_returns(df: pd.DataFrame, strategy: AlgorithmicTradingSystem) -> pd.DataFrame:
    results = df.copy()
    results["position"] = strategy.generate_positions(results)
    # Use the next-day return as the realized strategy outcome from today's signal.
    results["next_day_return"] = results.groupby("symbol")["return_1d"].shift(-1)
    results["strategy_return"] = results["position"] * results["next_day_return"]
    results["benchmark_return"] = results.groupby("date")["next_day_return"].transform("mean")
    results = results.dropna(subset=["next_day_return"]).reset_index(drop=True)
    return results


def aggregate_daily_returns(results: pd.DataFrame) -> pd.DataFrame:
    daily = (
        results.groupby("date")
        .agg(
            strategy_return=("strategy_return", "sum"),
            benchmark_return=("benchmark_return", "mean"),
        )
        .reset_index()
        .sort_values("date")
    )
    daily["strategy_equity"] = (1 + daily["strategy_return"].fillna(0.0)).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"].fillna(0.0)).cumprod()
    daily["strategy_drawdown"] = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1
    daily["benchmark_drawdown"] = daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1
    return daily


def evaluate_performance(df: pd.DataFrame, label: str) -> Dict[str, float]:
    daily = aggregate_daily_returns(df)
    strat_returns = daily["strategy_return"].fillna(0.0)

    periods = len(daily)
    cumulative_return = daily["strategy_equity"].iloc[-1] - 1
    annual_return = (
        daily["strategy_equity"].iloc[-1] ** (TRADING_DAYS / periods) - 1
        if periods > 0
        else np.nan
    )
    annual_vol = strat_returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_vol if annual_vol else np.nan

    return {
        "label": label,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": daily["strategy_drawdown"].min(),
        "hit_rate": (strat_returns > 0).mean(),
        "cumulative_return": cumulative_return,
    }


def run_backtest(
    data: pd.DataFrame,
    feature_columns: List[str],
    test_start: str = "2024-01-01",
) -> BacktestArtifacts:
    dataset = data.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    dataset["date"] = pd.to_datetime(dataset["date"])

    train_df = dataset[dataset["date"] < pd.Timestamp(test_start)].copy()
    test_df = dataset[dataset["date"] >= pd.Timestamp(test_start)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("The chosen split produced an empty train or test set.")

    strategy = AlgorithmicTradingSystem(feature_columns=feature_columns)
    strategy.fit(train_df)

    train_results = attach_strategy_returns(train_df, strategy)
    test_results = attach_strategy_returns(test_df, strategy)

    metrics = {
        "train": evaluate_performance(train_results, "train"),
        "test": evaluate_performance(test_results, "test"),
    }
    factor_importance = compute_factor_importance(strategy, train_df, feature_columns)
    stress_summary = counterfactual_stress_analysis(test_df, strategy, feature_columns)

    return BacktestArtifacts(
        data=dataset,
        feature_columns=feature_columns,
        train_results=train_results,
        test_results=test_results,
        metrics=metrics,
        stress_summary=stress_summary,
        factor_importance=factor_importance,
    )


def _attach_strategy_returns(df: pd.DataFrame, strategy: AlgorithmicTradingSystem) -> pd.DataFrame:
    return attach_strategy_returns(df, strategy)


def compute_factor_importance(
    strategy: AlgorithmicTradingSystem,
    train_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    importance = permutation_importance(
        strategy.model,
        train_df[feature_columns],
        train_df["forward_return_5d"],
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_squared_error",
    )
    return (
        pd.DataFrame(
            {
                "factor": feature_columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def counterfactual_stress_analysis(
    test_df: pd.DataFrame,
    strategy: AlgorithmicTradingSystem,
    feature_columns: List[str],
) -> pd.DataFrame:
    baseline = evaluate_performance(attach_strategy_returns(test_df, strategy), "baseline")
    rows: List[Dict[str, float | str]] = []

    scenario_builders: List[Tuple[str, Dict[str, float]]] = [
        ("higher_volatility", {"vol_20": 1.35}),
        ("momentum_crash", {"return_5d": -1.0, "return_21d": -1.0, "trend_gap": -0.75}),
        ("liquidity_shock", {"volume_z": -2.0}),
        ("oversold_mean_reversion", {"rsi_14": -20.0, "return_5d": -0.5}),
    ]

    for name, adjustments in scenario_builders:
        shocked = apply_counterfactual(test_df, adjustments, feature_columns)
        shocked_results = attach_strategy_returns(shocked, strategy)
        metrics = evaluate_performance(shocked_results, name)
        rows.append(
            {
                "scenario": name,
                "annual_return": metrics["annual_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "return_delta_vs_baseline": metrics["cumulative_return"] - baseline["cumulative_return"],
                "drawdown_delta_vs_baseline": metrics["max_drawdown"] - baseline["max_drawdown"],
            }
        )

    summary = pd.DataFrame(rows).sort_values("return_delta_vs_baseline")
    return summary.reset_index(drop=True)


def apply_counterfactual(
    df: pd.DataFrame,
    adjustments: Dict[str, float],
    feature_columns: List[str],
) -> pd.DataFrame:
    shocked = df.copy()
    for column, magnitude in adjustments.items():
        if column not in shocked.columns:
            continue
        if column in {"return_5d", "return_21d", "trend_gap"}:
            shocked[column] = shocked[column] * magnitude
        elif column == "vol_20":
            shocked[column] = shocked[column] * magnitude
        elif column == "volume_z":
            shocked[column] = shocked[column] + magnitude
        elif column == "rsi_14":
            shocked[column] = np.clip(shocked[column] + magnitude, 0, 100)

    missing = [col for col in feature_columns if col not in shocked.columns]
    if missing:
        raise ValueError(f"Counterfactual data is missing required features: {missing}")
    return shocked


def summarize_recent_window(results: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    daily = aggregate_daily_returns(results)
    cutoff = daily["date"].max() - pd.DateOffset(months=months)
    daily = daily[daily["date"] >= cutoff].copy().reset_index(drop=True)
    daily["strategy_equity"] = (1 + daily["strategy_return"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()
    return daily
