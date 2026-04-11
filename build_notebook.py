from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "financial_backtesting_stress_analysis.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(text).strip())


def build_notebook() -> None:
    cells = [
        md(
            """
            # A Book-Style Tutorial on Financial Backtesting, Stress Testing, and Causal Thinking

            This notebook is intentionally written like a slow, careful chapter in a textbook.

            It is for two audiences at the same time:

            1. a finance beginner who wants to understand what prices, returns, factors, backtests, and benchmarks mean,
            2. a causality beginner who wants to understand what a counterfactual is, what this notebook can and cannot prove, and why predictive success is not the same thing as causal truth.

            We will do six things:

            1. retrieve financial data,
            2. explore the raw data visually,
            3. engineer trading factors step by step,
            4. plug those factors into an algorithmic trading system,
            5. run counterfactual-style stress scenarios,
            6. test the strategy again on the most recent market window.

            The notebook is deliberately verbose. Code is broken into small cells, lines are commented heavily, and the explanations aim to go beyond the minimum.
            """
        ),
        md(
            """
            ## How to Read This Notebook

            Think of the notebook as a short book with parts and chapters:

            - **Part I. Setup**
              We import libraries, explain what each one does, and choose our research settings.
            - **Part II. Raw Data**
              We download market data, inspect it, and check whether it looks trustworthy.
            - **Part III. Factor Engineering**
              We turn raw prices and volume into features that a strategy can use.
            - **Part IV. Exploratory Analysis**
              We visualize the factors before modeling them.
            - **Part V. Train/Test Logic**
              We explain time splits and leakage risk.
            - **Part VI. Backtesting**
              We fit a trading system and turn predictions into positions and returns.
            - **Part VII. Factor Interpretation**
              We ask which factors the model used most heavily.
            - **Part VIII. Counterfactual Stress Analysis**
              We perturb the features in scenario form and see how the strategy responds.
            - **Part IX. Recent Re-Test**
              We isolate the most recent window to see whether recent behavior still looks acceptable.
            - **Part X. Interpretation and Limitations**
              We close by separating useful evidence from over-claiming.
            """
        ),
        md(
            """
            ## Vocabulary Before We Touch the Code

            A few beginner-friendly definitions:

            - **Ticker**: the short symbol used to identify an asset, such as `AAPL` for Apple.
            - **OHLCV data**: the standard columns in market data.
              `Open`, `High`, `Low`, `Close`, and `Volume`.
            - **Return**: the percentage change in price.
              If a stock goes from 100 to 101, the return is 1%.
            - **Factor / feature**: a variable we think might help explain or predict future returns.
              In this notebook, examples include momentum, volatility, trend gap, and RSI.
            - **Target**: the future quantity we try to predict.
              Here it is a 5-day forward return.
            - **Backtest**: a historical simulation where we pretend we were trading in the past using only information that would have been available at that time.
            - **Benchmark**: a simpler reference strategy used for comparison.
              Here we use an equal-weight market-style benchmark over the same universe.
            - **Drawdown**: the percentage drop from a previous peak in the equity curve.
            - **Counterfactual**: a “what if the world had looked different?” question.
            - **Causality**: the study of what truly causes what, not merely what correlates with what.

            The last two ideas matter a lot.
            This notebook does **not** run a randomized experiment on markets.
            It uses observational data.
            That means the stress analysis below is best understood as **counterfactual-style scenario analysis**, not a proof of causal structure.
            """
        ),
        md(
            """
            # Part I. Setup

            ## Chapter 1. Import the Tools and Explain Every Library

            Good research notebooks do not just import things and move on.
            They explain why each tool exists.
            """
        ),
        code(
            """
            from pathlib import Path  # Path gives us safe, readable filesystem paths.
            import sys  # sys lets us modify Python's import search path.
            import warnings  # warnings lets us quiet noisy library messages during teaching.

            from IPython.display import display  # display renders rich tables and objects nicely inside notebooks.
            import matplotlib.pyplot as plt  # Matplotlib is the foundational plotting library.
            import numpy as np  # NumPy gives us fast numerical arrays and math operations.
            import pandas as pd  # pandas gives us DataFrames, the main table object used in this notebook.
            import seaborn as sns  # seaborn makes statistical plots look cleaner and easier to build.

            NOTEBOOK_DIR = Path.cwd().resolve()  # This is the folder from which the notebook is running.
            ROOT = NOTEBOOK_DIR if (NOTEBOOK_DIR / "src").exists() else NOTEBOOK_DIR.parents[0]  # This finds the project root robustly.

            if str(ROOT) not in sys.path:  # If Python cannot yet import our local project modules...
                sys.path.insert(0, str(ROOT))  # ...prepend the project root so local imports work.

            from src.backtesting_system import AlgorithmicTradingSystem  # Our reusable trading system class.
            from src.backtesting_system import apply_counterfactual  # Our feature-shock helper for scenario analysis.
            from src.backtesting_system import compute_factor_importance  # Our permutation-importance helper.
            from src.backtesting_system import evaluate_performance  # Our performance-metric helper.
            from src.backtesting_system import fetch_price_data  # Our market-data retrieval helper.
            from src.backtesting_system import summarize_recent_window  # Our helper for the recent-window re-test.

            warnings.filterwarnings("ignore")  # We suppress noisy warnings so the teaching signal stays visible.

            pd.set_option("display.max_columns", 80)  # Show more columns before pandas truncates the view.
            pd.set_option("display.width", 180)  # Make wide tables easier to read in notebook output.
            sns.set_theme(style="whitegrid", context="talk")  # Use a clean plotting theme for explanation-heavy work.
            """
        ),
        md(
            """
            ## Chapter 2. Choose the Research Universe and the Time Horizon

            Before downloading data we must decide:

            - which assets we want to study,
            - how far back to go,
            - where the out-of-sample test period begins,
            - and how much “recent data” counts as recent for the final re-test.

            We deliberately use a mixed equity universe rather than a single stock.
            That makes the cross-sectional ranking strategy more meaningful.
            """
        ),
        code(
            """
            symbols = [  # A mixed universe helps the ranking model compare assets against one another.
                "AAPL",  # Apple
                "MSFT",  # Microsoft
                "NVDA",  # Nvidia
                "AMZN",  # Amazon
                "META",  # Meta
                "JPM",   # JPMorgan
                "XOM",   # Exxon Mobil
                "LLY",   # Eli Lilly
                "UNH",   # UnitedHealth
                "CAT",   # Caterpillar
                "SPY",   # S&P 500 ETF
                "QQQ",   # Nasdaq-100 ETF
            ]

            start_date = "2018-01-01"  # We start early enough to cover multiple market regimes.
            test_start = "2024-01-01"  # Everything from this date onward will be treated as out-of-sample.
            recent_months = 6  # At the end we will isolate the most recent six months.
            trading_days_per_year = 252  # This is the market convention for annualizing daily data.

            print("Number of symbols:", len(symbols))
            print("First few symbols:", symbols[:5])
            print("Train/Test split date:", test_start)
            """
        ),
        md(
            """
            # Part II. Retrieve and Inspect the Raw Data

            ## Chapter 3. Download the Historical Market Data

            We use the helper function `fetch_price_data`, which internally uses `yfinance`.

            Why not start from the strategy immediately?
            Because data quality problems are one of the fastest ways to ruin an analysis.
            The first responsibility of a backtest is to earn the right to be trusted.
            """
        ),
        code(
            """
            raw_data = fetch_price_data(  # Ask the helper to download OHLCV data.
                symbols=symbols,  # Use the symbol list chosen above.
                start=start_date,  # Download data from the chosen start date onward.
            )

            raw_data = raw_data.sort_values(["symbol", "date"]).reset_index(drop=True)  # Sort cleanly for all later rolling calculations.

            print("Raw data shape:", raw_data.shape)
            print("First available date:", raw_data["date"].min())
            print("Last available date:", raw_data["date"].max())

            raw_data.head(10)
            """
        ),
        code(
            """
            raw_data.tail(10)
            """
        ),
        md(
            """
            ## Chapter 4. Check Coverage by Symbol

            A multi-asset notebook should always confirm:

            - when each symbol starts,
            - when each symbol ends,
            - and whether each symbol has roughly the expected number of rows.

            This matters because irregular coverage can quietly bias both factor construction and backtest results.
            """
        ),
        code(
            """
            coverage_summary = (
                raw_data.groupby("symbol")
                .agg(
                    first_date=("date", "min"),
                    last_date=("date", "max"),
                    observations=("date", "count"),
                    min_close=("close", "min"),
                    max_close=("close", "max"),
                )
                .sort_index()
            )

            coverage_summary
            """
        ),
        md(
            """
            ## Chapter 5. Visualize the Raw Price Paths

            Raw prices of different assets live on very different scales.
            Apple is not directly comparable to SPY in dollar level alone.

            To compare them visually, we normalize each asset to start at 1.
            Then a value of 1.80 means “up 80% since the beginning of the sample,” regardless of the original price level.
            """
        ),
        code(
            """
            price_pivot = raw_data.pivot(index="date", columns="symbol", values="close")  # Turn long-form prices into a wide date-by-symbol matrix.

            first_valid_prices = price_pivot.apply(  # Find each symbol's own first valid close.
                lambda column: column.dropna().iloc[0] if column.notna().any() else np.nan
            )

            normalized_prices = price_pivot.divide(first_valid_prices, axis=1)  # Divide every column by its own first valid price.

            ax = normalized_prices.plot(  # Plot all normalized price paths together.
                figsize=(14, 7),
                title="Normalized Price Paths (Each Series Starts at 1)",
                alpha=0.85,
            )
            ax.set_ylabel("Normalized Price Level")
            ax.set_xlabel("Date")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 6. Convert Prices into Daily Returns

            Finance beginners often ask:
            “Why do quants use returns instead of raw prices?”

            Because returns are scale-free.
            A 1% move means the same thing whether the stock price is 20 or 200.

            We compute a one-day percentage return for each symbol:

            \\[
            r_t = \\frac{P_t}{P_{t-1}} - 1
            \\]
            """
        ),
        code(
            """
            raw_data["return_1d"] = raw_data.groupby("symbol")["close"].pct_change()  # Compute one-day percentage returns within each symbol.

            raw_data[["date", "symbol", "close", "return_1d"]].head(12)
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.histplot(  # A histogram shows the distribution of daily returns.
                raw_data["return_1d"].dropna(),
                bins=80,
                kde=True,
                color="steelblue",
            )
            plt.title("Distribution of One-Day Returns Across the Universe")
            plt.xlabel("One-Day Return")
            plt.ylabel("Count")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 7. Check for Missing Values and Oddities

            This is boring work.
            It is also what separates credible analysis from fragile analysis.

            Missing prices, zero volumes, and partial downloads can propagate into rolling features and create misleading results.
            """
        ),
        code(
            """
            missing_summary = raw_data.isna().sum().sort_values(ascending=False)  # Count missing values in every column.

            missing_summary[missing_summary > 0]
            """
        ),
        code(
            """
            sample_symbol = "AAPL"  # We will inspect one symbol closely as a sanity check.

            raw_data.query("symbol == @sample_symbol")[["date", "close", "volume", "return_1d"]].tail(15)
            """
        ),
        md(
            """
            # Part III. Engineer the Trading Factors Slowly

            ## Chapter 8. Why Factors Exist

            A factor is a distilled signal extracted from raw data.

            Instead of handing the model raw prices and hoping for the best, we transform the data into quantities with economic or behavioral meaning:

            - **Momentum**: has the asset been going up recently?
            - **Trend gap**: is the short moving average above the long moving average?
            - **Volatility**: how choppy or unstable has the asset been?
            - **Liquidity / volume surprise**: is trading activity unusually high or low?
            - **RSI**: is the asset recently overbought or oversold?

            We will build each feature in separate steps.
            """
        ),
        md(
            """
            ## Chapter 9. Start from a Fresh Working Table

            We create a new table so that feature engineering does not mutate the original raw dataset in a confusing way.
            """
        ),
        code(
            """
            factor_data = raw_data.copy()  # Work on a copy so the raw table stays intact.

            factor_data = factor_data.sort_values(["symbol", "date"]).reset_index(drop=True)  # Sorting matters before rolling windows.

            grouped_close = factor_data.groupby("symbol")["close"]  # This grouped Series lets us compute symbol-specific rolling features.
            grouped_volume = factor_data.groupby("symbol")["volume"]  # We keep grouped volume for later liquidity calculations.

            factor_data.head(5)
            """
        ),
        md(
            """
            ## Chapter 10. Compute Short-Horizon and Medium-Horizon Returns

            Returns over multiple horizons give us a simple momentum view.

            Intuition:

            - `return_5d` asks: “How much has this asset moved over roughly one trading week?”
            - `return_21d` asks: “How much has this asset moved over roughly one trading month?”
            """
        ),
        code(
            """
            factor_data["return_5d"] = grouped_close.pct_change(5)  # Five-day momentum.
            factor_data["return_21d"] = grouped_close.pct_change(21)  # Twenty-one-day momentum.

            factor_data[["date", "symbol", "close", "return_5d", "return_21d"]].head(12)
            """
        ),
        md(
            """
            ## Chapter 11. Build Moving Averages and the Trend Gap

            A moving average smooths noisy day-to-day changes.

            We use:

            - a 10-day moving average for the short trend,
            - a 50-day moving average for the longer trend.

            Then we define:

            \\[
            \\text{trend gap} = \\frac{SMA_{10}}{SMA_{50}} - 1
            \\]

            If this value is positive, the short trend is above the longer trend.
            """
        ),
        code(
            """
            factor_data["sma_10"] = grouped_close.transform(  # Compute the 10-day simple moving average.
                lambda series: series.rolling(10).mean()
            )

            factor_data["sma_50"] = grouped_close.transform(  # Compute the 50-day simple moving average.
                lambda series: series.rolling(50).mean()
            )

            factor_data["trend_gap"] = factor_data["sma_10"] / factor_data["sma_50"] - 1  # Convert the moving-average relation into a scale-free feature.

            factor_data[["date", "symbol", "close", "sma_10", "sma_50", "trend_gap"]].tail(12)
            """
        ),
        md(
            """
            ## Chapter 12. Estimate Rolling Volatility

            Volatility is not direction.
            It is variability.

            Highly volatile assets can be exciting, but they can also be risky and unstable.

            We estimate 20-day rolling volatility from one-day returns and annualize it using the market convention of 252 trading days:

            \\[
            \\sigma_{20} = \\text{std}(r_{t-19}, \\dots, r_t) \\times \\sqrt{252}
            \\]
            """
        ),
        code(
            """
            factor_data["vol_20"] = factor_data.groupby("symbol")["return_1d"].transform(  # Work within each symbol separately.
                lambda series: series.rolling(20).std() * np.sqrt(trading_days_per_year)  # Rolling standard deviation, then annualize.
            )

            factor_data[["date", "symbol", "return_1d", "vol_20"]].head(12)
            """
        ),
        md(
            """
            ## Chapter 13. Turn Raw Volume into a Relative Liquidity Signal

            Raw volume levels differ dramatically across assets.
            Comparing raw share counts directly is not very meaningful.

            So we:

            1. take the logarithm of volume,
            2. compare current log-volume to its recent rolling mean,
            3. scale that deviation by the rolling standard deviation.

            The result is a **z-score**:

            - positive values mean volume is unusually high,
            - negative values mean volume is unusually low.
            """
        ),
        code(
            """
            factor_data["log_volume"] = np.log(  # Take logs because trading volume is typically very skewed.
                factor_data["volume"].replace(0, np.nan)  # Replace zeros first so the logarithm is well-defined.
            )

            grouped_log_volume = factor_data.groupby("symbol")["log_volume"]  # Group log-volume by symbol for rolling statistics.

            rolling_log_volume_mean = grouped_log_volume.transform(  # Recent average level of log-volume.
                lambda series: series.rolling(20).mean()
            )

            rolling_log_volume_std = grouped_log_volume.transform(  # Recent variability of log-volume.
                lambda series: series.rolling(20).std()
            )

            factor_data["volume_z"] = (  # Standardized liquidity surprise.
                factor_data["log_volume"] - rolling_log_volume_mean
            ) / rolling_log_volume_std

            factor_data[["date", "symbol", "volume", "log_volume", "volume_z"]].head(12)
            """
        ),
        md(
            """
            ## Chapter 14. Compute RSI, a Popular Technical Indicator

            RSI stands for **Relative Strength Index**.

            Finance beginners usually encounter it as a way to summarize whether recent price changes have been more upward than downward.

            The usual interpretation is:

            - RSI above 70: maybe overbought,
            - RSI below 30: maybe oversold.

            Those rules are not laws of nature.
            They are heuristics.

            We calculate RSI from average gains and average losses over a 14-day window.
            """
        ),
        code(
            """
            factor_data["price_change"] = grouped_close.diff()  # Day-to-day dollar price change.

            factor_data["positive_change"] = factor_data["price_change"].clip(lower=0)  # Keep only gains; replace losses with zero.
            factor_data["negative_change"] = (-factor_data["price_change"].clip(upper=0))  # Keep only the magnitude of losses.

            factor_data["avg_gain_14"] = factor_data.groupby("symbol")["positive_change"].transform(  # Average gain over the last 14 days.
                lambda series: series.rolling(14).mean()
            )

            factor_data["avg_loss_14"] = factor_data.groupby("symbol")["negative_change"].transform(  # Average loss over the last 14 days.
                lambda series: series.rolling(14).mean()
            )

            relative_strength = factor_data["avg_gain_14"] / factor_data["avg_loss_14"].replace(0, np.nan)  # RS = average gain divided by average loss.

            factor_data["rsi_14"] = 100 - (100 / (1 + relative_strength))  # Convert relative strength into the standard RSI scale.

            factor_data[["date", "symbol", "price_change", "avg_gain_14", "avg_loss_14", "rsi_14"]].tail(12)
            """
        ),
        md(
            """
            ## Chapter 15. Define the Prediction Target

            A supervised model needs a target.

            Here the target is the **5-day forward return**:

            \\[
            \\text{forward return}_{5d} = \\frac{P_{t+5}}{P_t} - 1
            \\]

            This is not the same thing as tomorrow's return.
            We are asking the model to look at today's factors and estimate the next trading week's move.
            """
        ),
        code(
            """
            factor_data["forward_return_5d"] = grouped_close.shift(-5) / factor_data["close"] - 1  # Future five-day return from today's perspective.

            factor_data[["date", "symbol", "close", "forward_return_5d"]].head(12)
            """
        ),
        md(
            """
            ## Chapter 16. Keep a Clean Modeling Table

            Rolling features create missing values at the start of each symbol's history.
            The forward return target creates missing values at the end.

            That is normal.
            We now keep only the columns we need and drop rows that are incomplete.
            """
        ),
        code(
            """
            feature_columns = [  # These are the factors we will hand to the model.
                "return_5d",
                "return_21d",
                "trend_gap",
                "vol_20",
                "volume_z",
                "rsi_14",
            ]

            kept_columns = [  # These are the columns we want to retain for modeling and interpretation.
                "date",
                "symbol",
                "close",
                "volume",
                "return_1d",
                *feature_columns,
                "forward_return_5d",
            ]

            model_data = factor_data[kept_columns].replace([np.inf, -np.inf], np.nan)  # Replace infinite values before dropping missing rows.
            model_data = model_data.dropna().reset_index(drop=True)  # Keep only complete cases.

            print("Modeling table shape:", model_data.shape)
            model_data.head(10)
            """
        ),
        md(
            """
            # Part IV. Explore the Engineered Factors

            ## Chapter 17. Basic Summary Statistics

            Before fitting any model, ask:

            - Are the features on sensible ranges?
            - Are any columns suspiciously constant?
            - Do we see extreme tails?

            This is one of the simplest and most useful habits in quantitative work.
            """
        ),
        code(
            """
            model_data[feature_columns + ["forward_return_5d"]].describe().T
            """
        ),
        md(
            """
            ## Chapter 18. Inspect the Feature Distributions

            A model does not see “economic meaning.”
            It sees numbers.

            So it helps to understand the shape of those numbers:

            - symmetric or skewed,
            - narrow or wide,
            - centered near zero or not.
            """
        ),
        code(
            """
            plot_frame = model_data[feature_columns].copy()  # Copy the factors we want to visualize.
            plot_frame = plot_frame.sample(min(2500, len(plot_frame)), random_state=42)  # Sample rows so plots stay readable.
            long_plot_frame = plot_frame.melt(var_name="factor", value_name="value")  # Reshape to long format for faceted plotting.

            g = sns.FacetGrid(  # Build one histogram panel per factor.
                long_plot_frame,
                col="factor",
                col_wrap=3,
                sharex=False,
                sharey=False,
                height=4,
            )
            g.map_dataframe(sns.histplot, x="value", bins=40, color="steelblue")
            g.fig.suptitle("Distributions of the Engineered Factors", y=1.02)
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 19. Visualize Factor Correlations

            Correlation does not automatically imply redundancy.
            But highly correlated factors can make interpretation harder.

            This heatmap is descriptive, not causal.
            """
        ),
        code(
            """
            correlation_matrix = model_data[feature_columns + ["forward_return_5d"]].corr(numeric_only=True)  # Compute pairwise linear correlations.

            plt.figure(figsize=(9, 7))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                center=0,
            )
            plt.title("Correlation Heatmap for Factors and the 5-Day Forward Return")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 20. Plot a Few Feature-vs-Target Relationships

            These plots are useful for intuition.
            They are **not** proof of causality.

            A visible slope in an observational scatter plot can still be driven by:

            - omitted variables,
            - regime shifts,
            - selection effects,
            - or pure chance.
            """
        ),
        code(
            """
            scatter_sample = model_data.sample(min(1500, len(model_data)), random_state=7)  # Sample rows so the scatter is easier to see.

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            sns.regplot(
                data=scatter_sample,
                x="trend_gap",
                y="forward_return_5d",
                scatter_kws={"alpha": 0.20, "s": 15},
                line_kws={"color": "darkred"},
                ax=axes[0],
            )
            axes[0].set_title("Trend Gap vs 5-Day Forward Return")

            sns.regplot(
                data=scatter_sample,
                x="vol_20",
                y="forward_return_5d",
                scatter_kws={"alpha": 0.20, "s": 15},
                line_kws={"color": "darkred"},
                ax=axes[1],
            )
            axes[1].set_title("Volatility vs 5-Day Forward Return")

            sns.regplot(
                data=scatter_sample,
                x="rsi_14",
                y="forward_return_5d",
                scatter_kws={"alpha": 0.20, "s": 15},
                line_kws={"color": "darkred"},
                ax=axes[2],
            )
            axes[2].set_title("RSI vs 5-Day Forward Return")

            plt.tight_layout()
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 21. A Causality Interlude

            This is the right place to pause and say something very important.

            ### What we are doing

            We are using historical observational data to build a predictive trading model.

            ### What we are **not** doing

            We are **not** randomly assigning assets to different volatility levels, momentum states, or liquidity conditions.
            Markets are not laboratory mice.

            ### Why that matters

            If a factor appears important, there are several possibilities:

            - the factor may capture a genuine causal driver,
            - the factor may merely proxy for another hidden cause,
            - the factor may work only in one regime,
            - the factor may reflect data mining luck,
            - or the model may be exploiting a structural feature that later disappears.

            ### So why do the stress tests at all?

            Because scenario analysis is still useful.
            Even when it is not a clean causal identification strategy, it helps us ask:

            - Which signals is the strategy most sensitive to?
            - If volatility suddenly increases, does the system become fragile?
            - If momentum flips, what happens?

            That is valuable for decision-making, even without claiming causal proof.
            """
        ),
        code(
            """
            volatility_bucket_summary = (
                model_data.assign(  # Create a temporary bucketed version of the data.
                    vol_bucket=pd.qcut(model_data["vol_20"], q=5, duplicates="drop")  # Split volatility into quantile buckets.
                )
                .groupby("vol_bucket")["forward_return_5d"]  # For each volatility bucket...
                .mean()  # ...compute the average future return.
                .to_frame("mean_forward_return_5d")
            )

            trend_bucket_summary = (
                model_data.assign(
                    trend_bucket=pd.qcut(model_data["trend_gap"], q=5, duplicates="drop")
                )
                .groupby("trend_bucket")["forward_return_5d"]
                .mean()
                .to_frame("mean_forward_return_5d")
            )

            print("Average future return by volatility bucket:")
            display(volatility_bucket_summary)

            print("Average future return by trend-gap bucket:")
            display(trend_bucket_summary)
            """
        ),
        md(
            """
            # Part V. Train/Test Logic and Leakage Prevention

            ## Chapter 22. Why Time Splits Matter

            In finance, train/test splitting is different from many standard machine-learning examples.

            We do **not** want to randomly shuffle rows, because that would let information from the future leak into the past.

            Instead we use a temporal split:

            - train on dates before `2024-01-01`,
            - test on dates on or after `2024-01-01`.

            This better matches the real decision problem: learn from history, then act in a later period.
            """
        ),
        code(
            """
            train_df = model_data.loc[model_data["date"] < pd.Timestamp(test_start)].copy()  # Everything before the split is training data.
            test_df = model_data.loc[model_data["date"] >= pd.Timestamp(test_start)].copy()  # Everything from the split onward is test data.

            print("Train rows:", len(train_df))
            print("Test rows:", len(test_df))
            print("Train date range:", train_df["date"].min(), "->", train_df["date"].max())
            print("Test date range:", test_df["date"].min(), "->", test_df["date"].max())
            print("Unique train symbols:", train_df["symbol"].nunique())
            print("Unique test symbols:", test_df["symbol"].nunique())
            """
        ),
        md(
            """
            # Part VI. Plug the Factors into the Algorithmic Trading System

            ## Chapter 23. What the Trading System Does

            The reusable `AlgorithmicTradingSystem` follows a simple logic:

            1. fit a machine-learning model to predict 5-day forward returns from the factors,
            2. score each asset each day,
            3. rank the assets cross-sectionally on that day,
            4. go relatively long the stronger-ranked assets,
            5. go relatively short the weaker-ranked assets.

            The strategy is therefore:

            - **predictive**, not causal,
            - **cross-sectional**, because it compares assets to one another on the same day,
            - **rank-based**, because position decisions come from relative ranking rather than raw predicted values.
            """
        ),
        code(
            """
            strategy = AlgorithmicTradingSystem(  # Create the reusable strategy object.
                feature_columns=feature_columns,  # Tell the strategy which columns contain the factors.
                buy_threshold=0.55,  # Assets with rank percentiles above this threshold become long candidates.
                short_threshold=0.45,  # Assets with rank percentiles below this threshold become short candidates.
                max_leverage=1.0,  # Gross exposure is normalized to 1.0 each day.
            )

            strategy
            """
        ),
        code(
            """
            strategy.fit(train_df)  # Fit the random-forest model on the historical training sample.

            strategy.model.get_params()  # Inspect the model hyperparameters so the process is not a black box.
            """
        ),
        md(
            """
            ## Chapter 24. Convert Model Scores into Portfolio Returns

            This is a subtle but extremely important step.

            The model gives us an asset-level score.
            But the portfolio earns a **daily portfolio return**.

            That means we must:

            1. compute positions for each asset on each day,
            2. multiply each position by the next day's realized return,
            3. **sum** those asset-level contributions by date to get the actual strategy return.

            Why sum and not average?

            Because the asset-level `position` values are portfolio weights.
            The daily portfolio return is the weighted sum of constituent returns.
            """
        ),
        code(
            """
            def apply_strategy_to_frame(frame: pd.DataFrame, fitted_strategy: AlgorithmicTradingSystem) -> pd.DataFrame:
                \"\"\"Apply a fitted strategy to a feature table and keep the asset-level results.\"\"\"

                results = frame.copy()  # Start from a copy so the original data frame stays untouched.
                results["position"] = fitted_strategy.generate_positions(results)  # Compute long/short weights for every symbol-date row.
                results["next_day_return"] = results.groupby("symbol")["return_1d"].shift(-1)  # Realized return after today's signal is tomorrow's return.
                results["strategy_return"] = results["position"] * results["next_day_return"]  # Each row contributes weight times realized return.
                results["benchmark_return"] = results.groupby("date")["next_day_return"].transform("mean")  # Benchmark is the equal-weight average return that day.
                results = results.dropna(subset=["next_day_return"]).reset_index(drop=True)  # Drop rows where we cannot observe the next day.
                return results


            def build_daily_portfolio(asset_level_results: pd.DataFrame) -> pd.DataFrame:
                \"\"\"Aggregate asset-level contributions into daily portfolio results.\"\"\"

                daily = (
                    asset_level_results.groupby("date")  # Move from asset rows to one row per day.
                    .agg(
                        strategy_return=("strategy_return", "sum"),  # Sum weighted asset contributions to get the portfolio return.
                        benchmark_return=("benchmark_return", "mean"),  # Mean benchmark return because the benchmark is equal-weighted.
                    )
                    .reset_index()
                    .sort_values("date")
                )

                daily["strategy_equity"] = (1 + daily["strategy_return"]).cumprod()  # Grow $1 through time under the strategy.
                daily["benchmark_equity"] = (1 + daily["benchmark_return"]).cumprod()  # Grow $1 through time under the benchmark.
                daily["strategy_drawdown"] = daily["strategy_equity"] / daily["strategy_equity"].cummax() - 1  # Distance below the previous equity peak.
                daily["benchmark_drawdown"] = daily["benchmark_equity"] / daily["benchmark_equity"].cummax() - 1

                return daily
            """
        ),
        code(
            """
            train_results = apply_strategy_to_frame(train_df, strategy)  # Apply the fitted system to the training set.
            test_results = apply_strategy_to_frame(test_df, strategy)  # Apply the fitted system to the out-of-sample test set.

            train_results.head(10)
            """
        ),
        code(
            """
            test_results.head(10)
            """
        ),
        code(
            """
            train_daily = build_daily_portfolio(train_results)  # Aggregate the training asset-level contributions to daily returns.
            test_daily = build_daily_portfolio(test_results)  # Aggregate the test asset-level contributions to daily returns.

            test_daily.head(10)
            """
        ),
        md(
            """
            ## Chapter 25. Compute Standard Portfolio Metrics

            We will use the shared helper `evaluate_performance`, which computes:

            - annualized return,
            - annualized volatility,
            - Sharpe ratio,
            - maximum drawdown,
            - hit rate,
            - cumulative return.

            These metrics are based on the daily aggregated portfolio returns, not the raw asset-level rows.
            """
        ),
        code(
            """
            train_metrics = evaluate_performance(train_results, label="train")  # Summarize in-sample behavior.
            test_metrics = evaluate_performance(test_results, label="test")  # Summarize out-of-sample behavior.

            metrics = pd.DataFrame(  # Put both dictionaries into one tidy table.
                {
                    "train": train_metrics,
                    "test": test_metrics,
                }
            ).T

            metrics
            """
        ),
        md(
            """
            ## Chapter 26. Plot the Equity Curves

            The equity curve is often the most intuitive plot in a backtest.
            It answers a simple question:

            “If I started with $1, what would happen over time?”
            """
        ),
        code(
            """
            ax = test_daily.plot(
                x="date",
                y=["strategy_equity", "benchmark_equity"],
                figsize=(14, 6),
                title="Out-of-Sample Equity Curve: Strategy vs Benchmark",
            )
            ax.set_ylabel("Growth of $1")
            ax.set_xlabel("Date")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 27. Plot the Drawdowns

            A strategy can have a good long-run return and still be psychologically or economically difficult to hold.

            Drawdown tells us how painful the ride is between peaks.
            """
        ),
        code(
            """
            ax = test_daily.plot(
                x="date",
                y=["strategy_drawdown", "benchmark_drawdown"],
                figsize=(14, 6),
                title="Out-of-Sample Drawdown Comparison",
            )
            ax.set_ylabel("Drawdown")
            ax.set_xlabel("Date")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 28. Inspect the Worst Days

            Summary metrics are useful.
            But debugging often starts with the tails.

            Looking at the worst days helps us ask:

            - Did the strategy suffer during volatility spikes?
            - Did it fail in sector-specific shocks?
            - Were losses broad or concentrated?
            """
        ),
        code(
            """
            test_daily.nsmallest(10, "strategy_return")
            """
        ),
        md(
            """
            # Part VII. Which Factors Mattered Most?

            ## Chapter 29. Permutation Importance

            Permutation importance is a predictive importance tool.
            It asks:

            “If I scramble one factor so the model can no longer use its information, how much worse does prediction get?”

            This is still **not a causal effect estimate**.
            It is a measure of the model's reliance on that factor within this predictive setup.
            """
        ),
        code(
            """
            factor_importance = compute_factor_importance(  # Measure how much each factor matters to model prediction quality.
                strategy=strategy,
                train_df=train_df,
                feature_columns=feature_columns,
            )

            factor_importance
            """
        ),
        code(
            """
            plt.figure(figsize=(10, 5))
            sns.barplot(
                data=factor_importance,
                x="importance_mean",
                y="factor",
                hue="factor",
                palette="viridis",
                legend=False,
            )
            plt.title("Permutation Importance of the Trading Factors")
            plt.xlabel("Mean Importance")
            plt.ylabel("Factor")
            plt.show()
            """
        ),
        md(
            """
            # Part VIII. Counterfactual Stress Analysis

            ## Chapter 30. What We Mean by “Counterfactual” Here

            In strict causal inference, a counterfactual asks what would have happened under an alternative intervention.

            In a financial notebook like this, we usually do something more modest:

            - perturb the features in a way that resembles a market regime shift,
            - rerun the strategy logic,
            - measure how the portfolio outcome changes.

            This is **stress testing informed by counterfactual thinking**.
            It is useful, but it is not a clean identification design.
            """
        ),
        code(
            """
            scenarios = {  # Each scenario nudges one or more factors in a market-intuitive direction.
                "higher_volatility": {"vol_20": 1.35},  # Increase recent volatility by 35%.
                "momentum_crash": {  # Reverse recent momentum-style features.
                    "return_5d": -1.0,
                    "return_21d": -1.0,
                    "trend_gap": -0.75,
                },
                "liquidity_shock": {"volume_z": -2.0},  # Force a sharp negative liquidity surprise.
                "oversold_mean_reversion": {  # Push RSI lower and weaken short-horizon returns.
                    "rsi_14": -20.0,
                    "return_5d": -0.5,
                },
            }

            scenarios
            """
        ),
        code(
            """
            stress_rows = []  # We will collect one summary row per scenario.

            baseline_cumulative_return = test_metrics["cumulative_return"]  # We compare every scenario against the base-case strategy result.
            baseline_max_drawdown = test_metrics["max_drawdown"]  # We also compare drawdowns against the base case.

            for scenario_name, adjustments in scenarios.items():  # Loop through the scenario dictionary one scenario at a time.
                shocked_test_df = apply_counterfactual(  # Perturb the factor columns according to the scenario definition.
                    df=test_df,
                    adjustments=adjustments,
                    feature_columns=feature_columns,
                )

                shocked_results = apply_strategy_to_frame(  # Re-run the already-fitted strategy on the shocked factor table.
                    shocked_test_df,
                    strategy,
                )

                shocked_metrics = evaluate_performance(  # Summarize the shocked strategy outcome.
                    shocked_results,
                    label=scenario_name,
                )

                stress_rows.append(  # Store the scenario summary for later plotting and interpretation.
                    {
                        "scenario": scenario_name,
                        "annual_return": shocked_metrics["annual_return"],
                        "annual_volatility": shocked_metrics["annual_volatility"],
                        "sharpe_ratio": shocked_metrics["sharpe_ratio"],
                        "max_drawdown": shocked_metrics["max_drawdown"],
                        "cumulative_return": shocked_metrics["cumulative_return"],
                        "return_delta_vs_baseline": shocked_metrics["cumulative_return"] - baseline_cumulative_return,
                        "drawdown_delta_vs_baseline": shocked_metrics["max_drawdown"] - baseline_max_drawdown,
                    }
                )

            stress_summary = pd.DataFrame(stress_rows).sort_values("return_delta_vs_baseline").reset_index(drop=True)

            stress_summary
            """
        ),
        code(
            """
            plt.figure(figsize=(11, 5))
            sns.barplot(
                data=stress_summary,
                x="return_delta_vs_baseline",
                y="scenario",
                color="steelblue",
            )
            plt.axvline(0, color="black", linewidth=1)
            plt.title("Counterfactual Stress Scenarios: Change in Cumulative Return vs Baseline")
            plt.xlabel("Cumulative Return Delta vs Baseline")
            plt.ylabel("Scenario")
            plt.show()
            """
        ),
        md(
            """
            ## Chapter 31. How to Read the Stress Table

            If a scenario has:

            - a **negative** `return_delta_vs_baseline`, the strategy did worse than in the original test,
            - a **more negative** `max_drawdown`, the strategy suffered a deeper drawdown,
            - a much lower Sharpe ratio, risk-adjusted performance deteriorated.

            This does not prove the stressed factor “causes” the damage in a scientific sense.
            It shows that the strategy's behavior is sensitive to that modeled market condition.
            """
        ),
        md(
            """
            # Part IX. Re-Test on the Most Recent Window

            ## Chapter 32. Why Re-Test Recent Data Separately?

            Financial regimes change.

            A strategy can look fine over a long history and still be weak in the most recent period.
            By isolating the latest window, we get a rough answer to a practical question:

            “Is the strategy still behaving reasonably lately?”
            """
        ),
        code(
            """
            recent_window = summarize_recent_window(  # Collapse to the recent out-of-sample daily portfolio history.
                results=test_results,
                months=recent_months,
            )

            print("Recent window rows:", len(recent_window))
            recent_window.tail(10)
            """
        ),
        code(
            """
            ax = recent_window.plot(
                x="date",
                y=["strategy_equity", "benchmark_equity"],
                figsize=(14, 6),
                title=f"Recent {recent_months}-Month Re-Test",
            )
            ax.set_ylabel("Growth of $1")
            ax.set_xlabel("Date")
            plt.show()
            """
        ),
        code(
            """
            recent_summary = pd.Series(  # A compact summary for the most recent window.
                {
                    "recent_strategy_return": recent_window["strategy_equity"].iloc[-1] - 1,
                    "recent_benchmark_return": recent_window["benchmark_equity"].iloc[-1] - 1,
                    "recent_alpha": recent_window["strategy_equity"].iloc[-1] - recent_window["benchmark_equity"].iloc[-1],
                }
            )

            recent_summary
            """
        ),
        md(
            """
            # Part X. What We Learned, and What We Still Do Not Know

            ## Chapter 33. Main Lessons

            We built a complete research pipeline:

            1. downloaded historical market data,
            2. engineered interpretable trading factors,
            3. trained a ranking-based trading system,
            4. evaluated it on a later out-of-sample window,
            5. measured factor importance,
            6. stress-tested the strategy with counterfactual-style feature shocks,
            7. and re-evaluated behavior on the most recent months.

            ## Chapter 34. What This Notebook Does **Not** Prove

            This notebook does **not** prove:

            - that the factors are true causal drivers,
            - that the future will look like the past,
            - that the strategy will survive transaction costs, slippage, and market impact,
            - that the model is robust to universe changes,
            - or that observed predictive importance is the same as economic mechanism.

            ## Chapter 35. What a More Advanced Version Would Add

            A more professional research stack would likely add:

            - transaction costs and slippage,
            - walk-forward retraining,
            - exposure controls,
            - sector neutrality,
            - turnover analysis,
            - richer benchmarks,
            - and more careful causal identification if the goal were explanation rather than prediction.

            The important habit is this:
            be ambitious in experimentation, but humble in interpretation.
            """
        ),
        code(
            """
            debug_objects = {  # Keep a small index of the most useful notebook objects for later debugging.
                "raw_data": raw_data,
                "coverage_summary": coverage_summary,
                "factor_data": factor_data,
                "model_data": model_data,
                "train_df": train_df,
                "test_df": test_df,
                "train_results": train_results,
                "test_results": test_results,
                "train_daily": train_daily,
                "test_daily": test_daily,
                "factor_importance": factor_importance,
                "stress_summary": stress_summary,
                "recent_window": recent_window,
            }

            list(debug_objects.keys())
            """
        ),
    ]

    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13",
        },
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
