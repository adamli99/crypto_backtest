"""
Backtest engine for Martingale trading strategy
Extracted from the original backtest script for use in web service
"""

import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def timeframe_to_ms(tf: str) -> int:
    """Convert timeframe to milliseconds per candle"""
    tf = tf.lower()
    if 'm' in tf:  # Minutes
        minutes = int(tf.replace('m', ''))
        return minutes * 60 * 1000
    elif 'h' in tf:  # Hours
        hours = int(tf.replace('h', ''))
        return hours * 60 * 60 * 1000
    elif 'd' in tf:  # Days
        days = int(tf.replace('d', ''))
        return days * 24 * 60 * 60 * 1000
    elif 'w' in tf:  # Weeks
        weeks = int(tf.replace('w', ''))
        return weeks * 7 * 24 * 60 * 60 * 1000
    elif 'M' in tf:  # Months (approximated as 30 days)
        months = int(tf.replace('M', ''))
        return months * 30 * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported timeframe: {tf}")


def run_backtest(
    symbol: str = 'BTC/USDT',
    timeframe: str = '3m',
    backtest_start_date: str = '2025-12-01',
    backtest_end_date: str = '2026-01-21',
    single_bet_size: float = 20.0,
    martingale_multiples: float = 1.0,
    max_bets: int = 20,
    rsi_threshold: float = 40.0,
    profit_target: float = 1.5,
    martingale_price_drop: float = 0.85,
    stop_loss_percentage: float = 50.0,
    trading_fee_percentage: float = 0.08,
    ema_period: int = 11,
    sma_short_period: int = 5,
    sma_long_period: int = 10,
    lookback_candles: int = 300
) -> Dict:
    """
    Run a backtest with the specified parameters.
    
    Returns a dictionary containing all backtest results and metrics.
    """
    # Validate timeframe
    supported_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    if timeframe not in supported_timeframes:
        raise ValueError(f"Timeframe {timeframe} is not supported by Binance. Supported timeframes: {supported_timeframes}")

    # Initialize exchange with timeout and retry settings
    exchange = ccxt.binanceus({
        'enableRateLimit': True,
        'timeout': 30000,  # 30 seconds timeout
        'options': {
            'defaultType': 'spot',
        }
    })

    # Convert dates to timestamps
    backtest_start_ts = int(datetime.strptime(backtest_start_date, '%Y-%m-%d').timestamp() * 1000)
    backtest_end_ts = int(datetime.strptime(backtest_end_date, '%Y-%m-%d').timestamp() * 1000)

    # Calculate fetch start timestamp based on lookback_candles and timeframe
    candle_duration_ms = timeframe_to_ms(timeframe)
    fetch_start_ts = backtest_start_ts - (lookback_candles * candle_duration_ms)

    # Fetch historical OHLCV data at the specified timeframe
    all_ohlcv = []
    start_ts = fetch_start_ts
    max_iterations = 1000  # Prevent infinite loops
    iteration = 0
    
    try:
        while start_ts < backtest_end_ts and iteration < max_iterations:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=start_ts, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            if last_timestamp >= backtest_end_ts or last_timestamp <= start_ts:
                break
            start_ts = last_timestamp + 1
            iteration += 1
    except ccxt.NetworkError as e:
        raise ValueError(f"Network error while fetching data from Binance US: {str(e)}")
    except ccxt.ExchangeError as e:
        raise ValueError(f"Exchange error while fetching data from Binance US: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error fetching data: {str(e)}")

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Calculate RSI(14), EMA(11), SMA(5), and SMA(10)
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema_11'] = df['close'].ewm(span=ema_period, adjust=False).mean()
    df['sma_5'] = df['close'].rolling(window=sma_short_period).mean()
    df['sma_10'] = df['close'].rolling(window=sma_long_period).mean()

    # Filter data to backtest period
    df_backtest = df[(df.index >= pd.to_datetime(backtest_start_date)) & 
                     (df.index <= pd.to_datetime(backtest_end_date))].copy()

    # Check if backtest DataFrame is empty
    if df_backtest.empty:
        raise ValueError(f"No data available for the backtest period {backtest_start_date} to {backtest_end_date}.")

    # Calculate asset return (original)
    initial_price = df_backtest['close'].iloc[0]
    final_price = df_backtest['close'].iloc[-1]
    asset_percentage_return = (final_price - initial_price) / initial_price * 100

    # Initialize trading variables
    positions = []  # [entry_price, amount_btc]
    trades = []  # [timestamp, type, price, amount_btc, usdt_value, total_btc, total_usdt_cost, nav]
    total_btc = 0.0
    total_usdt_cost = 0.0
    nav_history = []  # [timestamp, nav]
    initial_capital = single_bet_size * max_bets
    cash = initial_capital
    cumulative_nav = initial_capital
    first_entry_price = 0.0
    profit_target_reached = False
    stop_loss_triggered = False
    cycle_start_time = None
    cycle_buy_count = 0
    cycles = []  # Store [start_time, end_time, duration, buy_count, cycle_profit_loss]

    # Trading logic
    for i in range(1, len(df_backtest)):
        low_price = df_backtest['low'].iloc[i]
        high_price = df_backtest['high'].iloc[i]
        close_price = df_backtest['close'].iloc[i]
        timestamp = df_backtest.index[i]
        rsi = df_backtest['rsi'].iloc[i]
        ema_11 = df_backtest['ema_11'].iloc[i]
        sma_5 = df_backtest['sma_5'].iloc[i]
        sma_10 = df_backtest['sma_10'].iloc[i]

        # Check for stop loss condition
        if len(positions) > 0 and first_entry_price > 0:
            price_drop_percentage = (first_entry_price - low_price) / first_entry_price * 100
            if price_drop_percentage >= stop_loss_percentage:
                usdt_received = total_btc * high_price
                fee = usdt_received * (trading_fee_percentage / 100)
                usdt_received -= fee
                cash += usdt_received
                cumulative_nav = cash
                trades.append([timestamp, 'STOP_LOSS_SELL', high_price, total_btc, usdt_received, 0.0, 0.0, cumulative_nav])
                nav_history.append([timestamp, cumulative_nav])
                # Record cycle duration, buy count, and profit/loss
                if cycle_start_time is not None:
                    cycle_duration = (timestamp - cycle_start_time).total_seconds() / 3600
                    cycle_trades = [trade for trade in trades if trade[0] >= cycle_start_time and trade[0] <= timestamp]
                    cycle_profit_loss = sum(trade[4] if trade[1] in ['SELL', 'STOP_LOSS_SELL'] else -trade[4] for trade in cycle_trades)
                    cycles.append([cycle_start_time, timestamp, cycle_duration, cycle_buy_count, cycle_profit_loss])
                    cycle_start_time = None
                    cycle_buy_count = 0
                positions = []
                total_btc = 0.0
                total_usdt_cost = 0.0
                first_entry_price = 0.0
                profit_target_reached = False
                stop_loss_triggered = True
                continue

        # Initial buy condition
        can_buy = False
        if stop_loss_triggered:
            if pd.notna(sma_5) and pd.notna(sma_10) and sma_5 > sma_10 and rsi < rsi_threshold:
                can_buy = True
        else:
            if rsi < rsi_threshold:
                can_buy = True

        if len(positions) == 0 and can_buy and cash >= single_bet_size:
            fee = single_bet_size * (trading_fee_percentage / 100)
            usdt_after_fee = single_bet_size - fee
            amount_btc = usdt_after_fee / low_price
            positions.append([low_price, amount_btc])
            total_btc += amount_btc
            total_usdt_cost += single_bet_size
            cash -= single_bet_size
            first_entry_price = low_price
            profit_target_reached = False
            stop_loss_triggered = False
            cycle_start_time = timestamp
            cycle_buy_count = 1
            cumulative_nav = cash + (total_btc * close_price)
            trades.append([timestamp, 'BUY', low_price, amount_btc, single_bet_size, total_btc, total_usdt_cost, cumulative_nav])
            nav_history.append([timestamp, cumulative_nav])

        # Martingale buy condition
        elif len(positions) > 0 and len(positions) < max_bets:
            price_drop = first_entry_price * (martingale_price_drop / 100)
            target_price = first_entry_price - (price_drop * len(positions))
            martingale_bet_size = single_bet_size * martingale_multiples
            if low_price <= target_price and cash >= martingale_bet_size:
                fee = martingale_bet_size * (trading_fee_percentage / 100)
                usdt_after_fee = martingale_bet_size - fee
                amount_btc = usdt_after_fee / low_price
                positions.append([low_price, amount_btc])
                total_btc += amount_btc
                total_usdt_cost += martingale_bet_size
                cash -= martingale_bet_size
                cycle_buy_count += 1
                cumulative_nav = cash + (total_btc * close_price)
                trades.append([timestamp, 'BUY', low_price, amount_btc, martingale_bet_size, total_btc, total_usdt_cost, cumulative_nav])
                nav_history.append([timestamp, cumulative_nav])

        # Sell condition
        if len(positions) > 0:
            current_value = total_btc * high_price
            profit_percentage = (current_value - total_usdt_cost) / total_usdt_cost * 100

            if profit_percentage >= profit_target:
                profit_target_reached = True

            if profit_target_reached and close_price < ema_11 and profit_percentage > 0:
                usdt_received = total_btc * high_price
                fee = usdt_received * (trading_fee_percentage / 100)
                usdt_received -= fee
                cash += usdt_received
                cumulative_nav = cash
                trades.append([timestamp, 'SELL', high_price, total_btc, usdt_received, 0.0, 0.0, cumulative_nav])
                nav_history.append([timestamp, cumulative_nav])
                # Record cycle duration, buy count, and profit/loss
                if cycle_start_time is not None:
                    cycle_duration = (timestamp - cycle_start_time).total_seconds() / 3600
                    cycle_trades = [trade for trade in trades if trade[0] >= cycle_start_time and trade[0] <= timestamp]
                    cycle_profit_loss = sum(trade[4] if trade[1] in ['SELL', 'STOP_LOSS_SELL'] else -trade[4] for trade in cycle_trades)
                    cycles.append([cycle_start_time, timestamp, cycle_duration, cycle_buy_count, cycle_profit_loss])
                    cycle_start_time = None
                    cycle_buy_count = 0
                positions = []
                total_btc = 0.0
                total_usdt_cost = 0.0
                first_entry_price = 0.0
                profit_target_reached = False
                stop_loss_triggered = False

        # Update NAV if no trade
        if len(trades) == 0 or trades[-1][0] != timestamp:
            cumulative_nav = cash + (total_btc * close_price) if total_btc > 0 else cash
            nav_history.append([timestamp, cumulative_nav])

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades, columns=['Timestamp', 'Type', 'Price', 'Amount_BTC', 'USDT_Value', 'Total_BTC', 'Total_USDT_Cost', 'NAV'])

    # Calculate total profit/loss
    final_nav = nav_history[-1][1]
    total_profit_loss = final_nav - initial_capital

    # Calculate strategy return
    strategy_percentage_return = (total_profit_loss / initial_capital) * 100

    # Calculate Maximum Drawdown
    nav_values = [nav[1] for nav in nav_history]
    peak = nav_values[0]
    max_drawdown_percentage = 0.0
    for nav in nav_values:
        if nav > peak:
            peak = nav
        drawdown = (peak - nav) / peak * 100
        if drawdown > max_drawdown_percentage:
            max_drawdown_percentage = drawdown

    # Calculate Sharpe Ratio
    nav_df = pd.DataFrame(nav_history, columns=['Timestamp', 'NAV'])
    nav_df.set_index('Timestamp', inplace=True)
    daily_nav = nav_df['NAV'].resample('D').last().ffill()
    daily_returns = daily_nav.pct_change().dropna()
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365) if std_daily_return != 0 else 0.0

    # Calculate number of trades
    num_buy_trades = len(trades_df[trades_df['Type'] == 'BUY'])
    num_sell_trades = len(trades_df[trades_df['Type'] == 'SELL'])
    num_stop_loss_trades = len(trades_df[trades_df['Type'] == 'STOP_LOSS_SELL'])

    # Find the longest Martingale cycle (by number of buys)
    longest_cycle = None
    if cycles:
        longest_cycle = max(cycles, key=lambda x: x[3])
        longest_cycle_start = longest_cycle[0]
        longest_cycle_end = longest_cycle[1]
        longest_cycle_duration = longest_cycle[2]
        longest_cycle_buy_count = longest_cycle[3]
    else:
        longest_cycle_start = None
        longest_cycle_end = None
        longest_cycle_duration = 0.0
        longest_cycle_buy_count = 0

    # Statistics for cycles with maximum buy counts
    max_buy_cycles = [cycle for cycle in cycles if cycle[3] == longest_cycle_buy_count] if longest_cycle_buy_count > 0 else []
    num_max_buy_cycles = len(max_buy_cycles)
    avg_max_buy_cycle_duration = np.mean([cycle[2] for cycle in max_buy_cycles]) if max_buy_cycles else 0.0
    total_max_buy_cycle_profit_loss = sum(cycle[4] for cycle in max_buy_cycles) if max_buy_cycles else 0.0

    # Calculate Return on Assets (ROA)
    if longest_cycle_buy_count > 0:
        roa_denominator = longest_cycle_buy_count * single_bet_size
        roa_percentage = (total_profit_loss / roa_denominator) * 100 if roa_denominator != 0 else 0.0
    else:
        roa_percentage = 0.0

    # Prepare result dictionary
    result = {
        'parameters': {
            'symbol': symbol,
            'timeframe': timeframe,
            'backtest_start_date': backtest_start_date,
            'backtest_end_date': backtest_end_date,
            'single_bet_size': single_bet_size,
            'martingale_multiples': martingale_multiples,
            'max_bets': max_bets,
            'rsi_threshold': rsi_threshold,
            'profit_target': profit_target,
            'martingale_price_drop': martingale_price_drop,
            'stop_loss_percentage': stop_loss_percentage,
            'trading_fee_percentage': trading_fee_percentage,
            'ema_period': ema_period,
            'sma_short_period': sma_short_period,
            'sma_long_period': sma_long_period,
        },
        'results': {
            'initial_capital': float(initial_capital),
            'final_nav': float(final_nav),
            'total_profit_loss': float(total_profit_loss),
            'strategy_percentage_return': float(strategy_percentage_return),
            'asset_percentage_return': float(asset_percentage_return),
            'return_on_assets': float(roa_percentage),
            'maximum_drawdown': float(max_drawdown_percentage),
            'sharpe_ratio': float(sharpe_ratio),
            'num_buy_trades': int(num_buy_trades),
            'num_sell_trades': int(num_sell_trades),
            'num_stop_loss_trades': int(num_stop_loss_trades),
        },
        'longest_cycle': {
            'start_time': str(longest_cycle_start) if longest_cycle_start else None,
            'end_time': str(longest_cycle_end) if longest_cycle_end else None,
            'duration_hours': float(longest_cycle_duration),
            'buy_count': int(longest_cycle_buy_count),
        },
        'max_buy_cycle_stats': {
            'num_cycles': int(num_max_buy_cycles),
            'avg_duration_hours': float(avg_max_buy_cycle_duration),
            'total_profit_loss': float(total_max_buy_cycle_profit_loss),
        },
        'trades': trades_df.to_dict('records'),
        'nav_history': [[str(nav[0]), float(nav[1])] for nav in nav_history],
        # Add data for plotting
        'plot_data': {
            'nav_history': nav_history,
            'trades_df': trades_df.to_dict('records'),
            'price_data': {
                'timestamps': [str(ts) for ts in df_backtest.index],
                'close_prices': df_backtest['close'].tolist(),
            },
            'initial_capital': float(initial_capital),
            'strategy_percentage_return': float(strategy_percentage_return),
            'asset_percentage_return': float(asset_percentage_return),
            'symbol': symbol,
            'timeframe': timeframe,
        }
    }

    return result
