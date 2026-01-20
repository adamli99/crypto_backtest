# Crypto Trading Backtest Web Service

A REST API service and web interface for running Martingale trading strategy backtests on cryptocurrency pairs using Binance US data.

## Features

- **Martingale Trading Strategy**: Implements a Martingale position sizing strategy with RSI-based entries and EMA-based exits
- **Comprehensive Metrics**: Returns detailed performance metrics including Sharpe ratio, maximum drawdown, and return on assets
- **Trade History**: Provides complete trade history with timestamps and details
- **NAV Tracking**: Tracks Net Asset Value over the entire backtest period
- **Cycle Analysis**: Analyzes Martingale cycles including longest cycle statistics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Service

### Development Server
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## API Endpoints

### POST /backtest

Run a backtest with specified parameters.

**Request Body:**
```json
{
  "symbol": "BTC/USDT",
  "timeframe": "3m",
  "backtest_start_date": "2025-12-01",
  "backtest_end_date": "2026-01-21",
  "single_bet_size": 20.0,
  "martingale_multiples": 1.0,
  "max_bets": 20,
  "rsi_threshold": 40.0,
  "profit_target": 1.5,
  "martingale_price_drop": 0.85,
  "stop_loss_percentage": 50.0,
  "trading_fee_percentage": 0.08,
  "ema_period": 11,
  "sma_short_period": 5,
  "sma_long_period": 10,
  "lookback_candles": 300
}
```

**Response:**
```json
{
  "success": true,
  "message": "Backtest completed successfully",
  "data": {
    "parameters": { ... },
    "results": {
      "initial_capital": 400.0,
      "final_nav": 421.55,
      "total_profit_loss": 21.55,
      "strategy_percentage_return": 5.39,
      "asset_percentage_return": -0.63,
      "return_on_assets": 10.77,
      "maximum_drawdown": 1.85,
      "sharpe_ratio": 4.71,
      "num_buy_trades": 88,
      "num_sell_trades": 23,
      "num_stop_loss_trades": 0
    },
    "longest_cycle": { ... },
    "max_buy_cycle_stats": { ... },
    "trades": [ ... ],
    "nav_history": [ ... ]
  }
}
```

### GET /backtest/default

Get default backtest parameters.

### GET /health

Health check endpoint.

### GET /

Root endpoint with API information.

## Example Usage

### Using curl:
```bash
curl -X POST "http://localhost:8000/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTC/USDT",
    "timeframe": "3m",
    "backtest_start_date": "2025-12-01",
    "backtest_end_date": "2026-01-21",
    "single_bet_size": 20.0,
    "max_bets": 20,
    "rsi_threshold": 40.0,
    "profit_target": 1.5
  }'
```

### Using Python:
```python
import requests

url = "http://localhost:8000/backtest"
data = {
    "symbol": "BTC/USDT",
    "timeframe": "3m",
    "backtest_start_date": "2025-12-01",
    "backtest_end_date": "2026-01-21",
    "single_bet_size": 20.0,
    "max_bets": 20,
    "rsi_threshold": 40.0,
    "profit_target": 1.5
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

## Parameters Description

- **symbol**: Trading pair (e.g., 'BTC/USDT')
- **timeframe**: Supported timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
- **backtest_start_date**: Start date in YYYY-MM-DD format
- **backtest_end_date**: End date in YYYY-MM-DD format
- **single_bet_size**: USDT amount per bet
- **martingale_multiples**: Multiplier for Martingale buys
- **max_bets**: Maximum number of positions in a cycle
- **rsi_threshold**: RSI(14) value for entry signal
- **profit_target**: Profit percentage to trigger sell consideration
- **martingale_price_drop**: Price drop percentage for additional buys
- **stop_loss_percentage**: Stop loss percentage from initial entry
- **trading_fee_percentage**: Trading fee as percentage (0.08% = 0.08)
- **ema_period**: Period for exponential moving average
- **sma_short_period**: Short-term SMA period
- **sma_long_period**: Long-term SMA period
- **lookback_candles**: Number of candles before start date for indicator calculation

## Strategy Overview

The backtest implements a Martingale trading strategy:

1. **Entry**: Buy when RSI(14) < threshold (or SMA5 > SMA10 and RSI < threshold after stop loss)
2. **Martingale Buys**: Add positions when price drops by martingale_price_drop% increments
3. **Exit**: Sell when profit target is reached AND price breaks below EMA
4. **Stop Loss**: Exit all positions if price drops by stop_loss_percentage from first entry

## License

This project is provided as-is for educational and research purposes.
