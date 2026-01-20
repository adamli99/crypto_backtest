"""
FastAPI web service for crypto trading backtest
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
from backtest_engine import run_backtest
import traceback
import ccxt
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64

app = FastAPI(
    title="Crypto Trading Backtest API",
    description="REST API for running Martingale trading strategy backtests",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class BacktestRequest(BaseModel):
    """Request model for backtest endpoint"""
    symbol: str = Field(default='BTC/USDT', description="Trading pair (e.g., 'BTC/USDT')")
    timeframe: str = Field(default='3m', description="Timeframe (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)")
    backtest_start_date: str = Field(default='2025-12-01', description="Start date (YYYY-MM-DD)")
    backtest_end_date: str = Field(default='2026-01-21', description="End date (YYYY-MM-DD)")
    single_bet_size: float = Field(default=20.0, description="USDT per bet")
    martingale_multiples: float = Field(default=1.0, description="Multiples of single_bet_size for Martingale buys")
    max_bets: int = Field(default=20, description="Maximum number of bets")
    rsi_threshold: float = Field(default=40.0, description="RSI(14) threshold for initial entry")
    profit_target: float = Field(default=1.5, description="Profit target percentage")
    martingale_price_drop: float = Field(default=0.85, description="Percentage drop for Martingale buys")
    stop_loss_percentage: float = Field(default=50.0, description="Stop loss as percentage drop")
    trading_fee_percentage: float = Field(default=0.08, description="Trading fee percentage")
    ema_period: int = Field(default=11, description="Period for EMA")
    sma_short_period: int = Field(default=5, description="Period for short-term SMA")
    sma_long_period: int = Field(default=10, description="Period for long-term SMA")
    lookback_candles: int = Field(default=300, description="Number of candles prior to start for indicators")


class BacktestResponse(BaseModel):
    """Response model for backtest endpoint"""
    success: bool
    message: str
    data: Optional[Dict] = None
    error: Optional[str] = None


@app.get("/")
async def root():
    """Serve the web interface"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/test-connection")
async def test_connection():
    """Test connection to Binance US exchange"""
    try:
        exchange = ccxt.binanceus({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
            }
        })
        # Try to fetch ticker to test connection
        ticker = exchange.fetch_ticker('BTC/USDT')
        return {
            "status": "success",
            "message": "Successfully connected to Binance US",
            "exchange": "binanceus",
            "test_symbol": "BTC/USDT",
            "current_price": ticker['last']
        }
    except ccxt.NetworkError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Network error: Unable to connect to Binance US. Please check your internet connection. Error: {str(e)}"
        )
    except ccxt.ExchangeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Exchange error: Binance US may be temporarily unavailable. Error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error testing connection: {str(e)}"
        )


@app.post("/backtest/plot")
async def generate_plot(request: BacktestRequest):
    """Generate and return the backtest plots as an image"""
    try:
        # Run the backtest
        result = run_backtest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            backtest_start_date=request.backtest_start_date,
            backtest_end_date=request.backtest_end_date,
            single_bet_size=request.single_bet_size,
            martingale_multiples=request.martingale_multiples,
            max_bets=request.max_bets,
            rsi_threshold=request.rsi_threshold,
            profit_target=request.profit_target,
            martingale_price_drop=request.martingale_price_drop,
            stop_loss_percentage=request.stop_loss_percentage,
            trading_fee_percentage=request.trading_fee_percentage,
            ema_period=request.ema_period,
            sma_short_period=request.sma_short_period,
            sma_long_period=request.sma_long_period,
            lookback_candles=request.lookback_candles
        )
        
        plot_data = result['plot_data']
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
        
        # Plot NAV in the first subplot
        nav_df = pd.DataFrame(plot_data['nav_history'], columns=['Timestamp', 'NAV'])
        nav_df['Timestamp'] = pd.to_datetime(nav_df['Timestamp'])
        ax1.plot(nav_df['Timestamp'], nav_df['NAV'], label='Net Asset Value (USDT)', color='blue')
        ax1.axhline(y=plot_data['initial_capital'], color='r', linestyle='--', label='Initial Capital')
        ax1.set_title(f"Net Asset Value Over Time ({plot_data['symbol']})")
        ax1.set_ylabel('NAV (USDT)')
        ax1.legend()
        ax1.grid(True)
        ax1.text(0.5, 0.98, f"Percentage of Return (Strategy): {plot_data['strategy_percentage_return']:.2f}%", 
                 transform=ax1.transAxes, fontsize=10, horizontalalignment='center', verticalalignment='top')
        
        # Plot BTC/USDT price in the second subplot
        price_df = pd.DataFrame({
            'timestamp': pd.to_datetime(plot_data['price_data']['timestamps']),
            'close': plot_data['price_data']['close_prices']
        })
        ax2.plot(price_df['timestamp'], price_df['close'], label=f"Close Price ({plot_data['timeframe']})", 
                color='green', linewidth=1)
        ax2.set_title(f"{plot_data['symbol']} Price ({plot_data['timeframe']} Timeframe)")
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USDT)')
        
        # Add annotations for BUY, SELL, and STOP_LOSS_SELL trades
        trades_df = pd.DataFrame(plot_data['trades_df'])
        if not trades_df.empty:
            trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'])
            buy_trades = trades_df[trades_df['Type'] == 'BUY']
            sell_trades = trades_df[trades_df['Type'] == 'SELL']
            stop_loss_trades = trades_df[trades_df['Type'] == 'STOP_LOSS_SELL']
            
            if not buy_trades.empty:
                ax2.scatter(buy_trades['Timestamp'], buy_trades['Price'], color='blue', marker='^', 
                           zorder=5, label='Buy Entry', s=50)
            if not sell_trades.empty:
                ax2.scatter(sell_trades['Timestamp'], sell_trades['Price'], color='red', marker='v', 
                           zorder=5, label='Sell Exit', s=50)
            if not stop_loss_trades.empty:
                ax2.scatter(stop_loss_trades['Timestamp'], stop_loss_trades['Price'], color='purple', 
                           marker='x', zorder=5, label='Stop Loss Exit', s=50)
        
        ax2.legend()
        ax2.grid(True)
        ax2.text(0.5, 0.98, f"Percentage of Return (Asset, Original): {plot_data['asset_percentage_return']:.2f}%", 
                 transform=ax2.transAxes, fontsize=10, horizontalalignment='center', verticalalignment='top')
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return {"plot_image": f"data:image/png;base64,{img_base64}"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except ccxt.NetworkError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Network error connecting to Binance US. Please check your internet connection and try again. Error: {str(e)}"
        )
    except ccxt.ExchangeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Binance US exchange error. The exchange may be temporarily unavailable. Error: {str(e)}"
        )
    except Exception as e:
        error_detail = f"Error generating plot: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest_api(request: BacktestRequest):
    """
    Run a backtest with the specified parameters.
    
    This endpoint executes a Martingale trading strategy backtest using the provided
    parameters and returns comprehensive results including performance metrics,
    trade history, and cycle statistics.
    """
    try:
        # Run the backtest
        result = run_backtest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            backtest_start_date=request.backtest_start_date,
            backtest_end_date=request.backtest_end_date,
            single_bet_size=request.single_bet_size,
            martingale_multiples=request.martingale_multiples,
            max_bets=request.max_bets,
            rsi_threshold=request.rsi_threshold,
            profit_target=request.profit_target,
            martingale_price_drop=request.martingale_price_drop,
            stop_loss_percentage=request.stop_loss_percentage,
            trading_fee_percentage=request.trading_fee_percentage,
            ema_period=request.ema_period,
            sma_short_period=request.sma_short_period,
            sma_long_period=request.sma_long_period,
            lookback_candles=request.lookback_candles
        )
        
        return BacktestResponse(
            success=True,
            message="Backtest completed successfully",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except ccxt.NetworkError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Network error connecting to Binance US. Please check your internet connection and try again. Error: {str(e)}"
        )
    except ccxt.ExchangeError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Binance US exchange error. The exchange may be temporarily unavailable. Error: {str(e)}"
        )
    except Exception as e:
        error_detail = f"Error running backtest: {str(e)}"
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/backtest/default")
async def get_default_parameters():
    """Get default backtest parameters"""
    return {
        "default_parameters": {
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
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
