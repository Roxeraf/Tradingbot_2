# Cryptocurrency Trading Bot

A production-ready, modular cryptocurrency trading bot with support for multiple exchanges, customizable strategies, backtesting capabilities, and comprehensive risk management.

## 🚀 Features

- **Multi-Exchange Support**: Binance, Coinbase, Kraken, and Bitpanda integration via CCXT
- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Advanced Trading Strategies**: MA Crossover, RSI, MACD, and Bollinger Bands
- **Comprehensive Backtesting Engine**: Test and optimize strategies on historical data
- **Risk Management**: Sophisticated position sizing and portfolio risk management
- **REST API**: Complete FastAPI backend for bot control and monitoring
- **Web Dashboard**: Modern React + TypeScript UI for real-time monitoring
- **Real-time Monitoring**: Comprehensive logging and performance tracking
- **Database Storage**: Persistent storage of trades, positions, and performance metrics
- **Paper Trading**: Test strategies without risking real capital on testnet/sandbox
- **Performance Analytics**: Sharpe ratio, Sortino ratio, drawdown analysis, and more
- **Configurable**: Easy configuration via environment variables

## 📋 Installation

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional)

### Local Installation

```bash
# Clone repository
git clone <repository-url>
cd Tradingbot_2

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### Docker Installation

```bash
docker-compose up -d
```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```bash
EXCHANGE_NAME=bitpanda
API_KEY=your_api_key
API_SECRET=your_api_secret
TESTNET=true
TRADING_PAIRS=BTC/EUR,ETH/EUR
STRATEGY_NAME=MovingAverageCrossover
```

## 🚀 Usage

### Run the Trading Bot

```bash
# Run bot
python src/main.py

# With Docker
docker-compose up -d
```

### Run the API Server

```bash
# Start the FastAPI server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc

### Run the Web Dashboard

```bash
cd frontend
npm install
npm run dev
```

The dashboard will be available at http://localhost:3000

### Run Backtests

```python
from src.backtesting.runner import BacktestRunner
from src.strategies.strategy_factory import StrategyFactory
import pandas as pd

# Load historical data
data = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)

# Create strategy
strategy = StrategyFactory.create('rsi', {'rsi_period': 14})

# Run backtest
results = BacktestRunner.run_backtest(
    data=data,
    strategy=strategy,
    symbol='BTC/USD',
    initial_capital=10000,
    print_report=True
)
```

## 🏗️ Architecture

```
Tradingbot_2/
├── src/
│   ├── config/              # Configuration management
│   ├── exchanges/           # Exchange adapters (Binance, Coinbase, Kraken, Bitpanda)
│   ├── strategies/          # Trading strategies (MA, RSI, MACD, Bollinger Bands)
│   ├── backtesting/         # Backtesting engine and performance metrics
│   ├── indicators/          # Technical indicators
│   ├── risk_management/     # Position sizing & portfolio management
│   ├── data/               # Database models & storage
│   ├── execution/          # Order management
│   ├── monitoring/         # Logging system
│   ├── api/                # FastAPI REST endpoints
│   │   ├── routers/        # API route handlers
│   │   └── models/         # Request/response models
│   └── main.py            # Main orchestrator
│
├── frontend/               # React + TypeScript web dashboard
│   ├── src/
│   │   ├── api/           # API client
│   │   ├── components/    # React components
│   │   ├── pages/         # Dashboard pages
│   │   └── types/         # TypeScript types
│   └── package.json
│
├── tests/                  # Test suite
├── docker-compose.yml      # Docker orchestration
└── requirements.txt        # Python dependencies
```

## 📊 Trading Strategies

### 1. Moving Average Crossover
- **BUY**: Fast MA crosses above slow MA (bullish crossover)
- **SELL**: Fast MA crosses below slow MA (bearish crossover)
- Includes volume confirmation and configurable periods

### 2. RSI (Relative Strength Index)
- **BUY**: RSI crosses above oversold threshold (default: 30)
- **SELL**: RSI crosses below overbought threshold (default: 70)
- Supports divergence detection and momentum confirmation

### 3. MACD (Moving Average Convergence Divergence)
- **BUY**: MACD line crosses above signal line
- **SELL**: MACD line crosses below signal line
- Includes histogram analysis and zero-line crossovers

### 4. Bollinger Bands
- **BUY**: Price bounces off lower band with reversal signal
- **SELL**: Price reverses at upper band
- Supports squeeze detection and mean reversion trading

### Custom Strategies
Create custom strategies by inheriting from `BaseStrategy` and implementing:
- `calculate_indicators()` - Add technical indicators to data
- `generate_signal()` - Generate trading signals
- `get_required_history()` - Specify minimum data requirements

## 🛡️ Risk Management

- Position sizing based on risk percentage
- Portfolio-wide risk limits
- Automatic stop loss/take profit
- Real-time position monitoring

## 🧪 Testing

```bash
pytest
pytest --cov=src tests/
```

## 🚀 Deployment

See deployment guide in documentation for:
- Docker deployment
- VPS setup (Hetzner)
- Production checklist
- Security best practices

## ⚠️ Disclaimer

**This software is for educational purposes only. Trading cryptocurrencies carries significant risk. Only trade with capital you can afford to lose. The authors are not responsible for any financial losses.**

## 📝 License

MIT License

## 🗺️ Roadmap

### Completed ✅
- [x] Core trading engine
- [x] Risk management system
- [x] Database storage
- [x] Logging and monitoring
- [x] **Web dashboard (React + TypeScript)**
- [x] **Advanced strategies (RSI, MACD, Bollinger Bands)**
- [x] **Backtesting engine with performance metrics**
- [x] **Multi-exchange support (Binance, Coinbase, Kraken)**
- [x] **REST API for bot control**

### In Progress 🚧
- [ ] Real-time alerts (Telegram/Discord)
- [ ] Strategy optimization algorithms
- [ ] Advanced portfolio management
- [ ] Machine learning integration

---

**Happy Trading! 📈**

*Remember: Never invest more than you can afford to lose.*
