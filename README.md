# Cryptocurrency Trading Bot 🤖📈

A **production-ready**, modular cryptocurrency trading bot with backtesting, REST API, real-time WebSocket updates, and a modern React dashboard. Built with Python, FastAPI, and React + TypeScript.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178c6.svg)](https://www.typescriptlang.org/)

---

## ✨ Features

### Core Trading Engine ✅
- ✅ **Multi-Exchange Support** - CCXT integration (Bitpanda, Binance, etc.)
- ✅ **Modular Architecture** - Clean separation of concerns, pluggable components
- ✅ **Multiple Strategies** - Moving Average Crossover (more coming soon)
- ✅ **15+ Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, etc.
- ✅ **Risk Management** - Position sizing, portfolio tracking, stop loss/take profit
- ✅ **Database Persistence** - SQLAlchemy with SQLite/PostgreSQL support
- ✅ **Comprehensive Logging** - Structured logging with loguru
- ✅ **Paper Trading** - Testnet/sandbox mode support

### Backtesting Engine ✅
- ✅ **Historical Testing** - Test strategies on historical data
- ✅ **Performance Metrics** - Sharpe ratio, Sortino ratio, max drawdown, win rate
- ✅ **Commission & Slippage** - Realistic simulation
- ✅ **Trade Analysis** - Detailed trade-by-trade breakdown
- ✅ **Equity Curve** - Visual performance tracking
- ✅ **CLI Tool** - Easy-to-use command-line interface

### REST API & WebSocket ✅
- ✅ **FastAPI Backend** - Modern, fast, auto-documented API
- ✅ **30+ REST Endpoints** - Trading, strategies, backtests, settings
- ✅ **Real-time WebSocket** - Live price updates, positions, orders, trades
- ✅ **Auto Documentation** - Swagger UI & ReDoc
- ✅ **Type Safety** - Pydantic models with validation
- ✅ **CORS Support** - Ready for frontend integration

### Web Dashboard ✅
- ✅ **React + TypeScript** - Modern, type-safe frontend
- ✅ **Material-UI Design** - Beautiful dark theme UI
- ✅ **Real-time Updates** - WebSocket integration
- ✅ **Responsive Design** - Works on desktop and tablet
- ✅ **Performance Dashboard** - Live metrics and charts
- ✅ **Multi-page Navigation** - Trading, Strategies, Backtesting, Settings, Logs

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#️-installation)
- [Configuration](#️-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Frontend Dashboard](#-frontend-dashboard)
- [Backtesting](#-backtesting)
- [Architecture](#️-architecture)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

Get up and running in 5 minutes:

### Option 1: Local Development (Recommended for Development)

```bash
# 1. Clone the repository
git clone <repository-url>
cd Tradingbot_2

# 2. Set up Python backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 4. Start the API backend
python src/api/main.py
# API available at: http://localhost:8000
# API docs at: http://localhost:8000/api/docs

# 5. In a new terminal, set up frontend
cd frontend
npm install
npm run dev
# Frontend available at: http://localhost:5173
```

### Option 2: Docker (Recommended for Production)

```bash
# 1. Clone and configure
git clone <repository-url>
cd Tradingbot_2
cp .env.example .env
# Edit .env with your settings

# 2. Start everything with Docker Compose
docker-compose up -d

# Access:
# - API: http://localhost:8000
# - Frontend: Build and serve from dist/
```

---

## 🛠️ Installation

### Prerequisites

**Backend:**
- Python 3.11 or higher
- pip or poetry

**Frontend:**
- Node.js 18+ and npm

**Optional:**
- Docker and Docker Compose
- PostgreSQL (for production)

### Backend Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import ccxt; print('CCXT:', ccxt.__version__)"
```

### Frontend Installation

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

---

## ⚙️ Configuration

Create a `.env` file in the root directory:

```bash
# Exchange Configuration
EXCHANGE_NAME=bitpanda
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
TESTNET=true  # ALWAYS start with testnet!

# Trading Configuration
TRADING_PAIRS=BTC/EUR,ETH/EUR
TIMEFRAME=1h
MAX_POSITION_SIZE=0.1  # 10% of portfolio per position
MAX_PORTFOLIO_RISK=0.02  # 2% risk per trade

# Strategy Configuration
STRATEGY_NAME=MovingAverageCrossover
STRATEGY_PARAMS={"fast_period": 20, "slow_period": 50, "min_confidence": 0.6}

# Risk Management
STOP_LOSS_PERCENTAGE=0.02  # 2% stop loss
TAKE_PROFIT_PERCENTAGE=0.04  # 4% take profit
TRAILING_STOP=false

# Database
DATABASE_URL=sqlite:///data/trading_bot.db
# For PostgreSQL: postgresql://user:password@localhost/trading_bot

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=your-secret-key-change-this-in-production

# Logging
LOG_LEVEL=INFO
LOG_TO_FILE=true
```

### Important Configuration Notes:

⚠️ **Security:**
- **NEVER commit your `.env` file to git**
- Always start with `TESTNET=true`
- Use read-only API keys when possible
- Keep `SECRET_KEY` secure and random

---

## 🎯 Usage

### 1. Start the Trading Bot

```bash
# Activate virtual environment
source venv/bin/activate

# Run the bot
python src/main.py
```

The bot will:
- Connect to the exchange
- Load the configured strategy
- Start monitoring markets
- Execute trades based on signals
- Log all activity

### 2. Start the API Backend

```bash
# Run FastAPI server
python src/api/main.py

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access:
- **API Base**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### 3. Start the Frontend Dashboard

```bash
cd frontend
npm run dev
```

Access: http://localhost:5173

### 4. Run a Backtest

```bash
# Basic backtest
python scripts/run_backtest.py \
    --strategy MovingAverageCrossover \
    --symbol BTC/EUR \
    --start 2023-01-01 \
    --end 2023-12-31

# With custom parameters
python scripts/run_backtest.py \
    --strategy MovingAverageCrossover \
    --symbol BTC/EUR \
    --timeframe 1h \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --capital 10000 \
    --fast-period 20 \
    --slow-period 50
```

---

## 📡 API Documentation

### Trading Endpoints

#### Get Bot Status
```bash
GET /api/trading/status
```

#### Start/Stop Bot
```bash
POST /api/trading/start
POST /api/trading/stop
```

#### Positions
```bash
GET /api/trading/positions              # Get all positions
POST /api/trading/positions/close       # Close position
POST /api/trading/positions/close-all   # Close all positions
PUT /api/trading/positions/update       # Update stop loss/take profit
```

#### Orders
```bash
GET /api/trading/orders                 # Get orders
POST /api/trading/orders                # Place order
DELETE /api/trading/orders/{order_id}   # Cancel order
```

#### Trading Data
```bash
GET /api/trading/balance                # Account balance
GET /api/trading/trades                 # Trade history
GET /api/trading/performance            # Performance metrics
GET /api/trading/performance/history    # Historical performance
```

### Strategy Endpoints

```bash
GET /api/strategies                     # List strategies
GET /api/strategies/available           # Available strategy types
POST /api/strategies                    # Create strategy
PUT /api/strategies/{id}                # Update strategy
POST /api/strategies/{id}/activate      # Activate strategy
POST /api/strategies/{id}/test          # Test strategy
GET /api/strategies/{id}/performance    # Strategy performance
```

### Backtest Endpoints

```bash
POST /api/backtest/run                  # Run backtest (async)
GET /api/backtest/status/{id}           # Check backtest status
GET /api/backtest/results               # List backtest results
GET /api/backtest/results/{id}          # Get detailed results
```

### Settings Endpoints

```bash
GET /api/settings                       # Get all settings
PUT /api/settings                       # Update settings
GET /api/settings/exchanges             # Available exchanges
GET /api/settings/pairs                 # Trading pairs
POST /api/settings/test-connection      # Test exchange connection
GET /api/settings/logs                  # System logs
```

### WebSocket

Connect to: `ws://localhost:8000/api/ws/live`

**Message Types:**
- `price_update` - Real-time price changes
- `position_update` - Position updates
- `order_update` - Order status changes
- `trade_execution` - Trade executions
- `portfolio_update` - Portfolio value updates
- `signal` - Trading signals
- `bot_status` - Bot status changes
- `log` - System logs

**Example JavaScript:**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/ws/live')

ws.onmessage = (event) => {
  const data = JSON.parse(event.data)
  console.log('Received:', data.type, data)
}
```

---

## 🖥️ Frontend Dashboard

### Pages

#### 1. Dashboard
- **Real-time Performance Metrics**
  - Total portfolio value
  - Realized & unrealized PnL
  - Win rate
  - Open positions count
  - Risk exposure
- **Bot Status** - Running/stopped indicator
- **Live Updates** - WebSocket-powered

#### 2. Trading (Coming Soon)
- Live positions with PnL
- Order book and recent trades
- Order management
- Trade history
- Interactive price charts

#### 3. Strategies (Coming Soon)
- Strategy list with performance
- Create/edit strategies
- Parameter configuration
- Activate/deactivate strategies
- Strategy comparison

#### 4. Backtesting (Coming Soon)
- Run backtests with custom parameters
- View detailed results
- Equity curve visualization
- Trade analysis
- Compare strategies

#### 5. Settings (Coming Soon)
- Exchange configuration
- Trading pairs selection
- Risk management settings
- API key management
- System preferences

#### 6. Logs (Coming Soon)
- Real-time log viewer
- Filter by level (DEBUG, INFO, WARNING, ERROR)
- Search functionality
- Export logs

### Frontend Tech Stack

- **React 18** - UI library
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Material-UI (MUI)** - Component library
- **React Router** - Navigation
- **React Query** - Data fetching & caching
- **Recharts** - Charts and visualizations
- **Axios** - HTTP client

---

## 🔬 Backtesting

### Features

The backtesting engine provides:

- ✅ **Realistic Simulation** - Includes commission and slippage
- ✅ **No Lookahead Bias** - Only uses data available at each point in time
- ✅ **Stop Loss/Take Profit** - Simulates automatic exits
- ✅ **Performance Metrics**:
  - Total return
  - Sharpe ratio (annualized)
  - Sortino ratio
  - Maximum drawdown & duration
  - Win rate
  - Profit factor
  - Average win/loss
  - Best/worst trade
- ✅ **Trade Analysis** - Complete trade history with entry/exit details
- ✅ **Equity Curve** - Track portfolio value over time

### Running Backtests

```bash
# Basic usage
python scripts/run_backtest.py \
    --strategy MovingAverageCrossover \
    --symbol BTC/EUR \
    --start 2023-01-01 \
    --end 2023-12-31

# Full options
python scripts/run_backtest.py \
    --strategy MovingAverageCrossover \
    --symbol BTC/EUR \
    --timeframe 1h \
    --start 2023-01-01 \
    --end 2023-12-31 \
    --capital 10000 \
    --fast-period 20 \
    --slow-period 50 \
    --no-save  # Don't save to database
```

### Example Output

```
================================================================================
BACKTEST RESULTS
================================================================================

📊 PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Initial Capital:    $10,000.00
Final Capital:      $12,500.00
Total Return:       +25.00%
Sharpe Ratio:       1.45
Sortino Ratio:      1.82
Max Drawdown:       -8.50%
Profit Factor:      2.15

📈 TRADING STATISTICS
--------------------------------------------------------------------------------
Total Trades:       45
Winning Trades:     30
Losing Trades:      15
Win Rate:           66.67%
Average Win:        $250.00
Average Loss:       $120.00
Best Trade:         $850.00
Worst Trade:        $-380.00
```

---

## 🏗️ Architecture

### Project Structure

```
Tradingbot_2/
├── src/
│   ├── api/                      # FastAPI REST API & WebSocket
│   │   ├── main.py              # API application
│   │   ├── models/              # Request/response models
│   │   └── routers/             # API endpoints
│   │       ├── trading.py       # Trading endpoints
│   │       ├── strategies.py    # Strategy management
│   │       ├── backtest.py      # Backtesting endpoints
│   │       ├── settings_router.py # Settings endpoints
│   │       └── websocket.py     # WebSocket handler
│   ├── backtesting/              # Backtesting engine
│   │   └── backtest_engine.py   # Core backtest logic
│   ├── config/                   # Configuration
│   │   └── settings.py          # Pydantic settings
│   ├── data/                     # Data layer
│   │   ├── models.py            # SQLAlchemy models
│   │   └── data_storage.py      # Database operations
│   ├── exchanges/                # Exchange connectors
│   │   ├── base_exchange.py     # Abstract base class
│   │   ├── bitpanda_exchange.py # Bitpanda implementation
│   │   └── exchange_factory.py  # Factory pattern
│   ├── execution/                # Order execution
│   │   └── order_manager.py     # Order management
│   ├── indicators/               # Technical indicators
│   │   └── technical_indicators.py # 15+ indicators
│   ├── monitoring/               # Logging & monitoring
│   │   └── logger.py            # Loguru setup
│   ├── risk_management/          # Risk management
│   │   ├── position_sizer.py    # Position sizing
│   │   └── portfolio_manager.py # Portfolio tracking
│   ├── strategies/               # Trading strategies
│   │   ├── base_strategy.py     # Abstract base class
│   │   ├── moving_average_strategy.py # MA Crossover
│   │   └── strategy_factory.py  # Factory pattern
│   ├── utils/                    # Utilities
│   └── main.py                   # Trading bot entry point
├── frontend/                     # React + TypeScript frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   │   └── Layout.tsx       # Main layout
│   │   ├── pages/               # Page components
│   │   │   ├── Dashboard.tsx    # Dashboard
│   │   │   ├── Trading.tsx      # Trading page
│   │   │   ├── Strategies.tsx   # Strategies page
│   │   │   ├── Backtesting.tsx  # Backtesting page
│   │   │   ├── Settings.tsx     # Settings page
│   │   │   └── Logs.tsx         # Logs page
│   │   ├── services/            # API services
│   │   │   └── api.ts           # API client
│   │   ├── hooks/               # Custom hooks
│   │   │   └── useWebSocket.ts  # WebSocket hook
│   │   ├── types/               # TypeScript types
│   │   ├── App.tsx              # Main app
│   │   └── main.tsx             # Entry point
│   ├── package.json
│   └── vite.config.ts
├── scripts/                      # Utility scripts
│   └── run_backtest.py          # Backtest runner
├── tests/                        # Test files
│   ├── unit/
│   └── integration/
├── data/                         # Data storage
│   ├── historical/              # Historical data
│   └── logs/                    # Log files
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Docker setup
├── Dockerfile                    # Docker image
├── .env.example                  # Environment template
└── README.md                     # This file
```

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     React Dashboard                          │
│                  (Real-time WebSocket)                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                           │
│              (REST API + WebSocket)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ↓               ↓               ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Trading    │  │  Backtesting │  │   Strategy   │
│    Engine    │  │    Engine    │  │   Manager    │
└──────┬───────┘  └──────────────┘  └──────┬───────┘
       │                                    │
       ↓                                    ↓
┌──────────────────────────────────────────────────┐
│              Exchange Layer (CCXT)                │
│         Bitpanda | Binance | Kraken | ...        │
└──────────────────────────────────────────────────┘
```

---

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_strategies.py
```

### Code Style

```bash
# Install dev dependencies
pip install black isort flake8

# Format code
black src/
isort src/

# Lint
flake8 src/
```

### Adding a New Strategy

1. Create a new file in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `calculate_indicators()` and `generate_signal()`
4. Register in `strategy_factory.py`

**Example:**

```python
# src/strategies/my_strategy.py
from src.strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Add your indicators
        df = data.copy()
        df['my_indicator'] = ...
        return df

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        df = self.calculate_indicators(data)

        # Your signal logic
        signal_type = SignalType.BUY  # or SELL or HOLD

        return TradingSignal(
            signal_type=signal_type,
            symbol=data.attrs['symbol'],
            confidence=0.8,
            entry_price=df['close'].iloc[-1],
            stop_loss=...,
            take_profit=...
        )
```

### Adding a New Exchange

1. Create a new file in `src/exchanges/`
2. Inherit from `BaseExchange`
3. Implement all required methods
4. Register in `exchange_factory.py`

---

## 🐳 Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f bot

# Stop services
docker-compose down
```

### Production

```yaml
# docker-compose.yml for production
version: '3.8'

services:
  bot:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    restart: unless-stopped
    environment:
      POSTGRES_DB: trading_bot
      POSTGRES_USER: bot_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./frontend/dist:/usr/share/nginx/html:ro

volumes:
  postgres_data:
```

---

## 🔒 Security Best Practices

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables
   - Enable IP whitelisting on exchange
   - Use read-only keys when possible

2. **Testing**
   - Always start with `TESTNET=true`
   - Test thoroughly before live trading
   - Start with small position sizes

3. **Monitoring**
   - Monitor logs regularly
   - Set up alerts for errors
   - Track performance metrics

4. **Updates**
   - Keep dependencies updated
   - Review security advisories
   - Test updates in testnet first

---

## 🗺️ Roadmap

### ✅ Completed (Phase 1 & 2)
- [x] Core trading engine
- [x] Risk management system
- [x] Database storage with SQLAlchemy
- [x] Comprehensive logging
- [x] Backtesting engine with metrics
- [x] FastAPI backend with 30+ endpoints
- [x] WebSocket real-time updates
- [x] React + TypeScript frontend
- [x] Dashboard with live performance metrics
- [x] Docker deployment setup

### 🚧 In Progress (Phase 3)
- [ ] Complete Trading page UI
- [ ] Strategy editor and configuration UI
- [ ] Backtest results visualization
- [ ] Comprehensive settings UI
- [ ] Logs viewer with filtering

### 📋 Planned (Phase 4+)
- [ ] Additional strategies (RSI, MACD, Bollinger Bands)
- [ ] Advanced charting (TradingView integration)
- [ ] Real-time alerts (Telegram/Discord)
- [ ] Portfolio optimization
- [ ] Machine learning strategies
- [ ] Multi-pair arbitrage
- [ ] Mobile responsive improvements
- [ ] Unit & integration tests
- [ ] CI/CD pipeline

---

## 📊 Performance

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 2GB
- Storage: 10GB

**Recommended:**
- CPU: 4+ cores
- RAM: 4GB+
- Storage: 20GB+ SSD

### Optimizations

- Async I/O for all exchange operations
- Efficient pandas vectorization
- Database connection pooling
- WebSocket for real-time updates (no polling)
- React Query caching

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 for Python code
- Use TypeScript for all frontend code
- Write docstrings for all functions
- Add type hints
- Include tests for new features
- Update documentation

---

## ⚠️ Disclaimer

**IMPORTANT: READ CAREFULLY**

This software is provided for **educational and research purposes only**.

- ❌ **Not Financial Advice** - This is not investment advice
- ❌ **No Warranty** - Provided "as is" without any guarantees
- ❌ **Trading Risk** - Cryptocurrency trading carries substantial risk
- ❌ **Potential Losses** - You may lose all your invested capital
- ❌ **Your Responsibility** - You are fully responsible for your trading decisions

**Only trade with money you can afford to lose.**

The authors and contributors are not responsible for any financial losses incurred while using this software.

---

## 📝 License

MIT License - see LICENSE file for details

---

## 📞 Support & Community

- **Issues**: [GitHub Issues](https://github.com/yourusername/Tradingbot_2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/Tradingbot_2/discussions)
- **Documentation**: [Wiki](https://github.com/yourusername/Tradingbot_2/wiki)

---

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://reactjs.org/) - UI library
- [Material-UI](https://mui.com/) - React component library
- [Loguru](https://github.com/Delgan/loguru) - Python logging library

---

## 📈 Stats

- **Total Lines of Code**: ~7,800
- **Python Files**: 40+
- **TypeScript Files**: 15+
- **API Endpoints**: 30+
- **Technical Indicators**: 15+
- **Test Coverage**: TBD

---

**Built with ❤️ for the crypto trading community**

**Happy Trading! 🚀📈**

*Remember: Past performance is not indicative of future results. Always do your own research.*
