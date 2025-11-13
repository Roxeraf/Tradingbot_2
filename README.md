# Cryptocurrency Trading Bot

A production-ready, modular cryptocurrency trading bot with support for multiple exchanges, customizable strategies, backtesting capabilities, and comprehensive risk management.

## 🚀 Features

- **Multi-Exchange Support**: Easy integration with multiple cryptocurrency exchanges via CCXT
- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Multiple Strategies**: Implements various trading strategies (MA Crossover, RSI, etc.)
- **Risk Management**: Sophisticated position sizing and portfolio risk management
- **Backtesting Engine**: Test strategies on historical data before live trading
- **Real-time Monitoring**: Comprehensive logging and performance tracking
- **Database Storage**: Persistent storage of trades, positions, and performance metrics
- **Paper Trading**: Test strategies without risking real capital
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

```bash
# Run bot
python src/main.py

# With Docker
docker-compose up -d
```

## 🏗️ Architecture

```
src/
├── config/              # Configuration management
├── exchanges/           # Exchange connectors
├── strategies/          # Trading strategies
├── indicators/          # Technical indicators
├── risk_management/     # Position sizing & portfolio management
├── data/               # Database models & storage
├── execution/          # Order management
├── monitoring/         # Logging system
└── main.py            # Main orchestrator
```

## 📊 Strategies

### Moving Average Crossover
- BUY: Fast MA crosses above slow MA
- SELL: Fast MA crosses below slow MA

### Custom Strategies
Create custom strategies by inheriting from `BaseStrategy` and implementing required methods.

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

- [x] Core trading engine
- [x] Risk management system
- [x] Database storage
- [x] Logging and monitoring
- [ ] Web dashboard (React + TypeScript)
- [ ] Advanced strategies (RSI, MACD, Bollinger Bands)
- [ ] Backtesting engine
- [ ] Real-time alerts (Telegram/Discord)

---

**Happy Trading! 📈**

*Remember: Never invest more than you can afford to lose.*
