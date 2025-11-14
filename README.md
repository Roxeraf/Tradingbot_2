# Cryptocurrency Trading Bot

A production-ready, modular cryptocurrency trading bot with support for multiple exchanges, customizable strategies, backtesting capabilities, and comprehensive risk management.

## 🚀 Features

### Core Trading
- **Multi-Exchange Support**: Binance, Coinbase, Kraken, and Bitpanda integration via CCXT
- **Modular Architecture**: Clean separation of concerns with pluggable components
- **Advanced Trading Strategies**: MA Crossover, RSI, MACD, Bollinger Bands, and ML-based
- **Risk Management**: Sophisticated position sizing and portfolio risk management
- **Paper Trading**: Test strategies without risking real capital on testnet/sandbox

### Strategy Optimization ⭐ NEW
- **Grid Search**: Exhaustive parameter optimization
- **Random Search**: Efficient random parameter sampling
- **Bayesian Optimization**: Smart parameter exploration with Gaussian Processes
- **Walk-Forward Optimization**: Out-of-sample validation to prevent overfitting

### Portfolio Management ⭐ NEW
- **8 Allocation Strategies**: Equal Weight, Risk Parity, Mean-Variance, Min Variance, Max Sharpe, Max Diversification, HRP
- **Automatic Rebalancing**: Periodic, threshold-based, tolerance band, and volatility-based rebalancing
- **Multi-Asset Support**: Manage portfolios across multiple cryptocurrencies

### Machine Learning ⭐ NEW
- **Feature Engineering**: 100+ technical, statistical, and time-based features
- **ML Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, LSTM
- **ML Strategy**: Trade using machine learning predictions with confidence thresholds
- **Model Training Pipeline**: Complete end-to-end ML training and evaluation

### Analytics & Monitoring
- **Comprehensive Backtesting**: Test strategies on historical data with detailed metrics
- **Performance Analytics**: Sharpe ratio, Sortino ratio, drawdown analysis, win rate, and more
- **Real-time Monitoring**: Comprehensive logging and performance tracking
- **Database Storage**: Persistent storage of trades, positions, and performance metrics

### API & UI
- **REST API**: Complete FastAPI backend with 30+ endpoints
- **Web Dashboard**: Modern React + TypeScript UI for real-time monitoring
- **API Documentation**: Interactive Swagger/OpenAPI docs at `/docs`
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

## 🔬 Strategy Optimization

The bot now includes advanced strategy optimization algorithms to find optimal parameters:

### Grid Search
Exhaustive search over parameter combinations:
```python
from src.optimization.strategy_optimizer import StrategyOptimizer, OptimizationMethod

optimizer = StrategyOptimizer(
    data=historical_data,
    strategy_name='ma_crossover',
    symbol='BTC/USD',
    optimization_metric='sharpe_ratio'
)

param_grid = {
    'fast_period': [10, 20, 30],
    'slow_period': [40, 50, 60]
}

results = optimizer.grid_search(param_grid)
print(f"Best params: {results['best_params']}")
```

### Random Search
Efficient random sampling from parameter distributions:
```python
param_distributions = {
    'fast_period': (5, 30, 'int'),
    'slow_period': (30, 100, 'int'),
    'threshold': (0.6, 0.9)
}

results = optimizer.random_search(param_distributions, n_iterations=50)
```

### Bayesian Optimization
Smart parameter exploration using Gaussian Processes:
```python
param_space = {
    'fast_period': (5, 50, 'int'),
    'slow_period': (50, 200, 'int'),
    'threshold': (0.5, 0.9, 'float')
}

results = optimizer.bayesian_optimization(param_space, n_iterations=50)
```

### Walk-Forward Optimization
Validates strategy robustness by testing on out-of-sample periods:
```python
from src.optimization.walk_forward import WalkForwardOptimizer

wf_optimizer = WalkForwardOptimizer(
    data=historical_data,
    strategy_name='ma_crossover',
    symbol='BTC/USD'
)

results = wf_optimizer.optimize(
    param_space=param_space,
    train_period_days=180,
    test_period_days=60,
    optimization_method=OptimizationMethod.RANDOM_SEARCH
)

print(f"Average in-sample Sharpe: {results['summary']['avg_train_metric']}")
print(f"Average out-of-sample Sharpe: {results['summary']['avg_test_metric']}")
```

## 💼 Advanced Portfolio Management

### Portfolio Allocation Strategies

The bot supports 8 sophisticated allocation methods:

1. **Equal Weight** - Simple equal allocation
2. **Market Cap Weight** - Weighted by market capitalization
3. **Risk Parity** - Equal risk contribution from each asset
4. **Mean-Variance** - Markowitz portfolio optimization
5. **Minimum Variance** - Lowest portfolio volatility
6. **Maximum Sharpe** - Highest risk-adjusted returns
7. **Maximum Diversification** - Optimal diversification ratio
8. **Hierarchical Risk Parity** - HRP using hierarchical clustering

```python
from src.portfolio.allocator import PortfolioAllocator, AllocationStrategy

allocator = PortfolioAllocator()

# Calculate risk parity allocation
weights = allocator.allocate(
    symbols=['BTC/USD', 'ETH/USD', 'SOL/USD'],
    strategy=AllocationStrategy.RISK_PARITY,
    returns=historical_returns
)

print(f"Optimal weights: {weights}")
```

### Automatic Portfolio Rebalancing

Multiple rebalancing strategies to maintain target allocation:

```python
from src.portfolio.rebalancer import PortfolioRebalancer, RebalancingStrategy

rebalancer = PortfolioRebalancer(
    target_weights={'BTC/USD': 0.5, 'ETH/USD': 0.5},
    rebalancing_cost=0.001
)

# Threshold-based rebalancing
trades = rebalancer.rebalance(
    current_date=datetime.now(),
    current_positions=current_positions,
    current_prices=current_prices,
    strategy=RebalancingStrategy.THRESHOLD,
    threshold=0.05  # Rebalance if drift > 5%
)

if trades:
    print(f"Rebalancing needed: {trades}")
```

## 🤖 Machine Learning Integration

### Feature Engineering

Automatically creates 100+ features from OHLCV data:

```python
from src.ml.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Create all features
features = engineer.create_all_features(
    data=ohlcv_data,
    include_price_features=True,
    include_technical_features=True,
    include_statistical_features=True,
    include_time_features=True
)

# Create target variable
features = engineer.create_target_variable(
    features,
    target_type='direction',  # or 'returns', 'classification'
    forward_periods=1
)
```

### ML Model Training

Train machine learning models for price prediction:

```python
from src.ml.model_trainer import MLModelTrainer, ModelType

# Train Random Forest
trainer = MLModelTrainer(
    model_type=ModelType.RANDOM_FOREST,
    task='classification'
)

X_train, X_test, y_train, y_test = trainer.prepare_data(
    data=features,
    feature_columns=feature_columns,
    target_column='target',
    train_size=0.8
)

trainer.train(X_train, y_train, n_estimators=100, max_depth=10)
metrics = trainer.evaluate(X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1']:.2f}")

# Save model
trainer.save_model('models/btc_predictor.pkl')
```

### ML-Based Trading Strategy

Use trained models for trading decisions:

```python
from src.ml.ml_strategy import MLStrategy

# Initialize ML strategy
ml_strategy = MLStrategy({
    'model_path': 'models/btc_predictor.pkl',
    'model_type': 'random_forest',
    'confidence_threshold': 0.7,
    'stop_loss_pct': 0.02,
    'take_profit_pct': 0.04
})

# Generate signal
signal = ml_strategy.generate_signal(recent_data)

if signal.is_actionable():
    print(f"Signal: {signal.signal_type.value}")
    print(f"Confidence: {signal.confidence:.2%}")
    print(f"Entry: ${signal.entry_price:.2f}")
```

### Supported ML Models

- **Random Forest** - Robust ensemble method
- **XGBoost** - High-performance gradient boosting
- **LightGBM** - Fast and memory-efficient
- **Gradient Boosting** - Reliable baseline
- **LSTM** - Deep learning for temporal patterns

## 🌐 API Endpoints

The REST API has been expanded with new endpoints:

### Strategy Optimization
- `POST /optimization/optimize` - Run optimization
- `POST /optimization/walk-forward` - Walk-forward optimization
- `GET /optimization/methods` - Available methods
- `GET /optimization/metrics` - Available metrics

### Portfolio Management
- `POST /portfolio/allocate` - Calculate allocation
- `POST /portfolio/compare-allocations` - Compare strategies
- `POST /portfolio/rebalance` - Calculate rebalancing trades
- `GET /portfolio/allocation-strategies` - List strategies
- `GET /portfolio/rebalancing-strategies` - List rebalancing methods

### Machine Learning
- `POST /ml/train` - Train ML model
- `POST /ml/predict` - Make predictions
- `POST /ml/predict-direction` - Predict price direction
- `POST /ml/engineer-features` - Generate features
- `GET /ml/model-types` - Available models
- `GET /ml/feature-types` - Available feature types

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

- [x] **Strategy Optimization Algorithms (Grid Search, Random Search, Bayesian)**
- [x] **Walk-Forward Optimization**
- [x] **Advanced Portfolio Management (8 allocation strategies)**
- [x] **Automatic Portfolio Rebalancing**
- [x] **Machine Learning Integration (Random Forest, XGBoost, LSTM)**
- [x] **ML-based Trading Strategy**

### In Progress 🚧
- [ ] Real-time alerts (Telegram/Discord)

---

**Happy Trading! 📈**

*Remember: Never invest more than you can afford to lose.*
