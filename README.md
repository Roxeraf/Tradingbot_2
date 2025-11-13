# Cryptocurrency Trading Bot

Ein produktionsreifer, modularer Krypto-Trading-Bot mit Multi-Exchange-Support, anpassbaren Strategien und umfassendem Risikomanagement.

## ✅ Implementierte Funktionen

### 🎯 Kern-Trading-Engine
- ✅ **Async Trading Bot Orchestrator**: Vollständige Orchestrierung aller Komponenten (`src/main.py`)
- ✅ **Trading Loop**: Kontinuierliche Marktüberwachung und Signalverarbeitung
- ✅ **Multi-Exchange Support**: Über CCXT-Bibliothek (Bitpanda vollständig implementiert)
- ✅ **Position Management**: Automatisches Öffnen und Schließen von Positionen
- ✅ **Price Monitoring**: Echtzeit-Preisaktualisierung für alle Trading-Paare

### 📊 Trading-Strategien
- ✅ **Moving Average Crossover**: Vollständig implementierte MA-Crossover-Strategie mit:
  - Bullish/Bearish Crossover Detection
  - Volume Confirmation
  - Confidence Scoring
  - Automatische Stop Loss/Take Profit Berechnung
- ✅ **Base Strategy Framework**: Erweiterbare Basis-Klasse für eigene Strategien
- ✅ **Strategy Factory**: Automatisches Laden von Strategien aus Konfiguration

### 📈 Technische Indikatoren (vollständig implementiert)
- ✅ **SMA** (Simple Moving Average)
- ✅ **EMA** (Exponential Moving Average)
- ✅ **RSI** (Relative Strength Index)
- ✅ **MACD** (Moving Average Convergence Divergence)
- ✅ **Bollinger Bands**
- ✅ **ATR** (Average True Range)
- ✅ **Stochastic Oscillator**
- ✅ **OBV** (On-Balance Volume)
- ✅ **VWAP** (Volume Weighted Average Price)
- ✅ **ADX** (Average Directional Index)
- ✅ **Ichimoku Cloud**
- ✅ **Fibonacci Retracement**

### 🛡️ Risk Management
- ✅ **Position Sizing**: Vier verschiedene Methoden implementiert:
  - Fixed Percentage: Fester Prozentsatz des Portfolios
  - Risk-Based: Basierend auf Stop-Loss-Distanz
  - Kelly Criterion: Mathematisch optimierte Positionsgröße
  - Fixed Amount: Fester Betrag pro Trade
- ✅ **Portfolio Manager**:
  - Echtzeit PnL-Berechnung (realized & unrealized)
  - Stop Loss/Take Profit Überwachung
  - Position Tracking mit Metadaten
  - Win Rate Berechnung
  - Portfolio Risk Exposure Monitoring
- ✅ **Risk Validation**: Automatische Validierung von Positionsgrößen
- ✅ **Portfolio Risk Limits**: Maximale Gesamt-Portfolio-Exposition

### 💼 Order Execution
- ✅ **Order Manager** mit vollständiger Implementierung:
  - Market Orders
  - Limit Orders
  - Stop Loss Orders
  - Take Profit Orders
- ✅ **Order Status Tracking**: Synchronisation mit Exchange
- ✅ **Order History**: Persistente Speicherung aller Orders
- ✅ **Automatic Order Cancellation**: Bei Bot-Shutdown

### 💾 Datenbank (SQLAlchemy)
- ✅ **Trade Storage**: Persistente Speicherung aller Trades
- ✅ **Position Tracking**: Offene und geschlossene Positionen
- ✅ **Performance Metrics**: Zeitreihen-Tracking der Portfolio-Performance
- ✅ **Historical Data Storage**: OHLCV-Daten für Backtesting
- ✅ **Backtest Results**: Speicherung von Backtest-Ergebnissen
- ✅ **System Logs**: Strukturierte Log-Speicherung
- ✅ **Strategy Configurations**: Versionierung von Strategie-Parametern

### 📝 Monitoring & Logging
- ✅ **Loguru-basiertes Logging**: Strukturiertes, konfigurierbares Logging
- ✅ **Performance Tracking**: Echtzeit-Performance-Metriken
- ✅ **Signal Logging**: Detaillierte Aufzeichnung aller Handelssignale
- ✅ **Trade Logging**: Vollständige Trade-Historie
- ✅ **Error Handling**: Umfassende Fehlerbehandlung und -protokollierung
- ✅ **Log Levels**: Konfigurierbare Log-Stufen (DEBUG, INFO, WARNING, ERROR)
- ✅ **File & Console Logging**: Parallel zu Datei und Konsole

### ⚙️ Konfiguration
- ✅ **Environment Variables**: `.env`-basierte Konfiguration
- ✅ **Pydantic Settings**: Typsichere Konfigurationsverwaltung
- ✅ **Exchange Configuration**: API-Keys, Testnet-Modus
- ✅ **Strategy Parameters**: Flexible Strategie-Konfiguration
- ✅ **Risk Parameters**: Konfigurierbare Risikoparameter

### 🔌 Exchange Integration
- ✅ **Base Exchange Interface**: Abstrakte Basis-Klasse für alle Exchanges
- ✅ **Bitpanda Exchange**: Vollständige Implementation
- ✅ **CCXT Integration**: Support für 100+ Exchanges
- ✅ **Exchange Factory**: Automatisches Laden von Exchange-Implementierungen

## ⚠️ Nicht implementierte Features

### 🚧 In Entwicklung / Geplant
- ⏳ **Web Dashboard** (React + TypeScript): Nur Ordnerstruktur vorhanden
- ⏳ **Backtesting Engine**: Nur Grundstruktur vorhanden
- ⏳ **Unit Tests**: Test-Framework vorhanden, Tests noch nicht implementiert
- ⏳ **Telegram/Discord Notifications**: Dependencies installiert, nicht implementiert
- ⏳ **Zusätzliche Strategien**: RSI, MACD, Bollinger Bands Strategien geplant

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

## 🏗️ Architektur

```
src/
├── config/              # ✅ Konfigurationsmanagement (Pydantic Settings)
├── exchanges/           # ✅ Exchange-Adapter (Bitpanda + Base Interface)
│   ├── base_exchange.py         # Abstrakte Basis-Klasse
│   ├── bitpanda_exchange.py     # Bitpanda Implementation
│   └── exchange_factory.py      # Factory Pattern
├── strategies/          # ✅ Trading-Strategien
│   ├── base_strategy.py         # Abstrakte Strategie-Klasse
│   ├── moving_average_strategy.py  # MA Crossover (vollständig)
│   └── strategy_factory.py      # Automatisches Laden
├── indicators/          # ✅ Technische Indikatoren (12+ implementiert)
│   └── technical_indicators.py  # Alle Indikatoren
├── risk_management/     # ✅ Risikomanagement
│   ├── position_sizer.py        # 4 Positionsgrößen-Methoden
│   └── portfolio_manager.py     # Portfolio-Tracking
├── data/               # ✅ Datenbank-Layer
│   ├── models.py                # SQLAlchemy Models
│   └── data_storage.py          # Datenbank-Operationen
├── execution/          # ✅ Order-Management
│   └── order_manager.py         # Vollständige Order-Ausführung
├── monitoring/         # ✅ Logging & Monitoring
│   └── logger.py                # Loguru-basiert
├── api/                # ⏳ Web API (noch nicht implementiert)
├── backtesting/        # ⏳ Backtesting (noch nicht implementiert)
└── main.py            # ✅ Haupt-Orchestrator
```

## 📊 Implementierte Strategie: Moving Average Crossover

### Funktionsweise
Die MA-Crossover-Strategie ist vollständig implementiert mit folgender Logik:

**Buy Signal:**
- Fast MA (Standard: 20 Perioden) kreuzt Slow MA (Standard: 50 Perioden) von unten nach oben
- Volume-Bestätigung: Erhöhtes Volumen verstärkt das Signal
- Confidence Score: Basierend auf Crossover-Stärke und Volumen (0-1)
- Automatische Stop Loss Berechnung: Entry Price - 2% (konfigurierbar)
- Automatische Take Profit Berechnung: Entry Price + 4% (konfigurierbar)

**Sell Signal:**
- Fast MA kreuzt Slow MA von oben nach unten
- Volume-Bestätigung für höhere Confidence
- Automatische Exit-Level-Berechnung

**Hold Signal:**
- Keine Crossover erkannt
- Confidence unter Minimum-Schwelle

### Parameter (konfigurierbar)
```python
{
    'fast_period': 20,        # Fast MA Periode
    'slow_period': 50,        # Slow MA Periode
    'min_confidence': 0.6,    # Minimum Confidence für Trade
    'stop_loss_pct': 0.02,    # 2% Stop Loss
    'take_profit_pct': 0.04   # 4% Take Profit
}
```

### Eigene Strategien erstellen
Erstelle eigene Strategien durch Vererbung von `BaseStrategy`:

```python
from src.strategies.base_strategy import BaseStrategy, TradingSignal

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # Indikatoren berechnen
        pass

    def generate_signal(self, data: pd.DataFrame) -> TradingSignal:
        # Trading-Signal generieren
        pass

    def get_required_history(self) -> int:
        # Benötigte historische Daten
        return 100
```

## 🛡️ Risk Management (vollständig implementiert)

### Position Sizing
Der Bot bietet 4 verschiedene Methoden zur Berechnung der Positionsgröße:

1. **Risk-Based Sizing** (Standard):
   - Berechnet Positionsgröße basierend auf Stop-Loss-Distanz
   - Respektiert max. Risiko pro Trade (Standard: 2%)
   - Berücksichtigt bereits existierende Portfolio-Exposition
   - Verhindert Überschreitung des Portfolio-Risiko-Limits (Standard: 6%)

2. **Fixed Percentage**:
   - Fester Prozentsatz des Portfolios pro Trade
   - Skaliert mit Signal-Confidence

3. **Kelly Criterion**:
   - Mathematisch optimierte Positionsgröße
   - Basierend auf historischer Win-Rate und Avg. Win/Loss

4. **Fixed Amount**:
   - Fixer Betrag pro Trade

### Portfolio Manager
- **Real-time PnL Tracking**: Unrealized und Realized PnL
- **Automatische Stop-Loss/Take-Profit-Überwachung**: Prüft jeden Tick
- **Position-Management**: Öffnen, Schließen, Aktualisieren von Positionen
- **Portfolio-Statistiken**: Total Value, Returns, Win Rate, Exposure
- **Trade-Historie**: Vollständige Aufzeichnung aller geschlossenen Trades

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

## 🗺️ Implementierungs-Status

### ✅ Vollständig implementiert
- ✅ **Core Trading Engine**: Async-basierter Trading-Loop mit vollständiger Orchestrierung
- ✅ **Risk Management System**: 4 Position-Sizing-Methoden + Portfolio-Manager
- ✅ **Database Storage**: SQLAlchemy mit 7 verschiedenen Tabellen
- ✅ **Logging and Monitoring**: Loguru-basiertes strukturiertes Logging
- ✅ **Technical Indicators**: 12+ Indikatoren (SMA, EMA, RSI, MACD, etc.)
- ✅ **Order Execution**: Market, Limit, Stop-Loss, Take-Profit Orders
- ✅ **Exchange Integration**: CCXT + Bitpanda vollständig implementiert
- ✅ **Moving Average Strategy**: Vollständig mit Confidence-Scoring

### 🚧 In Entwicklung
- ⏳ **Web Dashboard** (React + TypeScript): Ordnerstruktur vorhanden
- ⏳ **Backtesting Engine**: Datenbank-Models vorhanden, Engine fehlt
- ⏳ **Unit Tests**: pytest-Framework konfiguriert, Tests müssen geschrieben werden
- ⏳ **Advanced Strategies**: RSI, MACD, Bollinger Bands als Strategien (Indikatoren vorhanden)
- ⏳ **Real-time Alerts**: Telegram/Discord Dependencies installiert, nicht implementiert

### 📈 Roadmap für nächste Releases
1. **v1.1**: Backtesting-Engine implementieren
2. **v1.2**: Weitere Strategien (RSI, MACD, Multi-Indicator)
3. **v1.3**: Web Dashboard für Monitoring
4. **v1.4**: Telegram/Discord Notifications
5. **v2.0**: Machine Learning basierte Strategien

---

**Happy Trading! 📈**

*Remember: Never invest more than you can afford to lose.*
