/**
 * TypeScript types for the trading bot frontend
 */

export interface BotStatus {
  is_running: boolean
  exchange: string
  strategy: string
  trading_pairs: string[]
  uptime_seconds: number
  last_update: string
}

export interface Balance {
  balances: Record<string, number>
  total_value_eur?: number
  timestamp: string
}

export interface Position {
  symbol: string
  side: 'long' | 'short'
  entry_price: number
  current_price?: number
  size: number
  unrealized_pnl?: number
  unrealized_pnl_percentage?: number
  stop_loss?: number
  take_profit?: number
  entry_time: string
  strategy_name: string
}

export interface Order {
  id: string
  symbol: string
  type: string
  side: 'buy' | 'sell'
  amount: number
  price?: number
  status: string
  timestamp: string
}

export interface Trade {
  id: number
  symbol: string
  side: 'buy' | 'sell'
  amount: number
  price: number
  pnl?: number
  pnl_percentage?: number
  timestamp: string
  strategy_name?: string
}

export interface Performance {
  total_value: number
  cash_balance: number
  initial_capital: number
  unrealized_pnl: number
  realized_pnl: number
  total_pnl: number
  total_return: number
  num_open_positions: number
  num_closed_trades: number
  num_winning_trades: number
  num_losing_trades: number
  win_rate: number
  current_exposure: number
}

export interface PerformanceHistory {
  timestamp: string
  total_balance: number
  unrealized_pnl: number
  realized_pnl: number
  num_open_positions: number
  win_rate: number
}

export interface Strategy {
  id: number
  name: string
  strategy_type: string
  parameters: Record<string, any>
  is_active: boolean
  description?: string
  created_at: string
  updated_at: string
}

export interface AvailableStrategy {
  name: string
  class_name: string
  description: string
}

export interface BacktestRequest {
  strategy_name: string
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  initial_capital: number
  strategy_params: Record<string, any>
  enable_stop_loss: boolean
  enable_take_profit: boolean
}

export interface BacktestResult {
  id: number
  strategy_name: string
  symbol: string
  timeframe: string
  start_date: string
  end_date: string
  initial_capital: number
  final_capital: number
  total_return: number
  num_trades: number
  win_rate: number
  sharpe_ratio: number
  max_drawdown: number
  created_at: string
}

export interface BacktestDetailedResult extends BacktestResult {
  avg_win: number
  avg_loss: number
  profit_factor: number
  parameters: Record<string, any>
  trades: any[]
  equity_curve: number[]
}

export interface Settings {
  exchange: {
    name: string
    testnet: boolean
  }
  trading: {
    pairs: string[]
    timeframe: string
    max_position_size: number
    max_portfolio_risk: number
  }
  strategy: {
    name: string
    params: Record<string, any>
  }
  risk_management: {
    stop_loss_percentage: number
    take_profit_percentage: number
    trailing_stop: boolean
  }
  logging: {
    level: string
    to_file: boolean
  }
}

export interface LogEntry {
  id: number
  timestamp: string
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
  message: string
  module?: string
}

export interface WebSocketMessage {
  type:
    | 'connected'
    | 'price_update'
    | 'position_update'
    | 'order_update'
    | 'trade_execution'
    | 'bot_status'
    | 'log'
    | 'portfolio_update'
    | 'signal'
  timestamp: string
  [key: string]: any
}
