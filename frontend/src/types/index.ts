export interface BotStatus {
  status: 'running' | 'stopped' | 'starting' | 'stopping' | 'error'
  uptime_seconds?: number
  current_strategy?: string
  active_pairs: string[]
  open_positions: number
  total_trades: number
  current_equity?: number
  message?: string
}

export interface Performance {
  total_trades: number
  winning_trades: number
  losing_trades: number
  win_rate: number
  total_pnl: number
  total_pnl_percentage: number
  avg_trade_pnl: number
  best_trade: number
  worst_trade: number
  sharpe_ratio?: number
  max_drawdown?: number
  current_equity: number
  initial_equity: number
}

export interface Trade {
  id?: number
  symbol: string
  side: string
  entry_price: number
  exit_price?: number
  amount: number
  pnl?: number
  pnl_percentage?: number
  entry_time: string
  exit_time?: string
  status: string
  strategy?: string
}

export interface Position {
  symbol: string
  side: string
  entry_price: number
  current_price: number
  amount: number
  unrealized_pnl: number
  unrealized_pnl_percentage: number
  stop_loss?: number
  take_profit?: number
  entry_time: string
}

export interface Order {
  id: string
  symbol: string
  type: string
  side: string
  amount: number
  price?: number
  status: string
  filled?: number
  remaining?: number
  timestamp: string
}

export interface Strategy {
  name: string
  description?: string
  parameters: Record<string, any>
  required_history: number
}

export interface BacktestResult {
  strategy_name: string
  strategy_params: Record<string, any>
  initial_capital: number
  final_capital: number
  total_return: number
  total_return_pct: number
  annualized_return: number
  num_trades: number
  win_rate: number
  max_drawdown_pct: number
  sharpe_ratio: number
  sortino_ratio: number
  profit_factor: number
  expectancy: number
}

export interface Balance {
  balances: Record<string, number>
  total_value_usd?: number
  timestamp: string
}
