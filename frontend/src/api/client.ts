import axios from 'axios'
import type {
  BotStatus,
  Performance,
  Trade,
  Position,
  Order,
  Strategy,
  Balance,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Bot Control
export const botApi = {
  getStatus: () => api.get<BotStatus>('/bot/status'),
  start: (config?: Record<string, any>) =>
    api.post('/bot/start', { config_override: config }),
  stop: (cancelOrders = true, closePositions = false) =>
    api.post('/bot/stop', { cancel_orders: cancelOrders, close_positions: closePositions }),
  restart: () => api.post('/bot/restart'),
  updateConfig: (updates: Record<string, any>) =>
    api.put('/bot/config', { config_updates: updates }),
}

// Trading
export const tradingApi = {
  getBalance: () => api.get<Balance>('/trading/balance'),
  getPositions: (symbol?: string) =>
    api.get<Position[]>('/trading/positions', { params: { symbol } }),
  getOrders: (symbol?: string, status?: string) =>
    api.get<Order[]>('/trading/orders', { params: { symbol, status_filter: status } }),
  getTrades: (symbol?: string, limit = 100) =>
    api.get<Trade[]>('/trading/trades', { params: { symbol, limit } }),
  getPerformance: () => api.get<Performance>('/trading/performance'),
  placeOrder: (order: {
    symbol: string
    order_type: string
    side: string
    amount: number
    price?: number
    stop_loss?: number
    take_profit?: number
  }) => api.post<Order>('/trading/orders', order),
  cancelOrder: (orderId: string, symbol: string) =>
    api.delete(`/trading/orders/${orderId}`, { params: { symbol } }),
  closePosition: (symbol: string) =>
    api.post(`/trading/positions/${symbol}/close`),
}

// Strategies
export const strategyApi = {
  listStrategies: () => api.get<string[]>('/strategies/'),
  getStrategyInfo: (name: string) => api.get<Strategy>(`/strategies/${name}`),
  getCurrentStrategy: () => api.get<Strategy>('/strategies/current/info'),
  updateStrategy: (strategyName?: string, parameters?: Record<string, any>) =>
    api.put('/strategies/current', {
      strategy_name: strategyName,
      parameters: parameters || {},
    }),
  validateParams: (strategyName: string, parameters: Record<string, any>) =>
    api.post('/strategies/validate', null, {
      params: { strategy_name: strategyName, parameters },
    }),
}

export default api
