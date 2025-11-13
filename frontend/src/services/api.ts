/**
 * API client for communicating with the trading bot backend
 */
import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Add request interceptor for auth (if needed in future)
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('auth_token')
    }
    return Promise.reject(error)
  }
)

// Trading endpoints
export const tradingApi = {
  getStatus: () => api.get('/trading/status'),
  startBot: () => api.post('/trading/start'),
  stopBot: () => api.post('/trading/stop'),
  getBalance: () => api.get('/trading/balance'),
  getPositions: (symbol?: string) => api.get('/trading/positions', { params: { symbol } }),
  closePosition: (symbol: string, reason?: string) => api.post('/trading/positions/close', { symbol, reason }),
  closeAllPositions: () => api.post('/trading/positions/close-all'),
  updatePosition: (symbol: string, stopLoss?: number, takeProfit?: number) =>
    api.put('/trading/positions/update', { symbol, stop_loss: stopLoss, take_profit: takeProfit }),
  getOrders: (symbol?: string, status?: string) => api.get('/trading/orders', { params: { symbol, status } }),
  placeOrder: (data: any) => api.post('/trading/orders', data),
  cancelOrder: (orderId: string, symbol: string) => api.delete(`/trading/orders/${orderId}`, { params: { symbol } }),
  getTrades: (symbol?: string, limit?: number) => api.get('/trading/trades', { params: { symbol, limit } }),
  getPerformance: () => api.get('/trading/performance'),
  getPerformanceHistory: (days?: number) => api.get('/trading/performance/history', { params: { days } }),
}

// Strategy endpoints
export const strategyApi = {
  listStrategies: (activeOnly?: boolean) => api.get('/strategies', { params: { active_only: activeOnly } }),
  getStrategy: (strategyId: number) => api.get(`/strategies/${strategyId}`),
  getStrategyByName: (name: string) => api.get(`/strategies/name/${name}`),
  createStrategy: (data: any) => api.post('/strategies', data),
  updateStrategy: (strategyId: number, data: any) => api.put(`/strategies/${strategyId}`, data),
  activateStrategy: (strategyId: number) => api.post(`/strategies/${strategyId}/activate`),
  deactivateStrategy: (strategyId: number) => api.post(`/strategies/${strategyId}/deactivate`),
  testStrategy: (strategyId: number) => api.post(`/strategies/${strategyId}/test`),
  getStrategyPerformance: (strategyId: number, days?: number) =>
    api.get(`/strategies/${strategyId}/performance`, { params: { days } }),
  getAvailableStrategies: () => api.get('/strategies/available'),
}

// Backtest endpoints
export const backtestApi = {
  runBacktest: (data: any) => api.post('/backtest/run', data),
  getBacktestStatus: (backtestId: string) => api.get(`/backtest/status/${backtestId}`),
  listBacktestResults: (strategyName?: string, limit?: number) =>
    api.get('/backtest/results', { params: { strategy_name: strategyName, limit } }),
  getBacktestResult: (resultId: number) => api.get(`/backtest/results/${resultId}`),
  deleteBacktestResult: (resultId: number) => api.delete(`/backtest/results/${resultId}`),
}

// Settings endpoints
export const settingsApi = {
  getSettings: () => api.get('/settings'),
  updateSettings: (data: any) => api.put('/settings', data),
  getAvailableExchanges: () => api.get('/settings/exchanges'),
  getAvailablePairs: () => api.get('/settings/pairs'),
  getAvailableTimeframes: () => api.get('/settings/timeframes'),
  testExchangeConnection: () => api.post('/settings/test-connection'),
  getLogs: (level?: string, limit?: number) => api.get('/settings/logs', { params: { level, limit } }),
  getSystemInfo: () => api.get('/settings/system-info'),
}

export default api
