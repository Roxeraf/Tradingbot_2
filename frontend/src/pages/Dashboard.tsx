import { useEffect, useState } from 'react'
import { useQuery } from 'react-query'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Chip,
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
} from '@mui/icons-material'
import { tradingApi } from '../services/api'
import useWebSocket from '../hooks/useWebSocket'
import type { BotStatus, Performance } from '../types'

export default function Dashboard() {
  const [performance, setPerformance] = useState<Performance | null>(null)
  const { lastMessage } = useWebSocket()

  // Fetch bot status
  const { data: statusData, isLoading: statusLoading } = useQuery<{ data: BotStatus }>(
    'botStatus',
    tradingApi.getStatus,
    { refetchInterval: 5000 }
  )

  // Fetch performance
  const { data: perfData, isLoading: perfLoading } = useQuery<{ data: Performance }>(
    'performance',
    tradingApi.getPerformance,
    { refetchInterval: 10000 }
  )

  useEffect(() => {
    if (perfData) {
      setPerformance(perfData.data)
    }
  }, [perfData])

  // Update performance from WebSocket
  useEffect(() => {
    if (lastMessage?.type === 'portfolio_update') {
      setPerformance((prev) => ({
        ...prev!,
        ...lastMessage.portfolio,
      }))
    }
  }, [lastMessage])

  if (statusLoading || perfLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    )
  }

  const status = statusData?.data
  const perf = performance

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      {/* Bot Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Typography variant="h6">Bot Status</Typography>
            <Chip
              label={status?.is_running ? 'Running' : 'Stopped'}
              color={status?.is_running ? 'success' : 'default'}
            />
          </Box>
          <Box mt={2}>
            <Typography variant="body2" color="text.secondary">
              Exchange: {status?.exchange} | Strategy: {status?.strategy}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Trading Pairs: {status?.trading_pairs.join(', ')}
            </Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Performance Metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <AccountBalance color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Total Value</Typography>
              </Box>
              <Typography variant="h4">
                €{perf?.total_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Cash: €{perf?.cash_balance.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <ShowChart color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Total PnL</Typography>
              </Box>
              <Typography
                variant="h4"
                color={perf && perf.total_pnl >= 0 ? 'success.main' : 'error.main'}
              >
                {perf && perf.total_pnl >= 0 ? '+' : ''}€
                {perf?.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {perf && perf.total_return >= 0 ? '+' : ''}
                {perf?.total_return.toFixed(2)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <TrendingUp color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Win Rate</Typography>
              </Box>
              <Typography variant="h4">{perf?.win_rate.toFixed(1)}%</Typography>
              <Typography variant="body2" color="text.secondary">
                {perf?.num_winning_trades} wins / {perf?.num_losing_trades} losses
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={1}>
                <TrendingDown color="warning" sx={{ mr: 1 }} />
                <Typography variant="h6">Open Positions</Typography>
              </Box>
              <Typography variant="h4">{perf?.num_open_positions}</Typography>
              <Typography variant="body2" color="text.secondary">
                Exposure: {perf && (perf.current_exposure * 100).toFixed(1)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activity */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Activity
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Activity feed will be displayed here
          </Typography>
        </CardContent>
      </Card>
    </Box>
  )
}
