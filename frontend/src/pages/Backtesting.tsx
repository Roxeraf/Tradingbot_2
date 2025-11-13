import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  Divider,
} from '@mui/material'
import {
  PlayArrow as RunIcon,
  Delete as DeleteIcon,
  Visibility as ViewIcon,
} from '@mui/icons-material'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import { tradingApi } from '../services/api'
import type { BacktestResult, BacktestDetailedResult } from '../types'

export default function Backtesting() {
  const queryClient = useQueryClient()

  // State for backtest form
  const [backtestForm, setBacktestForm] = useState({
    strategy_name: 'moving_average_crossover',
    symbol: 'BTC/EUR',
    timeframe: '1h',
    start_date: '',
    end_date: '',
    initial_capital: '10000',
    fast_period: '10',
    slow_period: '30',
    enable_stop_loss: true,
    enable_take_profit: true,
  })

  // State for selected backtest result
  const [selectedBacktest, setSelectedBacktest] = useState<BacktestDetailedResult | null>(null)

  // Fetch backtest results
  const { data: backtestsData, isLoading: backtestsLoading } = useQuery<{ data: BacktestResult[] }>(
    'backtests',
    tradingApi.getBacktestResults,
    { refetchInterval: 5000 }
  )

  // Run backtest mutation
  const runBacktestMutation = useMutation(
    (request: any) => tradingApi.runBacktest(request),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('backtests')
      },
    }
  )

  // Delete backtest mutation
  const deleteBacktestMutation = useMutation(
    (id: number) => tradingApi.deleteBacktestResult(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('backtests')
        if (selectedBacktest?.id === arguments[0]) {
          setSelectedBacktest(null)
        }
      },
    }
  )

  // View backtest details
  const viewBacktest = async (id: number) => {
    try {
      const response = await tradingApi.getBacktestResult(id)
      setSelectedBacktest(response.data)
    } catch (error) {
      console.error('Error fetching backtest details:', error)
    }
  }

  const handleRunBacktest = () => {
    const request = {
      strategy_name: backtestForm.strategy_name,
      symbol: backtestForm.symbol,
      timeframe: backtestForm.timeframe,
      start_date: backtestForm.start_date,
      end_date: backtestForm.end_date,
      initial_capital: parseFloat(backtestForm.initial_capital),
      strategy_params: {
        fast_period: parseInt(backtestForm.fast_period),
        slow_period: parseInt(backtestForm.slow_period),
      },
      enable_stop_loss: backtestForm.enable_stop_loss,
      enable_take_profit: backtestForm.enable_take_profit,
    }
    runBacktestMutation.mutate(request)
  }

  // Prepare equity curve data for chart
  const equityCurveData = selectedBacktest?.equity_curve
    ? selectedBacktest.equity_curve.map((value, index) => ({
        index,
        equity: value,
      }))
    : []

  const backtests = backtestsData?.data || []

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Backtesting
      </Typography>

      <Grid container spacing={3}>
        {/* Run Backtest Form */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Run New Backtest
              </Typography>
              <Box component="form" sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Strategy</InputLabel>
                  <Select
                    value={backtestForm.strategy_name}
                    label="Strategy"
                    onChange={(e) =>
                      setBacktestForm({ ...backtestForm, strategy_name: e.target.value })
                    }
                  >
                    <MenuItem value="moving_average_crossover">MA Crossover</MenuItem>
                    <MenuItem value="rsi">RSI Strategy</MenuItem>
                    <MenuItem value="macd">MACD Strategy</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  label="Symbol"
                  value={backtestForm.symbol}
                  onChange={(e) => setBacktestForm({ ...backtestForm, symbol: e.target.value })}
                  placeholder="BTC/EUR"
                />

                <FormControl fullWidth>
                  <InputLabel>Timeframe</InputLabel>
                  <Select
                    value={backtestForm.timeframe}
                    label="Timeframe"
                    onChange={(e) =>
                      setBacktestForm({ ...backtestForm, timeframe: e.target.value })
                    }
                  >
                    <MenuItem value="1m">1 Minute</MenuItem>
                    <MenuItem value="5m">5 Minutes</MenuItem>
                    <MenuItem value="15m">15 Minutes</MenuItem>
                    <MenuItem value="1h">1 Hour</MenuItem>
                    <MenuItem value="4h">4 Hours</MenuItem>
                    <MenuItem value="1d">1 Day</MenuItem>
                  </Select>
                </FormControl>

                <TextField
                  fullWidth
                  label="Start Date"
                  type="date"
                  value={backtestForm.start_date}
                  onChange={(e) =>
                    setBacktestForm({ ...backtestForm, start_date: e.target.value })
                  }
                  InputLabelProps={{ shrink: true }}
                />

                <TextField
                  fullWidth
                  label="End Date"
                  type="date"
                  value={backtestForm.end_date}
                  onChange={(e) => setBacktestForm({ ...backtestForm, end_date: e.target.value })}
                  InputLabelProps={{ shrink: true }}
                />

                <TextField
                  fullWidth
                  label="Initial Capital"
                  type="number"
                  value={backtestForm.initial_capital}
                  onChange={(e) =>
                    setBacktestForm({ ...backtestForm, initial_capital: e.target.value })
                  }
                />

                <Divider />
                <Typography variant="subtitle2" color="text.secondary">
                  Strategy Parameters
                </Typography>

                <TextField
                  fullWidth
                  label="Fast Period"
                  type="number"
                  value={backtestForm.fast_period}
                  onChange={(e) =>
                    setBacktestForm({ ...backtestForm, fast_period: e.target.value })
                  }
                />

                <TextField
                  fullWidth
                  label="Slow Period"
                  type="number"
                  value={backtestForm.slow_period}
                  onChange={(e) =>
                    setBacktestForm({ ...backtestForm, slow_period: e.target.value })
                  }
                />

                <Button
                  variant="contained"
                  startIcon={<RunIcon />}
                  onClick={handleRunBacktest}
                  disabled={
                    runBacktestMutation.isLoading ||
                    !backtestForm.symbol ||
                    !backtestForm.start_date ||
                    !backtestForm.end_date
                  }
                  fullWidth
                >
                  {runBacktestMutation.isLoading ? 'Running...' : 'Run Backtest'}
                </Button>

                {runBacktestMutation.isError && (
                  <Alert severity="error">
                    Error running backtest. Please check your parameters.
                  </Alert>
                )}

                {runBacktestMutation.isSuccess && (
                  <Alert severity="success">Backtest started successfully!</Alert>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results Section */}
        <Grid item xs={12} md={8}>
          {/* Selected Backtest Details */}
          {selectedBacktest ? (
            <>
              <Card sx={{ mb: 3 }}>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                    <Typography variant="h6">Backtest Results</Typography>
                    <Button
                      size="small"
                      onClick={() => setSelectedBacktest(null)}
                    >
                      Back to List
                    </Button>
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Strategy
                        </Typography>
                        <Typography variant="h6">{selectedBacktest.strategy_name}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Symbol
                        </Typography>
                        <Typography variant="h6">{selectedBacktest.symbol}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Timeframe
                        </Typography>
                        <Typography variant="h6">{selectedBacktest.timeframe}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Total Return
                        </Typography>
                        <Typography
                          variant="h6"
                          color={selectedBacktest.total_return >= 0 ? 'success.main' : 'error.main'}
                        >
                          {selectedBacktest.total_return >= 0 ? '+' : ''}
                          {selectedBacktest.total_return.toFixed(2)}%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Sharpe Ratio
                        </Typography>
                        <Typography variant="h6">
                          {selectedBacktest.sharpe_ratio.toFixed(2)}
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Max Drawdown
                        </Typography>
                        <Typography variant="h6" color="error.main">
                          {selectedBacktest.max_drawdown.toFixed(2)}%
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Win Rate
                        </Typography>
                        <Typography variant="h6">{selectedBacktest.win_rate.toFixed(1)}%</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Total Trades
                        </Typography>
                        <Typography variant="h6">{selectedBacktest.num_trades}</Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={12} sm={6} md={4}>
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          Final Capital
                        </Typography>
                        <Typography variant="h6">
                          €{selectedBacktest.final_capital.toLocaleString()}
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>

              {/* Equity Curve */}
              {equityCurveData.length > 0 && (
                <Card sx={{ mb: 3 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Equity Curve
                    </Typography>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={equityCurveData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="index" label={{ value: 'Trade Number', position: 'insideBottom', offset: -5 }} />
                        <YAxis label={{ value: 'Capital (€)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line
                          type="monotone"
                          dataKey="equity"
                          stroke="#8884d8"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            /* Backtest Results List */
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Historical Backtest Results
                </Typography>
                {backtestsLoading ? (
                  <Box display="flex" justifyContent="center" p={3}>
                    <CircularProgress />
                  </Box>
                ) : backtests.length === 0 ? (
                  <Alert severity="info">No backtest results yet. Run a backtest to get started!</Alert>
                ) : (
                  <TableContainer component={Paper} variant="outlined">
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Date</TableCell>
                          <TableCell>Strategy</TableCell>
                          <TableCell>Symbol</TableCell>
                          <TableCell>Timeframe</TableCell>
                          <TableCell align="right">Return %</TableCell>
                          <TableCell align="right">Win Rate</TableCell>
                          <TableCell align="right">Sharpe</TableCell>
                          <TableCell align="right">Trades</TableCell>
                          <TableCell align="center">Actions</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {backtests.map((backtest) => (
                          <TableRow key={backtest.id}>
                            <TableCell>
                              {new Date(backtest.created_at).toLocaleDateString()}
                            </TableCell>
                            <TableCell>
                              <Chip label={backtest.strategy_name} size="small" />
                            </TableCell>
                            <TableCell>{backtest.symbol}</TableCell>
                            <TableCell>{backtest.timeframe}</TableCell>
                            <TableCell align="right">
                              <Typography
                                color={backtest.total_return >= 0 ? 'success.main' : 'error.main'}
                              >
                                {backtest.total_return >= 0 ? '+' : ''}
                                {backtest.total_return.toFixed(2)}%
                              </Typography>
                            </TableCell>
                            <TableCell align="right">{backtest.win_rate.toFixed(1)}%</TableCell>
                            <TableCell align="right">{backtest.sharpe_ratio.toFixed(2)}</TableCell>
                            <TableCell align="right">{backtest.num_trades}</TableCell>
                            <TableCell align="center">
                              <IconButton
                                size="small"
                                onClick={() => viewBacktest(backtest.id)}
                                title="View Details"
                              >
                                <ViewIcon fontSize="small" />
                              </IconButton>
                              <IconButton
                                size="small"
                                onClick={() => deleteBacktestMutation.mutate(backtest.id)}
                                title="Delete"
                                color="error"
                              >
                                <DeleteIcon fontSize="small" />
                              </IconButton>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  )
}
