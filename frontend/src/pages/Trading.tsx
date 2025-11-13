import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  IconButton,
  Alert,
  CircularProgress,
} from '@mui/material'
import {
  Close as CloseIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Add as AddIcon,
} from '@mui/icons-material'
import { tradingApi } from '../services/api'
import useWebSocket from '../hooks/useWebSocket'
import type { Position, Order, Trade } from '../types'

export default function Trading() {
  const queryClient = useQueryClient()
  const { lastMessage } = useWebSocket()

  // State for dialogs
  const [placeOrderOpen, setPlaceOrderOpen] = useState(false)
  const [updatePositionOpen, setUpdatePositionOpen] = useState(false)
  const [selectedPosition, setSelectedPosition] = useState<Position | null>(null)

  // State for forms
  const [orderForm, setOrderForm] = useState({
    symbol: '',
    side: 'buy',
    order_type: 'market',
    amount: '',
    price: '',
  })

  const [positionUpdateForm, setPositionUpdateForm] = useState({
    stop_loss: '',
    take_profit: '',
  })

  // Fetch positions
  const { data: positionsData, isLoading: positionsLoading } = useQuery<{ data: Position[] }>(
    'positions',
    () => tradingApi.getPositions(),
    { refetchInterval: 5000 }
  )

  // Fetch orders
  const { data: ordersData, isLoading: ordersLoading } = useQuery<{ data: Order[] }>(
    'orders',
    tradingApi.getOrders,
    { refetchInterval: 3000 }
  )

  // Fetch trades
  const { data: tradesData, isLoading: tradesLoading } = useQuery<{ data: Trade[] }>(
    'trades',
    () => tradingApi.getTrades(),
    { refetchInterval: 10000 }
  )

  // Mutations
  const placeOrderMutation = useMutation(
    (order: any) => tradingApi.placeOrder(order),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('orders')
        queryClient.invalidateQueries('positions')
        setPlaceOrderOpen(false)
        setOrderForm({
          symbol: '',
          side: 'buy',
          order_type: 'market',
          amount: '',
          price: '',
        })
      },
    }
  )

  const cancelOrderMutation = useMutation(
    (orderId: string) => tradingApi.cancelOrder(orderId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('orders')
      },
    }
  )

  const closePositionMutation = useMutation(
    (symbol: string) => tradingApi.closePosition(symbol),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('positions')
        queryClient.invalidateQueries('trades')
      },
    }
  )

  const updatePositionMutation = useMutation(
    ({ symbol, updates }: { symbol: string; updates: any }) =>
      tradingApi.updatePosition(symbol, updates),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('positions')
        setUpdatePositionOpen(false)
        setSelectedPosition(null)
      },
    }
  )

  // Handle WebSocket updates
  useEffect(() => {
    if (lastMessage?.type === 'position_update') {
      queryClient.invalidateQueries('positions')
    } else if (lastMessage?.type === 'order_update') {
      queryClient.invalidateQueries('orders')
    } else if (lastMessage?.type === 'trade_execution') {
      queryClient.invalidateQueries('trades')
      queryClient.invalidateQueries('positions')
    }
  }, [lastMessage, queryClient])

  const handlePlaceOrder = () => {
    const order = {
      symbol: orderForm.symbol,
      side: orderForm.side,
      order_type: orderForm.order_type,
      amount: parseFloat(orderForm.amount),
      ...(orderForm.order_type === 'limit' && { price: parseFloat(orderForm.price) }),
    }
    placeOrderMutation.mutate(order)
  }

  const handleUpdatePosition = () => {
    if (!selectedPosition) return

    const updates: any = {}
    if (positionUpdateForm.stop_loss) {
      updates.stop_loss = parseFloat(positionUpdateForm.stop_loss)
    }
    if (positionUpdateForm.take_profit) {
      updates.take_profit = parseFloat(positionUpdateForm.take_profit)
    }

    updatePositionMutation.mutate({
      symbol: selectedPosition.symbol,
      updates,
    })
  }

  const openUpdatePositionDialog = (position: Position) => {
    setSelectedPosition(position)
    setPositionUpdateForm({
      stop_loss: position.stop_loss?.toString() || '',
      take_profit: position.take_profit?.toString() || '',
    })
    setUpdatePositionOpen(true)
  }

  const positions = positionsData?.data || []
  const orders = ordersData?.data || []
  const trades = tradesData?.data || []

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Trading</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setPlaceOrderOpen(true)}
        >
          Place Order
        </Button>
      </Box>

      {/* Open Positions */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Open Positions
          </Typography>
          {positionsLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : positions.length === 0 ? (
            <Alert severity="info">No open positions</Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell align="right">Entry Price</TableCell>
                    <TableCell align="right">Current Price</TableCell>
                    <TableCell align="right">Amount</TableCell>
                    <TableCell align="right">Unrealized PnL</TableCell>
                    <TableCell align="right">PnL %</TableCell>
                    <TableCell align="right">Stop Loss</TableCell>
                    <TableCell align="right">Take Profit</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {positions.map((position) => {
                    const pnlColor = position.unrealized_pnl >= 0 ? 'success.main' : 'error.main'
                    return (
                      <TableRow key={position.symbol}>
                        <TableCell>{position.symbol}</TableCell>
                        <TableCell>
                          <Chip
                            label={position.side.toUpperCase()}
                            color={position.side === 'long' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">
                          €{position.entry_price.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          €{position.current_price.toFixed(2)}
                        </TableCell>
                        <TableCell align="right">
                          {position.size.toFixed(4)}
                        </TableCell>
                        <TableCell align="right" sx={{ color: pnlColor }}>
                          {position.unrealized_pnl && position.unrealized_pnl >= 0 ? '+' : ''}€
                          {position.unrealized_pnl?.toFixed(2) || '0.00'}
                        </TableCell>
                        <TableCell align="right" sx={{ color: pnlColor }}>
                          {position.unrealized_pnl_percentage && position.unrealized_pnl_percentage >= 0 ? '+' : ''}
                          {position.unrealized_pnl_percentage?.toFixed(2) || '0.00'}%
                        </TableCell>
                        <TableCell align="right">
                          {position.stop_loss ? `€${position.stop_loss.toFixed(2)}` : '-'}
                        </TableCell>
                        <TableCell align="right">
                          {position.take_profit ? `€${position.take_profit.toFixed(2)}` : '-'}
                        </TableCell>
                        <TableCell align="center">
                          <IconButton
                            size="small"
                            onClick={() => openUpdatePositionDialog(position)}
                            title="Update SL/TP"
                          >
                            <EditIcon fontSize="small" />
                          </IconButton>
                          <IconButton
                            size="small"
                            onClick={() => closePositionMutation.mutate(position.symbol)}
                            title="Close Position"
                            color="error"
                          >
                            <CloseIcon fontSize="small" />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Active Orders */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Active Orders
          </Typography>
          {ordersLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : orders.length === 0 ? (
            <Alert severity="info">No active orders</Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Order ID</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell align="right">Amount</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Time</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {orders.map((order) => (
                    <TableRow key={order.id}>
                      <TableCell>{order.id}</TableCell>
                      <TableCell>{order.symbol}</TableCell>
                      <TableCell>
                        <Chip label={order.type.toUpperCase()} size="small" />
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={order.side.toUpperCase()}
                          color={order.side === 'buy' ? 'success' : 'error'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell align="right">{order.amount.toFixed(4)}</TableCell>
                      <TableCell align="right">
                        {order.price ? `€${order.price.toFixed(2)}` : '-'}
                      </TableCell>
                      <TableCell>
                        <Chip label={order.status.toUpperCase()} size="small" />
                      </TableCell>
                      <TableCell>
                        {new Date(order.timestamp).toLocaleString()}
                      </TableCell>
                      <TableCell align="center">
                        {order.status === 'open' && (
                          <IconButton
                            size="small"
                            onClick={() => cancelOrderMutation.mutate(order.id)}
                            color="error"
                            title="Cancel Order"
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Trade History */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Trade History (Recent 20)
          </Typography>
          {tradesLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : trades.length === 0 ? (
            <Alert severity="info">No trades yet</Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Time</TableCell>
                    <TableCell>Symbol</TableCell>
                    <TableCell>Side</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">Amount</TableCell>
                    <TableCell align="right">PnL</TableCell>
                    <TableCell align="right">PnL %</TableCell>
                    <TableCell>Strategy</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {trades.slice(0, 20).map((trade, idx) => {
                    const pnlColor = trade.pnl && trade.pnl >= 0 ? 'success.main' : 'error.main'
                    return (
                      <TableRow key={idx}>
                        <TableCell>
                          {new Date(trade.timestamp).toLocaleString()}
                        </TableCell>
                        <TableCell>{trade.symbol}</TableCell>
                        <TableCell>
                          <Chip
                            label={trade.side.toUpperCase()}
                            color={trade.side === 'buy' ? 'success' : 'error'}
                            size="small"
                          />
                        </TableCell>
                        <TableCell align="right">€{trade.price.toFixed(2)}</TableCell>
                        <TableCell align="right">{trade.amount.toFixed(4)}</TableCell>
                        <TableCell align="right" sx={{ color: pnlColor }}>
                          {trade.pnl !== undefined && trade.pnl !== null ? (
                            `${trade.pnl >= 0 ? '+' : ''}€${trade.pnl.toFixed(2)}`
                          ) : '-'}
                        </TableCell>
                        <TableCell align="right" sx={{ color: pnlColor }}>
                          {trade.pnl_percentage !== undefined && trade.pnl_percentage !== null ? (
                            `${trade.pnl_percentage >= 0 ? '+' : ''}${trade.pnl_percentage.toFixed(2)}%`
                          ) : '-'}
                        </TableCell>
                        <TableCell>{trade.strategy_name || '-'}</TableCell>
                      </TableRow>
                    )
                  })}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Place Order Dialog */}
      <Dialog open={placeOrderOpen} onClose={() => setPlaceOrderOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Place New Order</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Symbol"
                value={orderForm.symbol}
                onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value })}
                placeholder="BTC/EUR"
              />
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Side</InputLabel>
                <Select
                  value={orderForm.side}
                  label="Side"
                  onChange={(e) => setOrderForm({ ...orderForm, side: e.target.value })}
                >
                  <MenuItem value="buy">Buy</MenuItem>
                  <MenuItem value="sell">Sell</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={6}>
              <FormControl fullWidth>
                <InputLabel>Order Type</InputLabel>
                <Select
                  value={orderForm.order_type}
                  label="Order Type"
                  onChange={(e) => setOrderForm({ ...orderForm, order_type: e.target.value })}
                >
                  <MenuItem value="market">Market</MenuItem>
                  <MenuItem value="limit">Limit</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Amount"
                type="number"
                value={orderForm.amount}
                onChange={(e) => setOrderForm({ ...orderForm, amount: e.target.value })}
              />
            </Grid>
            {orderForm.order_type === 'limit' && (
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Limit Price"
                  type="number"
                  value={orderForm.price}
                  onChange={(e) => setOrderForm({ ...orderForm, price: e.target.value })}
                />
              </Grid>
            )}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPlaceOrderOpen(false)}>Cancel</Button>
          <Button
            onClick={handlePlaceOrder}
            variant="contained"
            disabled={
              !orderForm.symbol ||
              !orderForm.amount ||
              (orderForm.order_type === 'limit' && !orderForm.price)
            }
          >
            Place Order
          </Button>
        </DialogActions>
      </Dialog>

      {/* Update Position Dialog */}
      <Dialog
        open={updatePositionOpen}
        onClose={() => setUpdatePositionOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          Update Position: {selectedPosition?.symbol}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Stop Loss"
                type="number"
                value={positionUpdateForm.stop_loss}
                onChange={(e) =>
                  setPositionUpdateForm({ ...positionUpdateForm, stop_loss: e.target.value })
                }
                helperText="Leave empty to remove stop loss"
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Take Profit"
                type="number"
                value={positionUpdateForm.take_profit}
                onChange={(e) =>
                  setPositionUpdateForm({ ...positionUpdateForm, take_profit: e.target.value })
                }
                helperText="Leave empty to remove take profit"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setUpdatePositionOpen(false)}>Cancel</Button>
          <Button onClick={handleUpdatePosition} variant="contained">
            Update
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
