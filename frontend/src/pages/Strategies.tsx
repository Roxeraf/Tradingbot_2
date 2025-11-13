import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
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
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material'
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  PlayArrow as ActivateIcon,
  Stop as DeactivateIcon,
  Assessment as PerformanceIcon,
} from '@mui/icons-material'
import { tradingApi } from '../services/api'
import type { Strategy, AvailableStrategy } from '../types'

export default function Strategies() {
  const queryClient = useQueryClient()

  // State
  const [createDialogOpen, setCreateDialogOpen] = useState(false)
  const [editDialogOpen, setEditDialogOpen] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null)

  // Form state
  const [strategyForm, setStrategyForm] = useState({
    name: '',
    strategy_type: 'moving_average_crossover',
    description: '',
    fast_period: '10',
    slow_period: '30',
    rsi_period: '14',
    rsi_oversold: '30',
    rsi_overbought: '70',
  })

  // Fetch strategies
  const { data: strategiesData, isLoading: strategiesLoading } = useQuery<{ data: Strategy[] }>(
    'strategies',
    tradingApi.getStrategies,
    { refetchInterval: 5000 }
  )

  // Fetch available strategy types
  const { data: availableStrategiesData } = useQuery<{ data: AvailableStrategy[] }>(
    'availableStrategies',
    tradingApi.getAvailableStrategies
  )

  // Mutations
  const createStrategyMutation = useMutation(
    (strategy: any) => tradingApi.createStrategy(strategy),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies')
        setCreateDialogOpen(false)
        resetForm()
      },
    }
  )

  const updateStrategyMutation = useMutation(
    ({ id, strategy }: { id: number; strategy: any }) => tradingApi.updateStrategy(id, strategy),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies')
        setEditDialogOpen(false)
        setSelectedStrategy(null)
        resetForm()
      },
    }
  )

  const deleteStrategyMutation = useMutation(
    (id: number) => tradingApi.deleteStrategy(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies')
      },
    }
  )

  const activateStrategyMutation = useMutation(
    (id: number) => tradingApi.activateStrategy(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies')
      },
    }
  )

  const deactivateStrategyMutation = useMutation(
    (id: number) => tradingApi.deactivateStrategy(id),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('strategies')
      },
    }
  )

  const resetForm = () => {
    setStrategyForm({
      name: '',
      strategy_type: 'moving_average_crossover',
      description: '',
      fast_period: '10',
      slow_period: '30',
      rsi_period: '14',
      rsi_oversold: '30',
      rsi_overbought: '70',
    })
  }

  const handleCreateStrategy = () => {
    const parameters: Record<string, any> = {}

    if (strategyForm.strategy_type === 'moving_average_crossover') {
      parameters.fast_period = parseInt(strategyForm.fast_period)
      parameters.slow_period = parseInt(strategyForm.slow_period)
    } else if (strategyForm.strategy_type === 'rsi') {
      parameters.rsi_period = parseInt(strategyForm.rsi_period)
      parameters.rsi_oversold = parseInt(strategyForm.rsi_oversold)
      parameters.rsi_overbought = parseInt(strategyForm.rsi_overbought)
    }

    const strategy = {
      name: strategyForm.name,
      strategy_type: strategyForm.strategy_type,
      description: strategyForm.description,
      parameters,
    }

    createStrategyMutation.mutate(strategy)
  }

  const openEditDialog = (strategy: Strategy) => {
    setSelectedStrategy(strategy)
    setStrategyForm({
      name: strategy.name,
      strategy_type: strategy.strategy_type,
      description: strategy.description || '',
      fast_period: strategy.parameters.fast_period?.toString() || '10',
      slow_period: strategy.parameters.slow_period?.toString() || '30',
      rsi_period: strategy.parameters.rsi_period?.toString() || '14',
      rsi_oversold: strategy.parameters.rsi_oversold?.toString() || '30',
      rsi_overbought: strategy.parameters.rsi_overbought?.toString() || '70',
    })
    setEditDialogOpen(true)
  }

  const handleUpdateStrategy = () => {
    if (!selectedStrategy) return

    const parameters: Record<string, any> = {}

    if (strategyForm.strategy_type === 'moving_average_crossover') {
      parameters.fast_period = parseInt(strategyForm.fast_period)
      parameters.slow_period = parseInt(strategyForm.slow_period)
    } else if (strategyForm.strategy_type === 'rsi') {
      parameters.rsi_period = parseInt(strategyForm.rsi_period)
      parameters.rsi_oversold = parseInt(strategyForm.rsi_oversold)
      parameters.rsi_overbought = parseInt(strategyForm.rsi_overbought)
    }

    const strategy = {
      name: strategyForm.name,
      strategy_type: strategyForm.strategy_type,
      description: strategyForm.description,
      parameters,
    }

    updateStrategyMutation.mutate({ id: selectedStrategy.id, strategy })
  }

  const strategies = strategiesData?.data || []
  const availableStrategies = availableStrategiesData?.data || []

  const renderStrategyParameters = () => {
    if (strategyForm.strategy_type === 'moving_average_crossover') {
      return (
        <>
          <TextField
            fullWidth
            label="Fast Period"
            type="number"
            value={strategyForm.fast_period}
            onChange={(e) => setStrategyForm({ ...strategyForm, fast_period: e.target.value })}
          />
          <TextField
            fullWidth
            label="Slow Period"
            type="number"
            value={strategyForm.slow_period}
            onChange={(e) => setStrategyForm({ ...strategyForm, slow_period: e.target.value })}
          />
        </>
      )
    } else if (strategyForm.strategy_type === 'rsi') {
      return (
        <>
          <TextField
            fullWidth
            label="RSI Period"
            type="number"
            value={strategyForm.rsi_period}
            onChange={(e) => setStrategyForm({ ...strategyForm, rsi_period: e.target.value })}
          />
          <TextField
            fullWidth
            label="Oversold Level"
            type="number"
            value={strategyForm.rsi_oversold}
            onChange={(e) => setStrategyForm({ ...strategyForm, rsi_oversold: e.target.value })}
          />
          <TextField
            fullWidth
            label="Overbought Level"
            type="number"
            value={strategyForm.rsi_overbought}
            onChange={(e) => setStrategyForm({ ...strategyForm, rsi_overbought: e.target.value })}
          />
        </>
      )
    }
    return null
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Strategies</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Strategy
        </Button>
      </Box>

      {/* Available Strategies Info */}
      {availableStrategies.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Available Strategy Types
            </Typography>
            <Grid container spacing={2}>
              {availableStrategies.map((strategy) => (
                <Grid item xs={12} sm={6} md={4} key={strategy.name}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" fontWeight="bold">
                      {strategy.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {strategy.description}
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Configured Strategies */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Configured Strategies
          </Typography>
          {strategiesLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : strategies.length === 0 ? (
            <Alert severity="info">
              No strategies configured yet. Click "Create Strategy" to get started!
            </Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Parameters</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell align="center">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {strategies.map((strategy) => (
                    <TableRow key={strategy.id}>
                      <TableCell>
                        <Typography variant="body1" fontWeight="bold">
                          {strategy.name}
                        </Typography>
                        {strategy.description && (
                          <Typography variant="body2" color="text.secondary">
                            {strategy.description}
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip label={strategy.strategy_type} size="small" />
                      </TableCell>
                      <TableCell>
                        <Box>
                          {Object.entries(strategy.parameters).map(([key, value]) => (
                            <Typography key={key} variant="body2" color="text.secondary">
                              {key}: {value}
                            </Typography>
                          ))}
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={strategy.is_active ? 'Active' : 'Inactive'}
                          color={strategy.is_active ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        {new Date(strategy.created_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell align="center">
                        {strategy.is_active ? (
                          <IconButton
                            size="small"
                            onClick={() => deactivateStrategyMutation.mutate(strategy.id)}
                            title="Deactivate"
                            color="warning"
                          >
                            <DeactivateIcon fontSize="small" />
                          </IconButton>
                        ) : (
                          <IconButton
                            size="small"
                            onClick={() => activateStrategyMutation.mutate(strategy.id)}
                            title="Activate"
                            color="success"
                          >
                            <ActivateIcon fontSize="small" />
                          </IconButton>
                        )}
                        <IconButton
                          size="small"
                          onClick={() => openEditDialog(strategy)}
                          title="Edit"
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                        <IconButton
                          size="small"
                          onClick={() => deleteStrategyMutation.mutate(strategy.id)}
                          title="Delete"
                          color="error"
                          disabled={strategy.is_active}
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

      {/* Create Strategy Dialog */}
      <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New Strategy</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              fullWidth
              label="Strategy Name"
              value={strategyForm.name}
              onChange={(e) => setStrategyForm({ ...strategyForm, name: e.target.value })}
              placeholder="My MA Crossover Strategy"
            />

            <FormControl fullWidth>
              <InputLabel>Strategy Type</InputLabel>
              <Select
                value={strategyForm.strategy_type}
                label="Strategy Type"
                onChange={(e) =>
                  setStrategyForm({ ...strategyForm, strategy_type: e.target.value })
                }
              >
                <MenuItem value="moving_average_crossover">MA Crossover</MenuItem>
                <MenuItem value="rsi">RSI Strategy</MenuItem>
                <MenuItem value="macd">MACD Strategy</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Description"
              multiline
              rows={2}
              value={strategyForm.description}
              onChange={(e) => setStrategyForm({ ...strategyForm, description: e.target.value })}
              placeholder="Optional description"
            />

            <Divider />
            <Typography variant="subtitle2" color="text.secondary">
              Strategy Parameters
            </Typography>

            {renderStrategyParameters()}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setCreateDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleCreateStrategy}
            variant="contained"
            disabled={!strategyForm.name || createStrategyMutation.isLoading}
          >
            {createStrategyMutation.isLoading ? 'Creating...' : 'Create'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Edit Strategy Dialog */}
      <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Edit Strategy</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              fullWidth
              label="Strategy Name"
              value={strategyForm.name}
              onChange={(e) => setStrategyForm({ ...strategyForm, name: e.target.value })}
            />

            <FormControl fullWidth>
              <InputLabel>Strategy Type</InputLabel>
              <Select
                value={strategyForm.strategy_type}
                label="Strategy Type"
                onChange={(e) =>
                  setStrategyForm({ ...strategyForm, strategy_type: e.target.value })
                }
              >
                <MenuItem value="moving_average_crossover">MA Crossover</MenuItem>
                <MenuItem value="rsi">RSI Strategy</MenuItem>
                <MenuItem value="macd">MACD Strategy</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Description"
              multiline
              rows={2}
              value={strategyForm.description}
              onChange={(e) => setStrategyForm({ ...strategyForm, description: e.target.value })}
            />

            <Divider />
            <Typography variant="subtitle2" color="text.secondary">
              Strategy Parameters
            </Typography>

            {renderStrategyParameters()}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleUpdateStrategy}
            variant="contained"
            disabled={!strategyForm.name || updateStrategyMutation.isLoading}
          >
            {updateStrategyMutation.isLoading ? 'Updating...' : 'Update'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
