import { useState, useEffect } from 'react'
import { useQuery, useMutation } from 'react-query'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Chip,
  IconButton,
  CircularProgress,
} from '@mui/material'
import {
  Save as SaveIcon,
  Refresh as TestIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material'
import { tradingApi } from '../services/api'
import type { Settings as SettingsType } from '../types'

export default function Settings() {
  const [settings, setSettings] = useState<SettingsType | null>(null)
  const [newPair, setNewPair] = useState('')
  const [testConnectionStatus, setTestConnectionStatus] = useState<string | null>(null)

  // Fetch settings
  const { data: settingsData, isLoading } = useQuery<{ data: SettingsType }>(
    'settings',
    tradingApi.getSettings
  )

  // Update settings mutation
  const updateSettingsMutation = useMutation(
    (newSettings: SettingsType) => tradingApi.updateSettings(newSettings),
    {
      onSuccess: () => {
        setTestConnectionStatus('Settings saved successfully!')
        setTimeout(() => setTestConnectionStatus(null), 3000)
      },
      onError: () => {
        setTestConnectionStatus('Error saving settings')
        setTimeout(() => setTestConnectionStatus(null), 3000)
      },
    }
  )

  // Test connection mutation
  const testConnectionMutation = useMutation(
    () => tradingApi.testExchangeConnection(),
    {
      onSuccess: (response) => {
        if (response.data.success) {
          setTestConnectionStatus('Connection successful!')
        } else {
          setTestConnectionStatus('Connection failed: ' + response.data.message)
        }
        setTimeout(() => setTestConnectionStatus(null), 5000)
      },
      onError: () => {
        setTestConnectionStatus('Connection test failed')
        setTimeout(() => setTestConnectionStatus(null), 3000)
      },
    }
  )

  useEffect(() => {
    if (settingsData?.data) {
      setSettings(settingsData.data)
    }
  }, [settingsData])

  const handleSaveSettings = () => {
    if (settings) {
      updateSettingsMutation.mutate(settings)
    }
  }

  const handleTestConnection = () => {
    testConnectionMutation.mutate()
  }

  const addTradingPair = () => {
    if (newPair && settings) {
      setSettings({
        ...settings,
        trading: {
          ...settings.trading,
          pairs: [...settings.trading.pairs, newPair],
        },
      })
      setNewPair('')
    }
  }

  const removeTradingPair = (pair: string) => {
    if (settings) {
      setSettings({
        ...settings,
        trading: {
          ...settings.trading,
          pairs: settings.trading.pairs.filter((p) => p !== pair),
        },
      })
    }
  }

  if (isLoading || !settings) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    )
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">Settings</Typography>
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            startIcon={<TestIcon />}
            onClick={handleTestConnection}
            disabled={testConnectionMutation.isLoading}
          >
            {testConnectionMutation.isLoading ? 'Testing...' : 'Test Connection'}
          </Button>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={handleSaveSettings}
            disabled={updateSettingsMutation.isLoading}
          >
            {updateSettingsMutation.isLoading ? 'Saving...' : 'Save Settings'}
          </Button>
        </Box>
      </Box>

      {testConnectionStatus && (
        <Alert
          severity={
            testConnectionStatus.includes('success') ? 'success' :
            testConnectionStatus.includes('failed') || testConnectionStatus.includes('Error') ? 'error' :
            'info'
          }
          sx={{ mb: 3 }}
        >
          {testConnectionStatus}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Exchange Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Exchange Configuration
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Exchange</InputLabel>
                  <Select
                    value={settings.exchange.name}
                    label="Exchange"
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        exchange: { ...settings.exchange, name: e.target.value },
                      })
                    }
                  >
                    <MenuItem value="bitpanda">Bitpanda</MenuItem>
                    <MenuItem value="binance">Binance</MenuItem>
                    <MenuItem value="kraken">Kraken</MenuItem>
                    <MenuItem value="coinbase">Coinbase</MenuItem>
                  </Select>
                </FormControl>

                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.exchange.testnet}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          exchange: { ...settings.exchange, testnet: e.target.checked },
                        })
                      }
                    />
                  }
                  label="Use Testnet"
                />

                <Alert severity="warning">
                  API keys are stored in environment variables and cannot be changed here. Update your .env file to change API credentials.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Trading Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Trading Configuration
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Trading Pairs
                  </Typography>
                  <Box display="flex" gap={1} mb={1}>
                    {settings.trading.pairs.map((pair) => (
                      <Chip
                        key={pair}
                        label={pair}
                        onDelete={() => removeTradingPair(pair)}
                        size="small"
                      />
                    ))}
                  </Box>
                  <Box display="flex" gap={1}>
                    <TextField
                      size="small"
                      placeholder="BTC/EUR"
                      value={newPair}
                      onChange={(e) => setNewPair(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          addTradingPair()
                        }
                      }}
                    />
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={addTradingPair}
                      startIcon={<AddIcon />}
                    >
                      Add
                    </Button>
                  </Box>
                </Box>

                <FormControl fullWidth>
                  <InputLabel>Timeframe</InputLabel>
                  <Select
                    value={settings.trading.timeframe}
                    label="Timeframe"
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        trading: { ...settings.trading, timeframe: e.target.value },
                      })
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
                  label="Max Position Size (€)"
                  type="number"
                  value={settings.trading.max_position_size}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      trading: {
                        ...settings.trading,
                        max_position_size: parseFloat(e.target.value),
                      },
                    })
                  }
                />

                <TextField
                  fullWidth
                  label="Max Portfolio Risk (%)"
                  type="number"
                  value={settings.trading.max_portfolio_risk * 100}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      trading: {
                        ...settings.trading,
                        max_portfolio_risk: parseFloat(e.target.value) / 100,
                      },
                    })
                  }
                  inputProps={{ min: 0, max: 100, step: 1 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Strategy Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Strategy Configuration
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Active Strategy</InputLabel>
                  <Select
                    value={settings.strategy.name}
                    label="Active Strategy"
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        strategy: { ...settings.strategy, name: e.target.value },
                      })
                    }
                  >
                    <MenuItem value="moving_average_crossover">MA Crossover</MenuItem>
                    <MenuItem value="rsi">RSI Strategy</MenuItem>
                    <MenuItem value="macd">MACD Strategy</MenuItem>
                  </Select>
                </FormControl>

                <Alert severity="info">
                  Strategy parameters can be configured on the Strategies page.
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Risk Management Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Risk Management
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  fullWidth
                  label="Stop Loss (%)"
                  type="number"
                  value={settings.risk_management.stop_loss_percentage * 100}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      risk_management: {
                        ...settings.risk_management,
                        stop_loss_percentage: parseFloat(e.target.value) / 100,
                      },
                    })
                  }
                  inputProps={{ min: 0, max: 100, step: 0.1 }}
                />

                <TextField
                  fullWidth
                  label="Take Profit (%)"
                  type="number"
                  value={settings.risk_management.take_profit_percentage * 100}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      risk_management: {
                        ...settings.risk_management,
                        take_profit_percentage: parseFloat(e.target.value) / 100,
                      },
                    })
                  }
                  inputProps={{ min: 0, max: 100, step: 0.1 }}
                />

                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.risk_management.trailing_stop}
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          risk_management: {
                            ...settings.risk_management,
                            trailing_stop: e.target.checked,
                          },
                        })
                      }
                    />
                  }
                  label="Enable Trailing Stop"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Logging Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Logging Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Log Level</InputLabel>
                    <Select
                      value={settings.logging.level}
                      label="Log Level"
                      onChange={(e) =>
                        setSettings({
                          ...settings,
                          logging: { ...settings.logging, level: e.target.value },
                        })
                      }
                    >
                      <MenuItem value="DEBUG">Debug</MenuItem>
                      <MenuItem value="INFO">Info</MenuItem>
                      <MenuItem value="WARNING">Warning</MenuItem>
                      <MenuItem value="ERROR">Error</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.logging.to_file}
                        onChange={(e) =>
                          setSettings({
                            ...settings,
                            logging: { ...settings.logging, to_file: e.target.checked },
                          })
                        }
                      />
                    }
                    label="Log to File"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}
