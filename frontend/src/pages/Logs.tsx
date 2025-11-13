import { useState } from 'react'
import { useQuery } from 'react-query'
import {
  Box,
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
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  CircularProgress,
  Alert,
  Button,
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  BugReport as DebugIcon,
  Info as InfoIcon,
  Warning as WarningIcon,
  Error as ErrorIcon,
} from '@mui/icons-material'
import { tradingApi } from '../services/api'
import type { LogEntry } from '../types'

const getLevelIcon = (level: string) => {
  switch (level) {
    case 'DEBUG':
      return <DebugIcon fontSize="small" color="disabled" />
    case 'INFO':
      return <InfoIcon fontSize="small" color="info" />
    case 'WARNING':
      return <WarningIcon fontSize="small" color="warning" />
    case 'ERROR':
      return <ErrorIcon fontSize="small" color="error" />
    default:
      return null
  }
}

const getLevelColor = (level: string): 'default' | 'info' | 'warning' | 'error' => {
  switch (level) {
    case 'INFO':
      return 'info'
    case 'WARNING':
      return 'warning'
    case 'ERROR':
      return 'error'
    default:
      return 'default'
  }
}

export default function Logs() {
  const [filterLevel, setFilterLevel] = useState<string>('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [limit] = useState(100)

  // Fetch logs
  const { data: logsData, isLoading, refetch } = useQuery<{ data: LogEntry[] }>(
    ['logs', filterLevel],
    () => tradingApi.getLogs(filterLevel === 'ALL' ? undefined : filterLevel, limit),
    { refetchInterval: 5000 }
  )

  const logs = logsData?.data || []

  // Filter logs by search query
  const filteredLogs = logs.filter(
    (log) =>
      log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (log.module && log.module.toLowerCase().includes(searchQuery.toLowerCase()))
  )

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">System Logs</Typography>
        <Button variant="outlined" startIcon={<RefreshIcon />} onClick={() => refetch()}>
          Refresh
        </Button>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" gap={2} flexWrap="wrap">
            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Log Level</InputLabel>
              <Select
                value={filterLevel}
                label="Log Level"
                onChange={(e) => setFilterLevel(e.target.value)}
              >
                <MenuItem value="ALL">All Levels</MenuItem>
                <MenuItem value="DEBUG">Debug</MenuItem>
                <MenuItem value="INFO">Info</MenuItem>
                <MenuItem value="WARNING">Warning</MenuItem>
                <MenuItem value="ERROR">Error</MenuItem>
              </Select>
            </FormControl>

            <TextField
              label="Search"
              variant="outlined"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search logs..."
              sx={{ flexGrow: 1, minWidth: 200 }}
            />
          </Box>

          <Box display="flex" gap={1} mt={2}>
            <Chip
              icon={<DebugIcon />}
              label={`Debug: ${logs.filter((l) => l.level === 'DEBUG').length}`}
              size="small"
            />
            <Chip
              icon={<InfoIcon />}
              label={`Info: ${logs.filter((l) => l.level === 'INFO').length}`}
              size="small"
              color="info"
            />
            <Chip
              icon={<WarningIcon />}
              label={`Warning: ${logs.filter((l) => l.level === 'WARNING').length}`}
              size="small"
              color="warning"
            />
            <Chip
              icon={<ErrorIcon />}
              label={`Error: ${logs.filter((l) => l.level === 'ERROR').length}`}
              size="small"
              color="error"
            />
          </Box>
        </CardContent>
      </Card>

      {/* Logs Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Recent Logs ({filteredLogs.length})
          </Typography>
          {isLoading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : filteredLogs.length === 0 ? (
            <Alert severity="info">No logs found matching your filters.</Alert>
          ) : (
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 600 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell width="50px">Level</TableCell>
                    <TableCell width="180px">Timestamp</TableCell>
                    <TableCell width="150px">Module</TableCell>
                    <TableCell>Message</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredLogs.map((log) => (
                    <TableRow
                      key={log.id}
                      sx={{
                        '&:hover': { backgroundColor: 'action.hover' },
                        backgroundColor:
                          log.level === 'ERROR'
                            ? 'error.light'
                            : log.level === 'WARNING'
                            ? 'warning.light'
                            : 'inherit',
                      }}
                    >
                      <TableCell>
                        <Chip
                          icon={getLevelIcon(log.level)}
                          label={log.level}
                          size="small"
                          color={getLevelColor(log.level)}
                          sx={{ minWidth: 80 }}
                        />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" fontFamily="monospace">
                          {new Date(log.timestamp).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {log.module || '-'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography
                          variant="body2"
                          fontFamily="monospace"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                          }}
                        >
                          {log.message}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>
    </Box>
  )
}
