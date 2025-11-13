import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Trading from './pages/Trading'
import Strategies from './pages/Strategies'
import Backtesting from './pages/Backtesting'
import Settings from './pages/Settings'
import Logs from './pages/Logs'

function App() {
  return (
    <Box sx={{ display: 'flex' }}>
      <Layout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/trading" element={<Trading />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/backtest" element={<Backtesting />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </Layout>
    </Box>
  )
}

export default App
