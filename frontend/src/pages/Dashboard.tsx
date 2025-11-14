import { useQuery } from '@tanstack/react-query'
import { TrendingUp, TrendingDown, DollarSign, Target } from 'lucide-react'
import { tradingApi, botApi } from '../api/client'
import PerformanceChart from '../components/PerformanceChart'
import PositionsTable from '../components/PositionsTable'
import RecentTradesTable from '../components/RecentTradesTable'

export default function Dashboard() {
  const { data: status } = useQuery({
    queryKey: ['botStatus'],
    queryFn: async () => {
      const response = await botApi.getStatus()
      return response.data
    },
    refetchInterval: 5000,
  })

  const { data: performance } = useQuery({
    queryKey: ['performance'],
    queryFn: async () => {
      const response = await tradingApi.getPerformance()
      return response.data
    },
    refetchInterval: 10000,
  })

  const { data: positions } = useQuery({
    queryKey: ['positions'],
    queryFn: async () => {
      const response = await tradingApi.getPositions()
      return response.data
    },
    refetchInterval: 5000,
  })

  const { data: trades } = useQuery({
    queryKey: ['trades'],
    queryFn: async () => {
      const response = await tradingApi.getTrades(undefined, 10)
      return response.data
    },
    refetchInterval: 10000,
  })

  const stats = [
    {
      name: 'Total P&L',
      value: performance?.total_pnl
        ? `$${performance.total_pnl.toFixed(2)}`
        : '$0.00',
      change: performance?.total_pnl_percentage
        ? `${performance.total_pnl_percentage.toFixed(2)}%`
        : '0%',
      icon: DollarSign,
      positive: (performance?.total_pnl || 0) >= 0,
    },
    {
      name: 'Win Rate',
      value: performance?.win_rate ? `${performance.win_rate.toFixed(1)}%` : '0%',
      change: `${performance?.winning_trades || 0} / ${performance?.total_trades || 0}`,
      icon: Target,
      positive: (performance?.win_rate || 0) >= 50,
    },
    {
      name: 'Total Trades',
      value: performance?.total_trades || 0,
      change: `Open: ${status?.open_positions || 0}`,
      icon: TrendingUp,
      positive: true,
    },
    {
      name: 'Current Equity',
      value: performance?.current_equity
        ? `$${performance.current_equity.toFixed(2)}`
        : '$0.00',
      change: `Initial: $${performance?.initial_equity || 0}`,
      icon: DollarSign,
      positive: (performance?.current_equity || 0) >= (performance?.initial_equity || 0),
    },
  ]

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-sm text-gray-600">
          Monitor your trading bot performance and active positions
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <div key={stat.name} className="card">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                <p className="mt-2 text-3xl font-semibold text-gray-900">
                  {stat.value}
                </p>
                <p
                  className={`mt-1 text-sm ${
                    stat.positive ? 'text-green-600' : 'text-red-600'
                  }`}
                >
                  {stat.change}
                </p>
              </div>
              <div
                className={`rounded-full p-3 ${
                  stat.positive ? 'bg-green-100' : 'bg-red-100'
                }`}
              >
                <stat.icon
                  className={`h-6 w-6 ${
                    stat.positive ? 'text-green-600' : 'text-red-600'
                  }`}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Performance Chart */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Performance Chart
        </h2>
        <PerformanceChart />
      </div>

      {/* Active Positions */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Active Positions
        </h2>
        <PositionsTable positions={positions || []} />
      </div>

      {/* Recent Trades */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Trades
        </h2>
        <RecentTradesTable trades={trades || []} />
      </div>
    </div>
  )
}
