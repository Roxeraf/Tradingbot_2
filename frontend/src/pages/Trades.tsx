import { useQuery } from '@tanstack/react-query'
import { tradingApi } from '../api/client'
import RecentTradesTable from '../components/RecentTradesTable'

export default function Trades() {
  const { data: trades } = useQuery({
    queryKey: ['allTrades'],
    queryFn: async () => {
      const response = await tradingApi.getTrades(undefined, 100)
      return response.data
    },
    refetchInterval: 10000,
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Trade History</h1>
        <p className="mt-2 text-sm text-gray-600">
          View all executed trades and their performance
        </p>
      </div>

      <div className="card">
        <RecentTradesTable trades={trades || []} />
      </div>
    </div>
  )
}
