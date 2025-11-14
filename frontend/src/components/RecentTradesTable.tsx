import type { Trade } from '../types'
import { format } from 'date-fns'

interface Props {
  trades: Trade[]
}

export default function RecentTradesTable({ trades }: Props) {
  if (trades.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No trades yet
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Symbol
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Side
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Entry
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Exit
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              P&L
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Time
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {trades.map((trade, idx) => (
            <tr key={trade.id || idx}>
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {trade.symbol}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span
                  className={`badge ${
                    trade.side === 'buy' ? 'badge-success' : 'badge-danger'
                  }`}
                >
                  {trade.side}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                ${trade.entry_price.toFixed(2)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                {trade.exit_price ? `$${trade.exit_price.toFixed(2)}` : '-'}
              </td>
              <td
                className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${
                  (trade.pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {trade.pnl ? `$${trade.pnl.toFixed(2)}` : '-'}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {format(new Date(trade.entry_time), 'MMM dd, HH:mm')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
