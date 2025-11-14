import type { Position } from '../types'
import { format } from 'date-fns'

interface Props {
  positions: Position[]
}

export default function PositionsTable({ positions }: Props) {
  if (positions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No open positions
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
              Amount
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Entry
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Current
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              P&L
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              P&L %
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {positions.map((position, idx) => (
            <tr key={idx}>
              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {position.symbol}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <span
                  className={`badge ${
                    position.side === 'long' ? 'badge-success' : 'badge-danger'
                  }`}
                >
                  {position.side}
                </span>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                {position.amount.toFixed(6)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                ${position.entry_price.toFixed(2)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-900">
                ${position.current_price.toFixed(2)}
              </td>
              <td
                className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${
                  position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                ${position.unrealized_pnl.toFixed(2)}
              </td>
              <td
                className={`px-6 py-4 whitespace-nowrap text-sm text-right font-medium ${
                  position.unrealized_pnl_percentage >= 0
                    ? 'text-green-600'
                    : 'text-red-600'
                }`}
              >
                {position.unrealized_pnl_percentage.toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
