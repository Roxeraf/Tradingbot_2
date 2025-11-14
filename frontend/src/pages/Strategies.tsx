import { useQuery } from '@tanstack/react-query'
import { strategyApi } from '../api/client'

export default function Strategies() {
  const { data: strategies } = useQuery({
    queryKey: ['strategies'],
    queryFn: async () => {
      const response = await strategyApi.listStrategies()
      return response.data
    },
  })

  const { data: currentStrategy } = useQuery({
    queryKey: ['currentStrategy'],
    queryFn: async () => {
      const response = await strategyApi.getCurrentStrategy()
      return response.data
    },
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Strategies</h1>
        <p className="mt-2 text-sm text-gray-600">
          Manage and configure trading strategies
        </p>
      </div>

      {/* Current Strategy */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Current Strategy
        </h2>
        {currentStrategy ? (
          <div className="space-y-3">
            <div>
              <span className="text-sm font-medium text-gray-500">Name:</span>
              <span className="ml-2 text-sm text-gray-900">
                {currentStrategy.name}
              </span>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-500">
                Description:
              </span>
              <p className="mt-1 text-sm text-gray-900">
                {currentStrategy.description || 'No description available'}
              </p>
            </div>
            <div>
              <span className="text-sm font-medium text-gray-500">
                Parameters:
              </span>
              <pre className="mt-1 text-sm text-gray-900 bg-gray-50 p-3 rounded">
                {JSON.stringify(currentStrategy.parameters, null, 2)}
              </pre>
            </div>
          </div>
        ) : (
          <p className="text-gray-500">Loading...</p>
        )}
      </div>

      {/* Available Strategies */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Available Strategies
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {strategies?.map((strategy) => (
            <div
              key={strategy}
              className="border rounded-lg p-4 hover:border-primary-500 cursor-pointer transition-colors"
            >
              <h3 className="font-medium text-gray-900">{strategy}</h3>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
