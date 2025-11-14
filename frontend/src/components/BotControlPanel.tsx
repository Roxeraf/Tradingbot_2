import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Play, Square, RefreshCw } from 'lucide-react'
import { botApi } from '../api/client'

export default function BotControlPanel() {
  const queryClient = useQueryClient()
  const [isLoading, setIsLoading] = useState(false)

  const { data: status } = useQuery({
    queryKey: ['botStatus'],
    queryFn: async () => {
      const response = await botApi.getStatus()
      return response.data
    },
    refetchInterval: 3000, // Refresh every 3 seconds
  })

  const startMutation = useMutation({
    mutationFn: () => botApi.start(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['botStatus'] }),
  })

  const stopMutation = useMutation({
    mutationFn: () => botApi.stop(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['botStatus'] }),
  })

  const restartMutation = useMutation({
    mutationFn: () => botApi.restart(),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['botStatus'] }),
  })

  const handleStart = async () => {
    setIsLoading(true)
    try {
      await startMutation.mutateAsync()
    } finally {
      setIsLoading(false)
    }
  }

  const handleStop = async () => {
    setIsLoading(true)
    try {
      await stopMutation.mutateAsync()
    } finally {
      setIsLoading(false)
    }
  }

  const handleRestart = async () => {
    setIsLoading(true)
    try {
      await restartMutation.mutateAsync()
    } finally {
      setIsLoading(false)
    }
  }

  const isRunning = status?.status === 'running'
  const isStopped = status?.status === 'stopped'

  return (
    <div className="space-y-3">
      {/* Status Badge */}
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700">Bot Status</span>
        <span
          className={`badge ${
            isRunning
              ? 'badge-success'
              : isStopped
              ? 'badge-danger'
              : 'badge-warning'
          }`}
        >
          {status?.status || 'unknown'}
        </span>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-2">
        <button
          onClick={handleStart}
          disabled={isRunning || isLoading}
          className="btn btn-success flex-1 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          <Play size={16} />
          Start
        </button>
        <button
          onClick={handleStop}
          disabled={isStopped || isLoading}
          className="btn btn-danger flex-1 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          <Square size={16} />
          Stop
        </button>
        <button
          onClick={handleRestart}
          disabled={isLoading}
          className="btn btn-secondary flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed text-sm p-2"
          title="Restart"
        >
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Uptime */}
      {status?.uptime_seconds && (
        <div className="text-xs text-gray-500 text-center">
          Uptime: {Math.floor(status.uptime_seconds / 3600)}h{' '}
          {Math.floor((status.uptime_seconds % 3600) / 60)}m
        </div>
      )}
    </div>
  )
}
