/**
 * WebSocket hook for real-time updates
 */
import { useEffect, useRef, useState, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws/live'

interface WebSocketMessage {
  type: string
  [key: string]: any
}

export const useWebSocket = () => {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const ws = useRef<WebSocket | null>(null)
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    try {
      ws.current = new WebSocket(WS_URL)

      ws.current.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        // Clear any reconnection timeout
        if (reconnectTimeout.current) {
          clearTimeout(reconnectTimeout.current)
          reconnectTimeout.current = null
        }
      }

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setLastMessage(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.current.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)

        // Attempt to reconnect after 5 seconds
        reconnectTimeout.current = setTimeout(() => {
          console.log('Attempting to reconnect WebSocket...')
          connect()
        }, 5000)
      }
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }, [])

  const disconnect = useCallback(() => {
    if (ws.current) {
      ws.current.close()
      ws.current = null
    }
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current)
      reconnectTimeout.current = null
    }
  }, [])

  const sendMessage = useCallback((data: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    } else {
      console.warn('WebSocket is not connected')
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    sendMessage,
    reconnect: connect,
  }
}

export default useWebSocket
