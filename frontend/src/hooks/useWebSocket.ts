'use client'

import { useEffect, useRef, useState } from 'react'

interface WSProps {
  onEvent?: (event: any) => void
}

export default function useWebSocket({ onEvent }: WSProps = {}) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      console.log('[WS] Connected')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (onEvent) onEvent(data)
        console.log('[WS] Event:', data.type)
      } catch (e) {
        console.error('[WS] Parse error:', e)
      }
    }

    ws.onclose = () => {
      setConnected(false)
      console.log('[WS] Disconnected, reconnecting...')
      reconnectTimer.current = setTimeout(() => {
        console.log('[WS] Reconnecting...')
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('[WS] Error:', error)
    }

    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      ws.close()
    }
  }, [onEvent])

  const send = (message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    }
  }

  const ping = () => {
    send({ type: 'ping' })
  }

  return { connected, send, ping }
}
