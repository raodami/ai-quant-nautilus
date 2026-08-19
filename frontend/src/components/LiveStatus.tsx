'use client'

import { useEffect, useState } from 'react'
import useWebSocket from '@/hooks/useWebSocket'

interface LiveStatusProps {
  className?: string
}

export default function LiveStatus({ className = '' }: LiveStatusProps) {
  const [events, setEvents] = useState<any[]>([])
  const [metrics, setMetrics] = useState<any>(null)

  const ws = useWebSocket({
    onEvent: (event) => {
      setEvents(prev => [event, ...prev].slice(0, 20))
      console.log('[Live] New event:', event.type)
    }
  })

  // Fetch metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const resp = await fetch('/api/metrics')
        const data = await resp.json()
        setMetrics(data.summary)
      } catch (e) {
        console.error('Failed to fetch metrics:', e)
      }
    }
    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className={`glass rounded-xl p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-300">Live Events</h3>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${ws.connected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-xs text-gray-400">{ws.connected ? 'Connected' : 'Reconnecting...'}</span>
        </div>
      </div>
      
      {events.length === 0 ? (
        <div className="text-center py-8 text-gray-500 text-sm">
          {ws.connected ? 'Waiting for events...' : 'Connecting...'}
        </div>
      ) : (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {events.map((event, i) => (
            <div 
              key={i} 
              className={`p-2 rounded-lg text-xs ${
                event.type === 'backtest_completed' 
                  ? 'bg-green-500/10 border border-green-500/20' 
                  : 'bg-primary/10 border border-primary/20'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium text-gray-200">
                  {event.type === 'backtest_completed' ? '✓ Backtest Done' : '📊 Strategy'}
                </span>
                <span className="text-gray-500">
                  {new Date().toLocaleTimeString()}
                </span>
              </div>
              {event.type === 'backtest_completed' && event.data && (
                <div className="mt-1 text-gray-400">
                  {event.data.strategyName} · {((event.data.totalReturn || 0) > 0 ? '+' : '')}{(event.data.totalReturn || 0).toFixed(2)}%
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
