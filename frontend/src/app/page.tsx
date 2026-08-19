'use client'

import KPIGrid from '@/components/KPIGrid'
import EquityChart from '@/components/EquityChart'
import DrawdownChart from '@/components/DrawdownChart'
import RecentTrades from '@/components/RecentTrades'
import StrategyList from '@/components/StrategyList'
import LiveStatus from '@/components/LiveStatus'
import { useEffect, useState } from 'react'

export default function Home() {
  const [metricsData, setMetricsData] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch('/api/metrics')
        const result = await response.json()
        if (result.metrics && result.metrics.length > 0) {
          setMetricsData(result.metrics)
        }
      } catch (e) {
        console.error('Failed to fetch metrics:', e)
      } finally {
        setLoading(false)
      }
    }
    
    load()
    const interval = setInterval(load, 30000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Dashboard Overview</h2>
          <p className="text-gray-400 text-sm mt-1">
            Real-time trading metrics and performance analytics
            {metricsData.length > 0 && (
              <span className="ml-2 text-green-400">• Live</span>
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <select className="px-4 py-2 rounded-xl glass text-sm focus:outline-none focus:border-primary">
            <option>Last 24 Hours</option>
            <option>Last 7 Days</option>
            <option>Last 30 Days</option>
            <option>All Time</option>
          </select>
          <button className="px-4 py-2 rounded-xl bg-primary hover:bg-primary/90 transition-colors text-sm font-medium flex items-center gap-2">
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Export
          </button>
        </div>
      </div>
      
      {/* KPI Cards */}
      <KPIGrid />
      
      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <EquityChart 
            data={metricsData.map(m => ({
              time: m.timestamp,
              equity: m.equity,
            }))}
            timeRange="24h"
          />
        </div>
        <div>
          <DrawdownChart />
        </div>
      </div>
      
      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <RecentTrades limit={10} />
        </div>
        <div>
          <StrategyList limit={5} />
        </div>
      </div>
      
      {/* Live Events */}
      <LiveStatus />
    </div>
  )
}
