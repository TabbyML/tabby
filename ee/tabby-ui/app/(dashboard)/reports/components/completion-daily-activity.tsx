'use client'

import { eachDayOfInterval } from 'date-fns'
import moment from 'moment'
import type { DateRange } from 'react-day-picker'
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'

import { DailyStatsQuery } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { Card, CardContent } from '@/components/ui/card'

function BarTooltip({
  active,
  payload,
  label
}: {
  active?: boolean
  label?: string
  payload?: {
    name: string
    payload: {
      views: number
      selects: number
      pendings: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { views, selects } = payload[0].payload
    if (!views) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completion:</span>
            <b>{views}</b>
          </p>
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Acceptance:</span>
            <b>{selects}</b>
          </p>
          <p className="text-muted-foreground">{label}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

export function DailyActivity({
  dailyStats,
  dateRange
}: {
  dailyStats?: DailyStatsQuery['dailyStats']
  dateRange: DateRange
}) {
  const { theme } = useCurrentTheme()
  const from = dateRange.from || new Date()
  const to = dateRange.to || from

  const dailyViewMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}

  dailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyViewMap[date] = dailyViewMap[date] || 0
    dailySelectMap[date] = dailySelectMap[date] || 0

    dailyViewMap[date] += stats.views
    dailySelectMap[date] += stats.selects
  }, {})

  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const chartData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const views = dailyViewMap[dateKey] || 0
    const selects = dailySelectMap[dateKey] || 0
    const pendings = views - selects
    return {
      name: moment(date).format('MMMM D'),
      views,
      selects,
      pendings
    }
  })
  return (
    <div className="rounded-lg border bg-primary-foreground/30 px-6 py-4">
      <h3 className="mb-5 text-sm font-medium tracking-tight">
        Daily Statistics
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          margin={{
            top: 5,
            right: 20,
            left: 20,
            bottom: 5
          }}
        >
          <Bar
            dataKey="selects"
            stackId="stats"
            fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
            radius={3}
          />
          <Bar
            dataKey="pendings"
            stackId="stats"
            fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
            radius={3}
          />
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12} width={20} allowDecimals={false} />
          <Tooltip cursor={{ fill: 'transparent' }} content={<BarTooltip />} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
