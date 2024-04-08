'use client'

import { eachDayOfInterval } from 'date-fns'
import moment from 'moment'
import { sum } from 'lodash-es'
import { useTheme } from 'next-themes'

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'
import { Card, CardContent } from '@/components/ui/card'

import type { DateRange } from 'react-day-picker'
import type { DailyStats } from '../types/stats'

function BarTooltip({
  active,
  payload,
  label
}: {
  active?: boolean;
  label?: string;
  payload?: {
    name: string;
    payload: {
      completion: number;
      select: number;
      pending: number;
    };
    
  }[]
}) {
  if (active && payload && payload.length) {
    const {completion, select} = payload[0].payload
    if (!completion) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completion:</span><b>{completion}</b>
          </p>
          <p className="flex items-center">
          <span className="mr-3 inline-block w-20">Acceptance:</span><b>{select}</b>
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
  dailyStats: DailyStats[] | undefined
  dateRange: DateRange
}) {
  const { theme } = useTheme()
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  const from = dateRange.from || new Date()
  const to = dateRange.to || from

  const dailyCompletionMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}

  dailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyCompletionMap[date] = stats.completions
    dailySelectMap[date] = stats.selects
  }, {})

  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const chartData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const completion = dailyCompletionMap[dateKey] || 0
    const select = dailySelectMap[dateKey] || 0
    const pending = completion - select
    return {
      name: moment(date).format('D MMM'),
      completion,
      select,
      pending
    }
  })
  return (
    <div className="rounded-lg border bg-primary-foreground/30 px-6 py-4">
      <h3 className="mb-5 text-sm font-medium tracking-tight">Completions</h3>
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
          <Bar dataKey="select" stackId="stats" fill={theme === 'dark' ? '#e8e1d3' : '#54452c'} radius={3} />
          <Bar dataKey="pending" stackId="stats" fill={theme === 'dark' ? '#423929' : '#e8e1d3'} radius={3} />
          <XAxis dataKey="name" fontSize={12} />
          <YAxis fontSize={12} width={20} allowDecimals={false} />
          <Tooltip cursor={{ fill: 'transparent' }} content={<BarTooltip />} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
