'use client'

import { eachDayOfInterval } from 'date-fns'
import { sum } from 'lodash-es'
import moment from 'moment'
import { useTheme } from 'next-themes'
import numeral from 'numeral'
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip
} from 'recharts'

import { DailyStatsQuery, Language } from '@/lib/gql/generates/graphql'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export type LanguageStats = Record<
  Language,
  {
    selects: number
    completions: number
    name: Language
  }
>

function LineTooltip({
  active,
  payload
}: {
  active?: boolean
  payload?: {
    name: string
    payload: {
      name: string
      select: number
      value: string
      completion: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { value, completion, name } = payload[0].payload
    if (!completion) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Rate:</span>
            <b>{value}%</b>
          </p>
          <p className="text-muted-foreground">{name}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

function BarTooltip({
  active,
  payload,
  type
}: {
  active?: boolean
  payload?: {
    name: string
    payload: {
      name: string
      completion: number
      select: number
      pending: number
    }
  }[]
  type: 'accept' | 'completion' | 'all'
}) {
  if (active && payload && payload.length) {
    const { completion, select, name } = payload[0].payload
    if (!completion) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          {(type === 'completion' || type === 'all') && (
            <p className="flex items-center">
              <span className="mr-3 inline-block w-20">Completions:</span>
              <b>{completion}</b>
            </p>
          )}
          {(type === 'accept' || type === 'all') && (
            <p className="flex items-center">
              <span className="mr-3 inline-block w-20">Acceptances:</span>
              <b>{select}</b>
            </p>
          )}
          <p className="text-muted-foreground">{name}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

export function CompletionCharts({
  from,
  to,
  dailyStats,
  dateRange
}: {
  from: Date
  to: Date
  dailyStats?: DailyStatsQuery['dailyStats']
  dateRange: number
}) {
  const { theme } = useTheme()
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  const totalAccepts = sum(dailyStats?.map(stats => stats.selects))
  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  // Mapping data of { date: amount }
  const dailyCompletionMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}
  dailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyCompletionMap[date] = stats.completions
    dailySelectMap[date] = stats.selects
  }, {})

  // Data for charts
  const averageAcceptance =
    totalCompletions === 0
      ? 0
      : ((totalAccepts / totalCompletions) * 100).toFixed(2)
  const acceptRateData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const completion = dailyCompletionMap[dateKey] || 0
    const select = dailySelectMap[dateKey] || 0
    return {
      name: moment(date).format('D MMM'),
      value: completion === 0 ? 0 : ((select / completion) * 100).toFixed(2),
      select,
      completion
    }
  })
  const completionData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const completion = dailyCompletionMap[dateKey] || 0
    const select = dailySelectMap[dateKey] || 0
    const pending = completion - select
    return {
      name: moment(date).format('D MMM'),
      completion,
      select,
      pending: completion === 0 ? 0.5 : pending,
      realPending: completion === 0 ? 0 : pending,
      completionPlaceholder: completion === 0 ? 0.5 : 0,
      selectPlaceholder: select === 0 ? 0.5 : 0
    }
  })

  return (
    <div>
      <div className="flex w-full flex-col items-center justify-center space-y-5 md:flex-row md:space-x-6 md:space-y-0 xl:justify-start">
        <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 md:block">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
            <CardTitle className="text-base font-normal tracking-tight">
              Accept Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{averageAcceptance}%</div>
          </CardContent>

          <ResponsiveContainer width="100%" height={60}>
            <LineChart
              data={acceptRateData}
              margin={{ top: 15, right: 30, left: 20, bottom: 5 }}
            >
              <Line
                type="monotone"
                dataKey="value"
                stroke={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                strokeWidth={1.5}
              />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<LineTooltip />}
              />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 md:block">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
            <CardTitle className="text-base font-normal tracking-tight">
              Completions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {numeral(totalCompletions).format('0,0')}
            </div>
          </CardContent>

          <ResponsiveContainer width="100%" height={60}>
            <BarChart
              data={completionData}
              margin={{
                top: totalCompletions === 0 ? 40 : 5,
                right: 20,
                left: 20,
                bottom: 5
              }}
            >
              <Bar
                dataKey="completion"
                stackId="stats"
                fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                radius={3}
              />
              <Bar
                dataKey="completionPlaceholder"
                stackId="stats"
                fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
                radius={3}
              />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<BarTooltip type="completion" />}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 md:block">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
            <CardTitle className="text-base font-normal tracking-tight">
              Acceptances
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {numeral(totalAccepts).format('0,0')}
            </div>
          </CardContent>

          <ResponsiveContainer width="100%" height={60}>
            <BarChart
              data={completionData}
              margin={{
                top: totalCompletions === 0 ? 40 : 5,
                right: 20,
                left: 20,
                bottom: 5
              }}
            >
              <Bar
                dataKey="select"
                stackId="stats"
                fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                radius={3}
              />
              <Bar
                dataKey="selectPlaceholder"
                stackId="stats"
                fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
                radius={3}
              />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<BarTooltip type="accept" />}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  )
}
