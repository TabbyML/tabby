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

import { DailyStatsQuery } from '@/lib/gql/generates/graphql'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

function LineTooltip({
  active,
  payload
}: {
  active?: boolean
  payload?: {
    name: string
    payload: {
      name: string
      selects: number
      value: string
      views: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { value, views, name } = payload[0].payload
    if (!views) return null
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
      views: number
      selects: number
      pendings: number
    }
  }[]
  type: 'accept' | 'view' | 'all'
}) {
  if (active && payload && payload.length) {
    const { views, selects, name } = payload[0].payload
    if (!views) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          {(type === 'view' || type === 'all') && (
            <p className="flex items-center">
              <span className="mr-3 inline-block w-20">Completions:</span>
              <b>{views}</b>
            </p>
          )}
          {(type === 'accept' || type === 'all') && (
            <p className="flex items-center">
              <span className="mr-3 inline-block w-20">Acceptances:</span>
              <b>{selects}</b>
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
  dailyStats
}: {
  from: Date
  to: Date
  dailyStats?: DailyStatsQuery['dailyStats']
}) {
  const { theme } = useTheme()
  const totalViews = sum(dailyStats?.map(stats => stats.views))
  const totalAccepts = sum(dailyStats?.map(stats => stats.selects))
  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const dailyViewMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}
  dailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyViewMap[date] = dailyViewMap[date] || 0
    dailySelectMap[date] = dailySelectMap[date] || 0

    dailyViewMap[date] += stats.views
    dailySelectMap[date] += stats.selects
  }, {})

  const averageAcceptance =
    totalViews === 0 ? 0 : ((totalAccepts / totalViews) * 100).toFixed(2)
  const acceptRateData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const views = dailyViewMap[dateKey] || 0
    const selects = dailySelectMap[dateKey] || 0
    return {
      name: moment(date).format('MMMM D'),
      value: views === 0 ? 0 : parseFloat(((selects / views) * 100).toFixed(2)),
      selects,
      views
    }
  })
  const viewData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const views = dailyViewMap[dateKey] || 0
    const selects = dailySelectMap[dateKey] || 0
    const pendings = views - selects
    return {
      name: moment(date).format('MMMM D'),
      views,
      selects,
      pending: views === 0 ? 0.5 : pendings,
      realPending: views === 0 ? 0 : pendings,
      viewPlaceholder: views === 0 ? 0.5 : 0,
      selectPlaceholder: selects === 0 ? 0.5 : 0
    }
  })
  return (
    <div>
      <div className="flex w-full flex-col items-center justify-center space-y-5 md:flex-row md:space-x-6 md:space-y-0 xl:justify-start">
        <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 md:block">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
            <CardTitle className="text-base font-normal tracking-tight">
              Acceptance Rate
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
              {numeral(totalViews).format('0,0')}
            </div>
          </CardContent>

          <ResponsiveContainer width="100%" height={60}>
            <BarChart
              data={viewData}
              margin={{
                top: totalViews === 0 ? 40 : 5,
                right: 20,
                left: 20,
                bottom: 5
              }}
            >
              <Bar
                dataKey="views"
                stackId="stats"
                fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                radius={3}
              />
              <Bar
                dataKey="viewPlaceholder"
                stackId="stats"
                fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
                radius={3}
              />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<BarTooltip type="view" />}
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
              data={viewData}
              margin={{
                top: totalViews === 0 ? 40 : 5,
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
