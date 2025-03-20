'use client'

import { eachDayOfInterval } from 'date-fns'
import { sum } from 'lodash-es'
import moment from 'moment'
import numeral from 'numeral'
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip
} from 'recharts'

import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import {
  useChatDailyStats,
  useCompletionDailyStats
} from '@/lib/hooks/use-statistics'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

import { AnimationWrapper } from './animation-wrapper'

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

function ChatBarTooltip({
  active,
  payload
}: {
  active?: boolean
  payload?: {
    name: string
    payload: {
      name: string
      chats: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { chats, name } = payload[0].payload
    if (!chats) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Chats:</span>
            <b>{chats}</b>
          </p>
          <p className="text-muted-foreground">{name}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

export function DailyCharts({
  from,
  to,
  sample,
  userId
}: {
  from: Date
  to: Date
  sample?: boolean
  userId: string | undefined
}) {
  const { theme } = useCurrentTheme()

  const { completionDailyStats, completionChartData } = useCompletionDailyStats(
    {
      dateRange: {
        from,
        to
      },
      sample,
      selectedMember: userId
    }
  )

  const { chatChartData, totalCount: totalChats } = useChatDailyStats({
    dateRange: {
      from,
      to
    },
    sample,
    selectedMember: userId
  })

  const totalViews = sum(completionChartData?.map(stats => stats.views))
  const totalAccepts = sum(completionDailyStats?.map(stats => stats.selects))

  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const dailyViewMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}
  completionDailyStats?.forEach(stats => {
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
  const completionViewData = daysBetweenRange.map(date => {
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

  const chatData = chatChartData?.map(x => {
    return {
      ...x,
      chatsPlaceholder: x.chats === 0 ? 0.5 : 0
    }
  })

  return (
    <div className="flex w-full flex-col items-center justify-center space-y-5 md:flex-row md:space-x-4 md:space-y-0 xl:justify-start">
      <AnimationWrapper
        viewport={{
          amount: 0.1
        }}
        delay={0.15}
        className="flex-1 self-stretch"
      >
        <Card className="flex flex-col justify-between self-stretch rounded-2xl bg-transparent pb-4">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 px-4 pb-1 pt-4">
            <CardTitle className="text-base font-medium tracking-normal">
              Acceptance Rate
            </CardTitle>
          </CardHeader>
          <CardContent className="mb-1 px-4 py-0">
            <div
              className="text-xl font-semibold"
              style={{ fontFamily: 'var(--font-montserrat)' }}
            >
              {averageAcceptance}%
            </div>
          </CardContent>
          <ResponsiveContainer width="100%" height={68}>
            <LineChart
              data={acceptRateData}
              margin={{ top: 10, right: 20, left: 15, bottom: 5 }}
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
      </AnimationWrapper>

      <AnimationWrapper
        viewport={{
          amount: 0.1
        }}
        delay={0.2}
        className="flex-1 self-stretch"
      >
        <Card className="flex flex-col justify-between self-stretch bg-transparent pb-4">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 px-4 pb-1 pt-4">
            <CardTitle className="text-base font-medium tracking-normal">
              Completions
            </CardTitle>
          </CardHeader>
          <CardContent className="mb-1 px-4 py-0">
            <div
              className="text-xl font-semibold"
              style={{ fontFamily: 'var(--font-montserrat)' }}
            >
              {numeral(totalViews).format('0,0')}
            </div>
          </CardContent>
          <ResponsiveContainer width="100%" height={68}>
            <BarChart
              data={completionViewData}
              margin={{
                top: totalViews === 0 ? 30 : 5,
                right: 15,
                left: 15,
                bottom: 0
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
      </AnimationWrapper>
      <AnimationWrapper
        viewport={{
          amount: 0.1
        }}
        delay={0.25}
        className="flex-1 self-stretch"
      >
        <Card className="flex flex-col justify-between self-stretch bg-transparent pb-4">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 px-4 pb-1 pt-4">
            <CardTitle className="text-base font-medium tracking-normal">
              Chats
            </CardTitle>
          </CardHeader>
          <CardContent className="mb-1 px-4 py-0">
            <div
              className="text-xl font-semibold"
              style={{ fontFamily: 'var(--font-montserrat)' }}
            >
              {numeral(totalChats).format('0,0')}
            </div>
          </CardContent>
          <ResponsiveContainer width="100%" height={68}>
            <BarChart
              data={chatData}
              margin={{
                top: totalViews === 0 ? 30 : 5,
                right: 15,
                left: 15,
                bottom: 0
              }}
            >
              <Bar
                dataKey="chats"
                stackId="stats"
                fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                radius={3}
              />
              <Bar
                dataKey="chatsPlaceholder"
                stackId="stats"
                fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
                radius={3}
              />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<ChatBarTooltip />}
              />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </AnimationWrapper>
    </div>
  )
}
