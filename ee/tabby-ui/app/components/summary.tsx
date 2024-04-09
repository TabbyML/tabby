'use client'

import { eachDayOfInterval } from 'date-fns'
import { maxBy, mean, sum } from 'lodash-es'
import moment from 'moment'
import { useTheme } from 'next-themes'
import numeral from 'numeral'
import {
  Bar,
  BarChart,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type LabelProps
} from 'recharts'

import { Language } from '@/lib/gql/generates/graphql'
import type { DailyStats } from '@/lib/types/stats'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export type LanguageStats = Record<
  Language,
  {
    selects: number
    completions: number
    name: string
  }
>

function BarTooltip({
  active,
  payload
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
}) {
  if (active && payload && payload.length) {
    const { completion, select, name } = payload[0].payload
    if (!completion) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completion:</span>
            <b>{completion}</b>
          </p>
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Acceptance:</span>
            <b>{select}</b>
          </p>
          <p className="text-muted-foreground">{name}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

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
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { value, select, name } = payload[0].payload
    if (!select) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Acceptance:</span>
            <b>{value}%</b>
          </p>
          <p className="text-muted-foreground">{name}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

const LanguageLabel: React.FC<
  LabelProps & { languageData: LanguageData }
> = props => {
  const { x, y, width, height, value, languageData } = props
  const myLanguageData = languageData.find(data => data.label === value)

  if (!myLanguageData || myLanguageData.selects === 0) {
    return null
  }

  const padding = 5
  return (
    <text
      x={+x! + +width! + padding}
      y={+y! + +height! / 2}
      fill="#666"
      fontSize={10}
      fontWeight="bold"
      textAnchor="start"
      dominantBaseline="middle"
    >
      {value}
    </text>
  )
}

type LanguageData = {
  name: string
  selects: number
  completions: number
  label: string
}[]

export function Summary({
  from,
  to,
  dailyStats,
  dateRange,
  languageStats
}: {
  from: Date
  to: Date
  dailyStats: DailyStats[] | undefined
  dateRange: number
  languageStats: LanguageStats
}) {
  const { theme } = useTheme()
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  const averageAcceptance = (
    mean(dailyStats?.map(stats => stats.selects / stats.completions)) * 100
  ).toFixed(2)
  const daysBetweenRange = eachDayOfInterval({
    start: from,
    end: to
  })

  const dailyCompletionMap: Record<string, number> = {}
  const dailySelectMap: Record<string, number> = {}
  dailyStats?.forEach(stats => {
    const date = moment(stats.start).format('YYYY-MM-DD')
    dailyCompletionMap[date] = stats.completions
    dailySelectMap[date] = stats.selects
  }, {})

  const completionData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const completion = dailyCompletionMap[dateKey] || 0
    const select = dailySelectMap[dateKey] || 0
    const pending = completion - select
    return {
      name: moment(date).format('D MMM'),
      completion,
      select,
      pending: completion === 0 ? 0.5 : pending
    }
  })
  const acceptanceData = daysBetweenRange.map(date => {
    const dateKey = moment(date).format('YYYY-MM-DD')
    const completion = dailyCompletionMap[dateKey] || 0
    const select = dailySelectMap[dateKey] || 0
    return {
      name: moment(date).format('D MMM'),
      value: completion === 0 ? 0 : ((select / completion) * 100).toFixed(2),
      select
    }
  })
  const languageData: LanguageData = Object.entries(languageStats).map(
    ([key, stats]) => {
      return {
        name: key,
        selects: stats.selects,
        completions: stats.completions,
        label: stats.name
      }
    }
  )
  const mostAcceptedLanguage = maxBy(languageData, data => data.selects)

  return (
    <div className="flex w-full flex-col items-center justify-center space-y-5 md:flex-row md:space-x-6 md:space-y-0 xl:justify-start">
      <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 dark:bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
          <CardTitle className="text-base font-normal tracking-tight">
            Total Completions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {totalCompletions > 0 && '+'}
            {numeral(totalCompletions).format('0,0')}
          </div>
          <p className="text-xs text-muted-foreground">
            In last {dateRange} days
          </p>
        </CardContent>

        <ResponsiveContainer width="100%" height={50}>
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
              dataKey="pending"
              stackId="stats"
              fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
              radius={3}
            />
            <Tooltip
              cursor={{ fill: 'transparent' }}
              content={<BarTooltip />}
            />
          </BarChart>
        </ResponsiveContainer>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 dark:bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
          <CardTitle className="text-base font-normal tracking-tight">
            Average Acceptance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {totalCompletions === 0 ? '0' : `${averageAcceptance}%`}
          </div>
          <p className="text-xs text-muted-foreground">
            In last {dateRange} days
          </p>
        </CardContent>

        <ResponsiveContainer width="100%" height={50}>
          <LineChart
            data={acceptanceData}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
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

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-transparent pb-6 dark:bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-1">
          <CardTitle className="text-base font-normal tracking-tight">
            Language Acceptance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {totalCompletions === 0 ? 'None' : mostAcceptedLanguage?.label}
          </div>
          <p className="text-xs text-muted-foreground">
            Most autocomplete accepts
          </p>
        </CardContent>

        <ResponsiveContainer height={50} width={300}>
          <BarChart
            layout="vertical"
            data={languageData}
            barCategoryGap={2}
            margin={{ top: 5, right: 80, left: 20, bottom: 5 }}
          >
            <Bar
              dataKey="selects"
              fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
              radius={3}
            >
              <LabelList
                dataKey="label"
                content={<LanguageLabel languageData={languageData} />}
              />
            </Bar>
            <XAxis type="number" hide />
            <YAxis type="category" width={100} dataKey="name" hide />
          </BarChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}
