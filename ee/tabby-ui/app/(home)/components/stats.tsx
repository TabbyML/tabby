'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { useWindowSize } from '@uidotdev/usehooks'
import { eachDayOfInterval } from 'date-fns'
import { sum } from 'lodash-es'
import moment from 'moment'
import { useTheme } from 'next-themes'
import ReactActivityCalendar from 'react-activity-calendar'
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type LabelProps
} from 'recharts'
import seedrandom from 'seedrandom'
import { useQuery } from 'urql'

import {
  DailyStatsInPastYearQuery,
  DailyStatsQuery,
  Language
} from '@/lib/gql/generates/graphql'
import { useMe } from '@/lib/hooks/use-me'
import { QueryVariables } from '@/lib/tabby/gql'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import languageColors from '../language-colors.json'
import { CompletionCharts, type LanguageStats } from './completion-charts'

const DATE_RANGE = 6

type LanguageData = {
  name: string
  selects: number
  completions: number
  label: string
}[]

// Find auto-completion stats of each language
function useLanguageStats({
  start,
  end,
  users
}: {
  start: Date
  end: Date
  users?: string
}) {
  const languages = Object.values(Language)
  const [lanIdx, setLanIdx] = useState(0)
  const [queryVariables, setQueryVariables] = useState<
    QueryVariables<typeof queryDailyStats>
  >({
    start: moment(start).utc().format(),
    end: moment(end).utc().format(),
    users,
    languages: languages[0]
  })
  const [languageStats, setLanguageStats] = useState<LanguageStats>(
    {} as LanguageStats
  )

  const [{ data, fetching }] = useQuery({
    query: queryDailyStats,
    variables: queryVariables
  })

  useEffect(() => {
    if (lanIdx >= languages.length) return
    if (!fetching && data?.dailyStats) {
      const language = languages[lanIdx]
      const newLanguageStats = { ...languageStats }
      newLanguageStats[language] = newLanguageStats[language] || {
        selects: 0,
        completions: 0,
        name: Object.keys(Language)[lanIdx]
      }
      newLanguageStats[language].selects += sum(
        data.dailyStats.map(stats => stats.selects)
      )
      newLanguageStats[language].completions += sum(
        data.dailyStats.map(stats => stats.completions)
      )

      const newLanIdx = lanIdx + 1
      setLanguageStats(newLanguageStats)
      setLanIdx(newLanIdx)
      if (newLanIdx < languages.length) {
        setQueryVariables({
          start: moment(start).utc().format(),
          end: moment(end).utc().format(),
          users,
          languages: languages[newLanIdx]
        })
      }
    }
  }, [queryVariables, lanIdx, fetching])

  return [languageStats]
}

const getLanguageColorMap = (): Record<string, string> => {
  return Object.entries(languageColors).reduce((acc, cur) => {
    const [lan, color] = cur
    return { ...acc, [lan.toLocaleLowerCase()]: color }
  }, {})
}

function ActivityCalendar({
  data
}: {
  data: {
    date: string
    count: number
    level: number
  }[]
}) {
  const { theme } = useTheme()
  const size = useWindowSize()
  const width = size.width || 0
  const blockSize =
    width >= 1300 ? 13 : width >= 1000 ? 8 : width >= 800 ? 6 : 9

  return (
    <ReactActivityCalendar
      data={data}
      colorScheme={theme === 'dark' ? 'dark' : 'light'}
      theme={{
        light: ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39'],
        dark: ['rgb(45, 51, 59)', '#0e4429', '#006d32', '#26a641', '#39d353']
      }}
      blockSize={blockSize}
      hideTotalCount
      showWeekdayLabels
    />
  )
}

const LanguageLabel: React.FC<
  LabelProps & { languageData: LanguageData; theme?: string }
> = props => {
  const { x, y, value, languageData, theme } = props
  const myLanguageData = languageData.find(data => data.label === value)

  if (!myLanguageData || myLanguageData.completions === 0) {
    return null
  }

  const padding = 5
  return (
    <text
      x={+x!}
      y={+y! - 7}
      fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
      fontSize={10}
      fontWeight="bold"
      textAnchor="start"
      dominantBaseline="middle"
    >
      {value}
    </text>
  )
}

function LanguageTooltip({
  active,
  payload
}: {
  active?: boolean
  payload?: {
    name: string
    payload: {
      label: string
      completions: number
      selects: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { completions, selects, label } = payload[0].payload
    const activities = completions + selects
    if (!activities) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completions:</span>
            <b>{completions}</b>
          </p>
          <p className="text-muted-foreground">{label}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

export default function Stats() {
  const [{ data }] = useMe()
  const { theme } = useTheme()
  const searchParams = useSearchParams()

  const sample = searchParams.get('sample') === 'true'
  const colorMap = getLanguageColorMap()
  const startDate = moment()
    .subtract(DATE_RANGE, 'day')
    .startOf('day')
    .utc()
    .format()
  const endDate = moment().endOf('day').utc().format()

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: startDate,
      end: endDate,
      users: data?.me?.id
    }
  })
  let dailyStats: DailyStatsQuery['dailyStats'] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: startDate,
      end: endDate
    })
    dailyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(moment(date).format('YYYY-MM-DD') + data?.me.id)
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 25)
      return {
        start: moment(date).startOf('day').toDate(),
        end: moment(date).endOf('day').toDate(),
        completions,
        selects
      }
    })
  } else {
    dailyStats = dailyStatsData?.dailyStats.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))
  }

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: data?.me?.id
    }
  })
  let lastYearCompletions = 0
  let yearlyStats: DailyStatsInPastYearQuery['dailyStatsInPastYear'] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: moment().toDate(),
      end: moment().subtract(365, 'days').toDate()
    })
    yearlyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(moment(date).format('YYYY-MM-DD') + data?.me.id)
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 10)
      return {
        start: moment(date).startOf('day').toDate(),
        end: moment(date).endOf('day').toDate(),
        completions,
        selects
      }
    })
  } else {
    yearlyStats = yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))
  }
  const dailyCompletionMap: Record<string, number> =
    yearlyStats?.reduce((acc, cur) => {
      const date = moment(cur.start).format('YYYY-MM-DD')
      lastYearCompletions += cur.completions
      return { ...acc, [date]: cur.completions }
    }, {}) || {}
  const activities = new Array(365)
    .fill('')
    .map((_, idx) => {
      const date = moment().subtract(idx, 'days').format('YYYY-MM-DD')
      const count = dailyCompletionMap[date] || 0
      const level = Math.min(4, Math.ceil(count / 5))
      return {
        date: date,
        count,
        level
      }
    })
    .reverse()

  // Query language stats
  const [languageStats] = useLanguageStats({
    start: moment(startDate).toDate(),
    end: moment(endDate).toDate(),
    users: data?.me?.id
  })
  let languageData: LanguageData | undefined
  if (sample) {
    const rng = seedrandom(data?.me.id)
    const rustCompletion = Math.ceil(rng() * 40)
    const pythonCompletion = Math.ceil(rng() * 25)
    languageData = [
      {
        name: 'rust',
        label: 'Rust',
        completions: rustCompletion,
        selects: rustCompletion
      },
      {
        name: 'python',
        label: 'Python',
        completions: pythonCompletion,
        selects: pythonCompletion
      }
    ]
  } else {
    languageData = Object.entries(languageStats)
      .map(([key, stats]) => {
        return {
          name: key,
          selects: stats.selects,
          completions: stats.completions,
          label: stats.name
        }
      })
      .filter(item => item.completions)
      .slice(0, 5)
  }
  languageData = languageData.sort((a, b) => b.completions - a.completions)
  if (languageData.length === 0) {
    // Placeholder when there is no completions
    languageData = [
      {
        name: 'none',
        selects: 0,
        completions: 0.01,
        label: 'None'
      }
    ]
  }

  if (!data?.me?.id) return <></>

  return (
    <div className="flex w-full flex-col gap-y-8">
      <LoadingWrapper
        loading={fetchingYearlyStats}
        fallback={<Skeleton className="h-48" />}
      >
        <div>
          <h3 className="mb-2 text-sm font-medium tracking-tight">
            <b>{lastYearCompletions}</b> contributions in the last year
          </h3>
          <div className="flex items-end justify-center rounded-xl border p-5">
            <ActivityCalendar data={activities} />
          </div>
        </div>
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={
          <Skeleton className="h-48" />
        }
      >
        <CompletionCharts
          dailyStats={dailyStats}
          from={moment(startDate).toDate()}
          to={moment(endDate).toDate()}
          dateRange={DATE_RANGE}
        />
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={
          <Skeleton className="h-48" />
        }
      >
        <div>
          <h3 className="mb-2 text-sm font-medium tracking-tight">
            Top programming languages
          </h3>
          <div className="flex items-end justify-center rounded-xl border p-5">
            <ResponsiveContainer
              width="100%"
              height={(languageData.length + 1) * 50}
            >
              <BarChart
                layout="vertical"
                data={languageData}
                barCategoryGap={12}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
              >
                <Bar dataKey="completions" radius={3}>
                  <LabelList
                    dataKey="label"
                    content={
                      <LanguageLabel
                        languageData={languageData}
                        theme={theme}
                      />
                    }
                  />
                  {languageData.map((entry, index) => {
                    const lanColor = colorMap[entry.label.toLocaleLowerCase()]
                    const color = lanColor
                      ? lanColor
                      : theme === 'dark'
                      ? '#e8e1d3'
                      : '#54452c'
                    return <Cell key={`cell-${index}`} fill={color} />
                  })}
                </Bar>
                <XAxis type="number" fontSize={12} allowDecimals={false} />
                <YAxis
                  type="category"
                  dataKey="name"
                  hide
                  padding={{ bottom: 10 }}
                />
                <Tooltip
                  cursor={{ fill: 'transparent' }}
                  content={<LanguageTooltip />}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </LoadingWrapper>
    </div>
  )
}
