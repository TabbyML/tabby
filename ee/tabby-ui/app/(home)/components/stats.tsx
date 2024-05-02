'use client'

import { useSearchParams } from 'next/navigation'
import { useWindowSize } from '@uidotdev/usehooks'
import { eachDayOfInterval } from 'date-fns'
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
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { getLanguageColor, getLanguageDisplayName } from '@/lib/language-utils'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import { CompletionCharts } from './completion-charts'

const DATE_RANGE = 6

type LanguageData = {
  name: Language | 'NONE'
  views: number
}[]

type LanguageStats = Record<
  Language,
  {
    views: number
    name: Language
  }
>

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
    width >= 1300 ? 13 : width >= 1000 ? 8 : width >= 800 ? 10 : 9

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
  const myLanguageData = languageData.find(data => data.name === value)

  if (!myLanguageData || myLanguageData.views === 0) {
    return null
  }

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
      {getLanguageDisplayName(value as Language)}
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
      name: Language | 'NONE'
      views: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { views, name } = payload[0].payload
    if (!views || name === 'NONE') return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completions:</span>
            <b>{views}</b>
          </p>
          <p className="text-muted-foreground">
            {getLanguageDisplayName(name)}
          </p>
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
  const isDemoMode = useIsDemoMode()

  const sample = isDemoMode || searchParams.get('sample') === 'true'
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
      start: moment().subtract(DATE_RANGE, 'day').toDate(),
      end: moment().toDate()
    })
    dailyStats = daysBetweenRange.map(date => {
      const languages = [Language.Typescript, Language.Python, Language.Rust]
      const rng = seedrandom(moment(date).format('YYYY-MM-DD') + data?.me.id)
      const selects = Math.ceil(rng() * 20)
      const completions = Math.ceil(selects / 0.35)
      return {
        start: moment(date).utc().format(),
        end: moment(date).add(1, 'day').utc().format(),
        completions,
        selects,
        views: completions,
        language: languages[selects % languages.length]
      }
    })
  } else {
    dailyStats = dailyStatsData?.dailyStats.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects,
      views: item.views,
      language: item.language
    }))
  }

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: data?.me?.id
    }
  })
  let lastYearActivities = 0
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
        start: moment(date).format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        end: moment(date).add(1, 'day').format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        completions,
        selects,
        views: completions
      }
    })
  } else {
    yearlyStats = yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects,
      views: item.views
    }))
  }
  const dailyCompletionMap: Record<string, number> =
    yearlyStats?.reduce((acc, cur) => {
      const date = moment.utc(cur.start).format('YYYY-MM-DD')
      lastYearActivities += cur.views
      lastYearActivities += cur.selects
      return { ...acc, [date]: cur.views }
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

  // Prepare and structure data for populating the language usage charts
  const languageStats = {} as LanguageStats
  dailyStats?.forEach(stats => {
    languageStats[stats.language] = languageStats[stats.language] || {
      views: 0,
      name: stats.language
    }
    languageStats[stats.language].views += stats.views
  })
  let languageData: LanguageData = Object.entries(languageStats)
    .map(([_, stats]) => {
      return {
        name: stats.name,
        views: stats.views
      }
    })
    .filter(item => item.views)
    .sort((a, b) => b.views - a.views)
    .slice(0, 5)
  if (languageData.length === 0) {
    // Placeholder when there is no views
    languageData = [
      {
        name: 'NONE',
        views: 0.01
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
            <b>{lastYearActivities}</b> activities in the last year
          </h3>
          <div className="flex items-end justify-center rounded-xl border p-5">
            <ActivityCalendar data={activities} />
          </div>
        </div>
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-48" />}
      >
        <CompletionCharts
          dailyStats={dailyStats}
          from={moment().subtract(DATE_RANGE, 'day').toDate()}
          to={moment().toDate()}
        />
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-48" />}
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
                <Bar dataKey="views" radius={3}>
                  <LabelList
                    dataKey="name"
                    content={
                      <LanguageLabel
                        languageData={languageData}
                        theme={theme}
                      />
                    }
                  />
                  {languageData.map((entry, index) => {
                    const lanColor = getLanguageColor(entry.name)
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
