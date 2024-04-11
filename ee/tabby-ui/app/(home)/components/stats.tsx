'use client'

import { useWindowSize } from '@uidotdev/usehooks'
import { useTheme } from 'next-themes'
import { useQuery } from 'urql'
import moment from 'moment'
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
import GithubColors from 'github-colors'

import { useMe } from '@/lib/hooks/use-me'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
import { Card, CardContent } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'

import { useLanguageStats } from '../use-language-stats'
import { CompletionCharts } from './completion-charts'

const DATE_RANGE = 6

type LanguageData = {
  name: string
  selects: number
  completions: number
  label: string
}[]

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
  const { x, y, width, height, value, languageData, theme } = props
  const myLanguageData = languageData.find(data => data.label === value)

  if (!myLanguageData || myLanguageData.completions === 0) {
    return null
  }

  const padding = 5
  return (
    <text
      x={+x! + +width! + padding}
      y={+y! + +height! / 2}
      fill={theme === 'dark' ? '#fff' : '#000'}
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
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { completions, label } = payload[0].payload
    if (!completions) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Comletions:</span>
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

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: data?.me?.id
    }
  })
  let lastYearCompletions = 0
  const dailyCompletionMap: Record<string, number> =
    yearlyStatsData?.dailyStatsInPastYear?.reduce((acc, cur) => {
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
  let languageData: LanguageData = Object.entries(languageStats)
    .map(
      ([key, stats]) => {
        return {
          name: key,
          selects: stats.selects,
          completions: stats.completions,
          label: stats.name
        }
      }
    )
    .filter(item => item.completions)
    .slice(0, 5)
  languageData = languageData.sort((a, b) => b.completions - a.completions)

  if (!data?.me?.id) return <></>

  return (
    <div className="flex flex-col gap-y-8">
      <LoadingWrapper
        loading={fetchingYearlyStats}
        fallback={<Skeleton className="mb-8 h-48 md:w-[32rem] xl:w-[61rem]" />}
      >
        <div>
          <h3 className="mb-2 text-sm font-medium tracking-tight">
            <b>{lastYearCompletions}</b> activities in the last year
          </h3>
          <div className="flex items-end justify-center rounded-xl border p-5">
            <ActivityCalendar data={activities} />
          </div>
        </div>
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={
          <Skeleton className="h-96 w-full md:w-[32rem] xl:w-[61rem]" />
        }
      >
        <CompletionCharts
          dailyStats={dailyStatsData?.dailyStats}
          from={moment(startDate).toDate()}
          to={moment(endDate).toDate()}
          dateRange={DATE_RANGE}
        />
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={
          <Skeleton className="h-48 w-full md:w-[32rem] xl:w-[61rem]" />
        }
      >
        <div>
          <h3 className="mb-2 text-sm font-medium tracking-tight">
            Language completion stats
          </h3>
          <div className="flex items-end justify-center rounded-xl border p-5">
            <ResponsiveContainer width="100%" height={(languageData.length + 1 ) * 45}>
              <BarChart
                layout="vertical"
                data={languageData}
                barCategoryGap={5}
                margin={{ top: 5, right: 70, left: 10, bottom: 5 }}
              >
                <Bar dataKey="completions" radius={3}>
                  <LabelList
                    dataKey="label"
                    content={
                      <LanguageLabel languageData={languageData} theme={theme} />
                    }
                  />
                  {languageData.map((entry, index) => {
                    const lan = entry.label
                    const lanColor = GithubColors.get(lan)
                    const color = lanColor
                      ? lanColor.color
                      : theme === 'dark'
                      ? '#e8e1d3'
                      : '#54452c'
                    return <Cell key={`cell-${index}`} fill={color} />
                  })}
                </Bar>
                <XAxis type="number" fontSize={12} allowDecimals={false}  />
                <YAxis type="category" dataKey="name" hide padding={{ bottom: 20 }} />
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
