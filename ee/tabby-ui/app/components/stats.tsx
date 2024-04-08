import moment from 'moment'
import { sum } from 'lodash-es'
import numeral from 'numeral'
import { useQuery } from 'urql'
import { summary as summaryStreak } from 'date-streaks';
import { useTheme } from 'next-themes'
import { useWindowSize } from '@uidotdev/usehooks'

import { useMe } from '@/lib/hooks/use-me'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'

import {
  IconTrendingUp,
  IconCheck,
  IconCode,
} from '@/components/ui/icons'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import ReactActivityCalendar from 'react-activity-calendar'
import LoadingWrapper from '@/components/loading-wrapper'

import type { DailyStats } from '@/lib/types/stats'

const DATE_RANGE = 7

function Summary({
  dailyStats
}: {
  dailyStats: DailyStats[] | undefined
}) {
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  const totalAcceptances = sum(dailyStats?.map(stats => stats.selects))
  const { currentStreak } = summaryStreak({
    dates: dailyStats?.map(stats => new Date(stats.start)) || []
  })
  return (
    <div className="flex w-full items-center justify-center space-x-6 xl:justify-start">
      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Completions In Last {DATE_RANGE} Days</CardTitle>
          <IconCode className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{numeral(totalCompletions).format('0,0')}</div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Acceptances In Last {DATE_RANGE} Days
          </CardTitle>
          <IconCheck className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {numeral(totalAcceptances).format('0,0')}
          </div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Streak
          </CardTitle>
          <IconTrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{currentStreak}</div>
        </CardContent>
      </Card>
    </div>
  )
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
  const blockSize = width >= 1300 ? 13 : width >= 1000 ? 9 : 5

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

export default function Stats() {
  const [{ data }] = useMe()

  if (!data?.me?.email) return null

  const startDate = moment().subtract(DATE_RANGE, 'day').startOf('day').utc().format()
  const endDate = moment().endOf('day').utc().format()

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: startDate,
      end: endDate,
      users: data.me.id
    }
  })
  const dailyStats: DailyStats[] | undefined = dailyStatsData?.dailyStats.map(
    item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    })
  )

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: data.me.id
    }
  })
  const yearlyStats: DailyStats[] | undefined =
    yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))
  let lastYearCompletions = 0
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

  return (
    <div className="flex flex-col gap-y-8">
      <LoadingWrapper
        loading={fetchingYearlyStats}
        fallback={<Skeleton className="mb-8 h-48" />}
      >
        <div>
          <h3 className="mb-2 text-sm font-medium tracking-tight">
            <b>{lastYearCompletions}</b> activities in the last year
          </h3>
          <div className="flex items-end justify-center rounded-xl bg-primary-foreground/30 py-5">
            <ActivityCalendar data={activities} />
          </div>
        </div>
      </LoadingWrapper>

      {/* <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-24" />}>
        <Summary dailyStats={dailyStats} />
      </LoadingWrapper> */}

      {/* <div>
        <p className="mb-2 text-sm text-secondary-foreground">
          My top coding languages
        </p>
        <div className="flex flex-col gap-y-5 rounded-xl bg-primary-foreground/50 px-11 py-6">
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Rust</p>
            <div className="flex-1">
              <div className="h-2 w-[80%] rounded-full bg-yellow-600 dark:bg-yellow-600" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Python</p>
            <div className="flex-1">
              <div className="h-2 w-[30%] rounded-full bg-blue-500 dark:bg-blue-300" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">Javascript</p>
            <div className="flex-1">
              <div className="h-2 w-[2%] rounded-full bg-yellow-400 dark:bg-yellow-300" />
            </div>
          </div>
          <div className="flex items-center">
            <p className="text-mute w-20 pr-5 text-sm">CSS</p>
            <div className="flex-1">
              <div className="h-2 w-[2%] rounded-full bg-red-500 dark:bg-red-400" />
            </div>
          </div>
        </div>
      </div> */}
    </div>
  )
}
