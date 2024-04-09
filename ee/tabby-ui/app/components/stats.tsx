'use client'

import moment from 'moment'
import { useQuery } from 'urql'
import { useTheme } from 'next-themes'
import { useWindowSize } from '@uidotdev/usehooks'
import { sum } from 'lodash-es'

import { useMe } from '@/lib/hooks/use-me'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
import { Language } from '@/lib/gql/generates/graphql'

import { Skeleton } from '@/components/ui/skeleton'

import ReactActivityCalendar from 'react-activity-calendar'
import LoadingWrapper from '@/components/loading-wrapper'
import { Summary, type LanguageStats } from './summary';

import type { DailyStats } from '@/lib/types/stats'

const DATE_RANGE = 7

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
  const blockSize = width >= 1300 ? 13 : width >= 1000 ? 11 : 8

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

  // Query language stats
  const languageStats: LanguageStats = {} as LanguageStats
  for (const [languageName, language] of Object.entries(Language)) {
    const [{ data: dailyStatsData }] = useQuery({
      query: queryDailyStats,
      variables: {
        start: startDate,
        end: endDate,
        users: data.me.id,
        languages: [language]
      }
    })
    if (dailyStatsData?.dailyStats) {
      languageStats[language] = languageStats[language] || {
        selects: 0,
        completions: 0,
        name: languageName
      }
      languageStats[language].selects += sum(dailyStatsData.dailyStats.map(stats => stats.selects))
      languageStats[language].completions += sum(dailyStatsData.dailyStats.map(stats => stats.completions))
    }
  }
  return (
    <div className="flex flex-col gap-y-8">
      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-24" />}>
        <Summary
          dailyStats={dailyStats}
          from={moment(startDate).toDate()}
          to={moment(endDate).toDate()}
          dateRange={DATE_RANGE}
          languageStats={languageStats} />
      </LoadingWrapper>

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
    </div>
  )
}
