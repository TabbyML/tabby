'use client'

import { useSearchParams } from 'next/navigation'
import { useWindowSize } from '@uidotdev/usehooks'
import { eachDayOfInterval } from 'date-fns'
import moment from 'moment'
import ReactActivityCalendar from 'react-activity-calendar'
import seedrandom from 'seedrandom'
import { useQuery } from 'urql'

import {
  DailyStatsInPastYearQuery,
  DailyStatsQuery,
  Language
} from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useMe } from '@/lib/hooks/use-me'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'

import { AnimationWrapper } from './animation-wrapper'
import { CompletionCharts } from './completion-charts'

const DATE_RANGE = 6

function ActivityCalendar({
  data
}: {
  data: {
    date: string
    count: number
    level: number
  }[]
}) {
  const { theme } = useCurrentTheme()
  const size = useWindowSize()
  const width = size.width || 0
  const blockSize = width >= 968 ? 12 : 11

  return (
    <div className="h-[152px]">
      <ReactActivityCalendar
        data={data}
        colorScheme={theme === 'dark' ? 'dark' : 'light'}
        theme={{
          light: ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39'],
          dark: ['rgb(45, 51, 59)', '#0e4429', '#006d32', '#26a641', '#39d353']
        }}
        blockSize={blockSize}
        hideTotalCount
        fontSize={12}
      />
    </div>
  )
}

export default function Stats() {
  const [{ data }] = useMe()
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

  if (!data?.me?.id) return <></>

  return (
    <>
      <AnimationWrapper
        viewport={{
          amount: 0.1
        }}
        style={{ width: '100%' }}
        delay={0.1}
      >
        <div className="rounded-2xl border px-[1.125rem] py-4">
          <div className="mb-3 text-base">
            <span className="font-semibold">{lastYearActivities}</span>{' '}
            activities in the past year
          </div>
          <ActivityCalendar data={activities} />
        </div>
      </AnimationWrapper>
      <CompletionCharts
        dailyStats={dailyStats}
        from={moment().subtract(DATE_RANGE, 'day').toDate()}
        to={moment().toDate()}
      />
    </>
  )
}
