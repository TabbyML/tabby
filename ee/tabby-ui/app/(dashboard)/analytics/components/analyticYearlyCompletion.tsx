import moment from 'moment'

import ActivityCalendar from '@/components/activity-calendar'

import type { DailyStats } from '../types/stats'

export function AnalyticYearlyCompletion({
  yearlyStats
}: {
  yearlyStats: DailyStats[] | undefined
}) {
  let lastYearCompletions = 0
  const dailyCompletionMap: Record<string, number> =
    yearlyStats?.reduce((acc, cur) => {
      const date = moment(cur.start).format('YYYY-MM-DD')
      lastYearCompletions += cur.completions
      return { ...acc, [date]: cur.completions }
    }, {}) || {}

  const data = new Array(365)
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
    <div className="flex h-full flex-col rounded-lg border bg-primary-foreground/30 p-4">
      <h1 className="text-xl font-bold">Activity</h1>
      <p className="mt-0.5 text-xs text-muted-foreground">
        {lastYearCompletions} completions in the last year
      </p>
      <div className="mt-5 flex flex-1 items-center justify-center xl:mt-0">
        <ActivityCalendar data={data} />
      </div>
    </div>
  )
}
