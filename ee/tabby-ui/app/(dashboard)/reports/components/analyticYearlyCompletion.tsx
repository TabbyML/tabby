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
    <div className="flex h-full flex-col rounded-lg border bg-primary-foreground/30 px-6 py-4">
      <h3 className="mb-5 text-sm font-medium tracking-tight">
        <b>{lastYearCompletions}</b> activities in the last year
      </h3>
      <div className="flex flex-1 items-center justify-center">
        <ActivityCalendar data={data} />
      </div>
    </div>
  )
}
