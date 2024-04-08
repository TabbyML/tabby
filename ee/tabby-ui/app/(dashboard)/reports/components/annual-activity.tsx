'use client'

import { useWindowSize } from '@uidotdev/usehooks'
import moment from 'moment'
import { useTheme } from 'next-themes'
import ReactActivityCalendar from 'react-activity-calendar'

import type { DailyStats } from '../types/stats'

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

export function AnnualActivity({
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
