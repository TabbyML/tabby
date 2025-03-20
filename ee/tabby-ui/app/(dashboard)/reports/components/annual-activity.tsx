'use client'

import { useWindowSize } from '@uidotdev/usehooks'
import ReactActivityCalendar from 'react-activity-calendar'

import { useCurrentTheme } from '@/lib/hooks/use-current-theme'

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
  const blockSize =
    width >= 1300 ? 13 : width >= 1100 ? 9 : width >= 900 ? 6 : 5

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
    />
  )
}

export function AnnualActivity({
  totalCount,
  dailyData
}: {
  totalCount: number
  dailyData: Array<{
    date: string
    count: number
    level: number
  }>
}) {
  return (
    <div className="flex h-full flex-col rounded-lg border bg-primary-foreground/30 px-6 py-4">
      <h3 className="mb-5 text-sm font-medium tracking-tight">
        <b>{totalCount}</b> activities in the last year
      </h3>
      <div className="flex flex-1 items-center justify-center">
        <ActivityCalendar data={dailyData} />
      </div>
    </div>
  )
}
