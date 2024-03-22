'use client'

import dynamic from 'next/dynamic'
import { useTheme } from 'next-themes'

// withou using dynamic, we got error "Error: calcTextDimensions() requires browser APIs"
const ReactActivityCalendar = dynamic(() => import('react-activity-calendar'), {
  ssr: false
})

export default function ActivityCalendar({
  data
}: {
  data: {
    date: string
    count: number
    level: number
  }[]
}) {
  const { theme } = useTheme()
  return (
    <ReactActivityCalendar
      data={data}
      colorScheme={theme === 'dark' ? 'dark' : 'light'}
      theme={{
        light: ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39'],
        dark: ['rgb(45, 51, 59)', '#0e4429', '#006d32', '#26a641', '#39d353']
      }}
      blockSize={13}
      hideTotalCount
      showWeekdayLabels
    />
  )
}
