'use client'

import { useSearchParams } from 'next/navigation'
import { useWindowSize } from '@uidotdev/usehooks'
import moment from 'moment'
import ReactActivityCalendar from 'react-activity-calendar'

import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useMe } from '@/lib/hooks/use-me'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { useYearlyStats } from '@/lib/hooks/use-statistics'

import { AnimationWrapper } from './animation-wrapper'
import { DailyCharts } from './daily-charts'

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

  const { dailyData, totalCount } = useYearlyStats({
    selectedMember: data?.me?.id,
    sample
  })

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
            <span className="mr-1 font-semibold">{totalCount}</span>
            activities in the past year
          </div>
          <ActivityCalendar data={dailyData} />
        </div>
      </AnimationWrapper>
      <DailyCharts
        sample={sample}
        from={moment().subtract(DATE_RANGE, 'day').toDate()}
        to={moment().toDate()}
        userId={data?.me.id}
      />
    </>
  )
}
