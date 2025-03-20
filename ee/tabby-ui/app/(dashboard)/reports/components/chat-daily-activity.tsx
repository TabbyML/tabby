'use client'

import { useState } from 'react'
import moment from 'moment'
import numeral from 'numeral'
import { DateRange } from 'react-day-picker'
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'

import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useChatDailyStats } from '@/lib/hooks/use-statistics'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IconMessageSquare } from '@/components/ui/icons'
import { Skeleton } from '@/components/ui/skeleton'
import DateRangePicker from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'

import { DEFAULT_DATE_RANGE } from './constants'

export function ChatDailyActivity({
  selectedMember,
  sample
}: {
  selectedMember: string
  sample?: boolean
}) {
  const { theme } = useCurrentTheme()
  const [dateRange, setDateRange] = useState<DateRange>({
    from: moment().add(parseInt(DEFAULT_DATE_RANGE, 10), 'day').toDate(),
    to: moment().toDate()
  })

  const from = dateRange.from || new Date()
  const to = dateRange.to || from

  const { fetchingChatDailyStats, chatChartData, chatDailyStats, totalCount } =
    useChatDailyStats({
      sample,
      selectedMember,
      dateRange: {
        from,
        to
      }
    })
  return (
    <LoadingWrapper
      loading={fetchingChatDailyStats}
      fallback={
        <div className="flex flex-col gap-5">
          <div className="flex justify-between gap-5">
            <Skeleton className="h-32 flex-1" />
            <Skeleton className="h-32 flex-1" />
            <Skeleton className="h-32 flex-1" />
          </div>
          <Skeleton className="h-56" />
        </div>
      }
    >
      <div className="mb-10 flex flex-col gap-y-5">
        <div className="flex flex-col items-center justify-between gap-y-1 md:flex-row md:gap-y-0">
          <h2 className="font-semibold">Chats</h2>
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-y-0">
            <DateRangePicker
              options={[
                { label: 'Last 7 days', value: '-7d' },
                { label: 'Last 14 days', value: '-14d' },
                { label: 'Last 30 days', value: '-30d' }
              ]}
              defaultValue={DEFAULT_DATE_RANGE}
              onSelect={setDateRange}
              hasToday
              hasYesterday
            />
          </div>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 lg:block">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Chats</CardTitle>
              <IconMessageSquare className="text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {numeral(totalCount).format('0,0')}
              </div>
            </CardContent>
          </Card>
        </div>
        <div className="rounded-lg border bg-primary-foreground/30 px-6 py-4">
          <h3 className="mb-5 text-sm font-medium tracking-tight">
            Daily Statistics
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={chatChartData}
              margin={{
                top: 5,
                right: 20,
                left: 20,
                bottom: 5
              }}
            >
              <Bar
                dataKey="chats"
                stackId="stats"
                fill={theme === 'dark' ? '#423929' : '#e8e1d3'}
                radius={3}
              />
              <XAxis dataKey="name" fontSize={12} />
              <YAxis fontSize={12} width={20} allowDecimals={false} />
              <Tooltip
                cursor={{ fill: 'transparent' }}
                content={<BarTooltip />}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </LoadingWrapper>
  )
}

function BarTooltip({
  active,
  payload,
  label
}: {
  active?: boolean
  label?: string
  payload?: {
    name: string
    payload: {
      chats: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { chats } = payload[0].payload
    if (!chats) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Chat:</span>
            <b>{chats}</b>
          </p>
          <p className="text-muted-foreground">{label}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}
