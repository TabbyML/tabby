'use client'

import { useState } from 'react'
import { sum } from 'lodash-es'
import moment from 'moment'
import numeral from 'numeral'
import type { DateRange } from 'react-day-picker'
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts'

import { DailyStatsQuery, Language } from '@/lib/gql/generates/graphql'
import { useCurrentTheme } from '@/lib/hooks/use-current-theme'
import { useCompletionDailyStats } from '@/lib/hooks/use-statistics'
import { getLanguageDisplayName } from '@/lib/language-utils'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator
} from '@/components/ui/command'
import {
  IconActivity,
  IconCheck,
  IconChevronUpDown,
  IconCode
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import DateRangePicker from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'

import { DEFAULT_DATE_RANGE } from './constants'

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
      views: number
      selects: number
      pendings: number
    }
  }[]
}) {
  if (active && payload && payload.length) {
    const { views, selects } = payload[0].payload
    if (!views) return null
    return (
      <Card>
        <CardContent className="flex flex-col gap-y-0.5 px-4 py-2 text-sm">
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Completion:</span>
            <b>{views}</b>
          </p>
          <p className="flex items-center">
            <span className="mr-3 inline-block w-20">Acceptance:</span>
            <b>{selects}</b>
          </p>
          <p className="text-muted-foreground">{label}</p>
        </CardContent>
      </Card>
    )
  }

  return null
}

export function CompletionDailyActivity({
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
  const [selectedLanguage, setSelectedLanguage] = useState<Language[]>([])
  // Query stats of selected date range
  const {
    completionChartData,
    completionDailyStats,
    fetchingCompletionDailyStats
  } = useCompletionDailyStats({
    selectedMember,
    dateRange,
    sample,
    languages: selectedLanguage
  })

  return (
    <LoadingWrapper
      loading={fetchingCompletionDailyStats}
      fallback={
        <div className="mb-10 flex flex-col gap-5">
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
        <h1 className="text-xl font-semibold">Usage</h1>
        <div className="flex flex-col gap-y-5">
          <div className="-mb-2 flex flex-col items-center justify-between gap-y-1 md:flex-row md:gap-y-0">
            <h2 className="font-semibold">Completions</h2>
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-y-0">
              <Popover>
                <PopoverTrigger asChild>
                  <div className="flex h-9 w-[240px] items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed md:w-[150px]">
                    <div className="w-full">
                      {selectedLanguage.length === 0 && (
                        <p className="w-full overflow-hidden text-ellipsis">
                          All languages
                        </p>
                      )}
                      {selectedLanguage.length === 1 && (
                        <p className="w-full overflow-hidden text-ellipsis">
                          {getLanguageDisplayName(selectedLanguage[0])}
                        </p>
                      )}
                      {selectedLanguage.length > 1 && (
                        <span className="px-1">
                          {selectedLanguage.length} selected
                        </span>
                      )}
                    </div>
                    <IconChevronUpDown className="h-3 w-3" />
                  </div>
                </PopoverTrigger>
                <PopoverContent
                  className="w-[240px] p-0 md:w-[180px]"
                  align="end"
                >
                  <Command>
                    <CommandInput placeholder="Language" />
                    <CommandList>
                      <CommandEmpty>No results found.</CommandEmpty>

                      <CommandGroup>
                        {Object.entries(Language)
                          .sort((_, b) => (b[1] === Language.Other ? -1 : 0))
                          .map(([_, value]) => {
                            const isSelected = selectedLanguage.includes(value)
                            return (
                              <CommandItem
                                key={value}
                                onSelect={() => {
                                  const newSelect = [...selectedLanguage]
                                  if (isSelected) {
                                    const idx = newSelect.findIndex(
                                      item => item === value
                                    )
                                    if (idx !== -1) newSelect.splice(idx, 1)
                                  } else {
                                    newSelect.push(value)
                                  }
                                  setSelectedLanguage(newSelect)
                                }}
                                className="!pointer-events-auto cursor-pointer !opacity-100"
                              >
                                <div
                                  className={cn(
                                    'mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary',
                                    isSelected
                                      ? 'bg-primary text-primary-foreground'
                                      : 'opacity-50 [&_svg]:invisible'
                                  )}
                                >
                                  <IconCheck className={cn('h-4 w-4')} />
                                </div>
                                <span>{getLanguageDisplayName(value)}</span>
                              </CommandItem>
                            )
                          })}
                      </CommandGroup>
                      {selectedLanguage.length > 0 && (
                        <>
                          <CommandSeparator />
                          <CommandGroup>
                            <CommandItem
                              onSelect={() => setSelectedLanguage([])}
                              className="!pointer-events-auto cursor-pointer justify-center text-center !opacity-100"
                            >
                              Clear filters
                            </CommandItem>
                          </CommandGroup>
                        </>
                      )}
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>

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
          <StatsSummary dailyStats={completionDailyStats} />
          <div className="rounded-lg border bg-primary-foreground/30 px-6 py-4">
            <h3 className="mb-5 text-sm font-medium tracking-tight">
              Daily Statistics
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={completionChartData}
                margin={{
                  top: 5,
                  right: 20,
                  left: 20,
                  bottom: 5
                }}
              >
                <Bar
                  dataKey="selects"
                  stackId="stats"
                  fill={theme === 'dark' ? '#e8e1d3' : '#54452c'}
                  radius={3}
                />
                <Bar
                  dataKey="pendings"
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
      </div>
    </LoadingWrapper>
  )
}

function StatsSummary({
  dailyStats
}: {
  dailyStats?: DailyStatsQuery['dailyStats']
}) {
  const totalViews = sum(dailyStats?.map(stats => stats.views))
  const totalAcceptances = sum(dailyStats?.map(stats => stats.selects))
  const acceptRate =
    totalAcceptances === 0
      ? 0
      : ((totalAcceptances / totalViews) * 100).toFixed(2)
  return (
    <div className="flex w-full flex-col items-start justify-center space-y-3 md:flex-row md:items-center md:space-x-6 md:space-y-0 xl:justify-start">
      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 lg:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Acceptance Rate</CardTitle>
          <IconActivity className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{acceptRate}%</div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 lg:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total Completions
          </CardTitle>
          <IconCode className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {numeral(totalViews).format('0,0')}
          </div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 lg:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total Acceptances
          </CardTitle>
          <IconCheck className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalAcceptances}</div>
        </CardContent>
      </Card>
    </div>
  )
}
