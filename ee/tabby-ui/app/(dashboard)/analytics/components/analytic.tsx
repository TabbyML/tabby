'use client'

import { useState } from 'react'
import moment from 'moment'
import { useQuery } from 'urql'
import { DateRange } from "react-day-picker"
import { sum } from "lodash-es"
import numeral from "numeral"

import { queryDailyStatsInPastYear, queryDailyStats } from '@/lib/tabby/query'
import { Language } from '@/lib/gql/generates/graphql'
import { useAllMembers } from '@/lib/hooks/use-all-members'

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from '@/components/ui/skeleton'
import DatePickerWithRange from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'
import { AnalyticDailyCompletion } from './analyticDailyCompletion'
import { AnlyticAcceptance } from './analyticAccptance'
import { AnalyticYearlyCompletion } from './analyticYearlyCompletion'

import type { DailyStats } from '../types/stats'

const INITIAL_DATE_RANGE = 14
const KEY_SELECT_ALL = 'all'

function AnalyticSummary({
  dailyStats,
}: {
  dailyStats: DailyStats[] | undefined,
}) {
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  return (
    <div className="flex items-center justify-center space-x-4 xl:justify-start">
      <div className="space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4 lg:min-w-[250px]">
        <p className="text-xs text-muted-foreground lg:text-sm">Total completions</p>
        <p className="font-bold lg:text-3xl">
          {numeral(totalCompletions).format('0,0')}
        </p>
      </div>

      <div className="space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4 lg:min-w-[250px]">
        <p className="text-xs text-muted-foreground lg:text-sm">Minutes saved / completion</p>
        <p className="font-bold lg:text-3xl">2</p>
      </div>

      <div className="space-y-0.5 rounded-lg border bg-primary-foreground/30 p-4 lg:min-w-[250px]">
        <p className="text-xs text-muted-foreground lg:text-sm">Hours saved in total</p>
        <p className="font-bold lg:text-3xl">100</p>
      </div>
    </div>
  )
}

export function Analytic() {
  const [members] = useAllMembers()
  const [dateRange, setDateRange] = useState<DateRange>({
    from: moment().subtract(INITIAL_DATE_RANGE, 'day').toDate(),
    to: moment().toDate()
  })
  const [selectedMember, setSelectedMember] = useState(KEY_SELECT_ALL)
  const [selectedLanguage, setSelectedLanguage] = useState<'all' | Language>(KEY_SELECT_ALL)

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: moment(dateRange.from).startOf('day').utc().format(),
      end: moment(dateRange.to).endOf('day').utc().format(),
      users: selectedMember === KEY_SELECT_ALL ? undefined : [selectedMember],
      languages: selectedLanguage === KEY_SELECT_ALL ? undefined : [selectedLanguage],
    }
  })
  const dailyStats: DailyStats[] | undefined = dailyStatsData?.dailyStats.map(item => ({
    start: item.start,
    end: item.end,
    completions: item.completions,
    selects: item.selects,
  }))
  console.log(dailyStats)

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    }
  })
  const yearlyStats: DailyStats[] | undefined = yearlyStatsData?.dailyStatsInPastYear.map(item => ({
    start: item.start,
    end: item.end,
    completions: item.completions,
    selects: item.selects,
  }))

  const onDateOpenChange = (isOpen: boolean, dateRange: DateRange | undefined) => {
    if (!isOpen) {
      if (dateRange) {
        setDateRange(dateRange)
      }
    }
  }

  return (
    <div className="flex flex-col gap-y-6">
      <div className="flex flex-col items-center justify-between gap-y-3 xl:flex-row xl:gap-y-0">
        <div className="flex flex-col justify-center xl:justify-start">
          <h1 className="mb-1.5 scroll-m-20 text-center text-4xl font-extrabold tracking-tight lg:text-5xl xl:text-left">
            Analytics
          </h1>
          <p className="text-muted-foreground">Overview of code completion usage</p>
        </div>

        <div className="flex flex-col items-center gap-y-2 lg:flex-row lg:gap-y-0 lg:space-x-4">
          <Select defaultValue={KEY_SELECT_ALL} onValueChange={setSelectedMember}>
            <SelectTrigger className="w-[300px] lg:w-[180px]" >
              <div className="flex w-full items-center truncate ">
                <span className="mr-1.5 text-muted-foreground">
                  Member:
                </span>
                <div className="overflow-hidden text-ellipsis">
                  <SelectValue />
                </div>
              </div>
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value={KEY_SELECT_ALL}>All</SelectItem>
                {members.map(member => (
                  <SelectItem value={member.id} key={member.id}>{member.email}</SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>

          <Select defaultValue={KEY_SELECT_ALL} onValueChange={(value: 'all' | Language) => setSelectedLanguage(value)}>
            <SelectTrigger className="w-[300px] lg:w-[180px]" >
              <div className="flex w-full items-center truncate">
                <span className="mr-1.5 text-muted-foreground">
                  Language:
                </span>
                <div className="overflow-hidden text-ellipsis">
                  <SelectValue />
                </div>
              </div>
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                <SelectItem value={'all'}>All</SelectItem>
                {Object.entries(Language).map(([key, value]) => (
                  <SelectItem key={value} value={value}>{key}</SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>

          <DatePickerWithRange
            buttonClassName="h-full"
            contentAlign="end"
            dateRange={dateRange}
            onOpenChange={onDateOpenChange} />
        </div>
      </div>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-24 w-1/2" />}>
        <AnalyticSummary dailyStats={dailyStats} />
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
        fallback={<Skeleton className="h-64 w-full" />}>
        <AnalyticDailyCompletion
          dailyStats={dailyStats}
          dateRange={dateRange} />
      </LoadingWrapper>


      <div className="flex flex-col gap-y-6 xl:flex-row xl:gap-x-6 xl:gap-y-0">
        <div className="flex-1">
          <LoadingWrapper
            loading={fetchingDailyState}
            fallback={<Skeleton className="h-64 w-full" />}>
            <AnlyticAcceptance
              dailyStats={dailyStats}
              dateRange={dateRange} />
          </LoadingWrapper>
        </div>
        <div style={{ flex: 3 }}>
          <LoadingWrapper
            loading={fetchingDailyState || fetchingYearlyStats}
            fallback={<Skeleton className="h-64 w-full" />}>
            <AnalyticYearlyCompletion yearlyStats={yearlyStats} />
          </LoadingWrapper>
        </div>
      </div>
    </div>
  )
}