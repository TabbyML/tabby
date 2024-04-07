'use client'

import { useState } from 'react'
import { sum } from 'lodash-es'
import moment from 'moment'
import numeral from 'numeral'
import { DateRange } from 'react-day-picker'
import { useQuery } from 'urql'

import { Language } from '@/lib/gql/generates/graphql'
import { useAllMembers } from '../hooks/use-all-members'
import { queryDailyStats, queryDailyStatsInPastYear } from '../query'

import { IconActivity, IconCode, IconCheck } from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import DatePickerWithRange from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

import { AnalyticDailyCompletion } from './analyticDailyCompletion'
import { AnalyticYearlyCompletion } from './analyticYearlyCompletion'

import type { DailyStats } from '../types/stats'

const INITIAL_DATE_RANGE = 14
const KEY_SELECT_ALL = 'all'

function AnalyticSummary({
  dailyStats
}: {
  dailyStats: DailyStats[] | undefined
}) {
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  return (
    <div className="flex w-full items-center justify-center space-x-6 xl:justify-start">
      <Card className="flex-1 bg-primary-foreground/30">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Accept Rate
          </CardTitle>
          <IconActivity className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">TBD</div>
          <p className="text-xs text-muted-foreground">
            +TBD from last week
          </p>
        </CardContent>
      </Card>

      <Card className="flex-1 bg-primary-foreground/30">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total completions
          </CardTitle>
          <IconCode className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{numeral(totalCompletions).format('0,0')}</div>
          <p className="text-xs text-muted-foreground">
            +TBD from last week
          </p>
        </CardContent>
      </Card>

      <Card className="flex-1 bg-primary-foreground/30">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total acceptances
          </CardTitle>
          <IconCheck className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">TBD</div>
          <p className="text-xs text-muted-foreground">
            +TBD from last week
          </p>
        </CardContent>
      </Card>
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
  const [selectedLanguage, setSelectedLanguage] = useState<'all' | Language>(
    KEY_SELECT_ALL
  )

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: moment(dateRange.from).startOf('day').utc().format(),
      end: moment(dateRange.to).endOf('day').utc().format(),
      users: selectedMember === KEY_SELECT_ALL ? undefined : [selectedMember],
      languages:
        selectedLanguage === KEY_SELECT_ALL ? undefined : [selectedLanguage]
    }
  })
  const dailyStats: DailyStats[] | undefined = dailyStatsData?.dailyStats.map(
    item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    })
  )

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    }
  })
  const yearlyStats: DailyStats[] | undefined =
    yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))

  const onDateOpenChange = (
    isOpen: boolean,
    dateRange: DateRange | undefined
  ) => {
    if (!isOpen) {
      if (dateRange) {
        setDateRange(dateRange)
      }
    }
  }

  return (
    <div className="mx-auto max-w-5xl">
      <div className="mb-3 flex flex-col items-center justify-between gap-y-3 xl:flex-row xl:gap-y-0">
        <div className="flex flex-col justify-center xl:justify-start">
          <h1 className="mb-1.5 scroll-m-20 text-center text-4xl font-extrabold tracking-tight lg:text-5xl xl:text-left">
            Reports
          </h1>
          <p className="text-muted-foreground">
            Statistics around Tabby IDE / Extensions
          </p>
        </div>

        <Select
          defaultValue={KEY_SELECT_ALL}
          onValueChange={setSelectedMember}
        >
          <SelectTrigger className="w-[300px] lg:w-[150px]">
            <div className="flex w-full items-center truncate ">
              <span className="mr-1.5 text-muted-foreground">Member:</span>
              <div className="overflow-hidden text-ellipsis">
                <SelectValue />
              </div>
            </div>
          </SelectTrigger>
          <SelectContent align='end'>
            <SelectGroup>
              <SelectItem value={KEY_SELECT_ALL}>All</SelectItem>
              {members.map(member => (
                <SelectItem value={member.id} key={member.id}>
                  {member.email}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      <LoadingWrapper>
        <div className="mb-10 flex flex-col gap-y-5">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Usage</h1>
            
            <div className="flex items-center gap-x-3">
              <Select
                defaultValue={KEY_SELECT_ALL}
                onValueChange={(value: 'all' | Language) =>
                  setSelectedLanguage(value)
                }
              >
                <SelectTrigger className="w-[300px] lg:w-[180px]">
                  <div className="flex w-full items-center truncate">
                    <span className="mr-1.5 text-muted-foreground">Language:</span>
                    <div className="overflow-hidden text-ellipsis">
                      <SelectValue />
                    </div>
                  </div>
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value={'all'}>All</SelectItem>
                    {Object.entries(Language).map(([key, value]) => (
                      <SelectItem key={value} value={value}>
                        {key}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>

              <DatePickerWithRange
                buttonClassName="h-full"
                contentAlign="end"
                dateRange={dateRange}
                onOpenChange={onDateOpenChange}
              />
            </div>
          </div>

          <AnalyticSummary dailyStats={dailyStats} />

          <AnalyticDailyCompletion
            dailyStats={dailyStats}
            dateRange={dateRange}
          />
        </div>
      </LoadingWrapper>
      
      <LoadingWrapper>
        <div className="mb-10">
          <h1 className="mb-3 text-xl font-semibold">Activity</h1>
          <AnalyticYearlyCompletion yearlyStats={yearlyStats} />
        </div>
      </LoadingWrapper>

      {/* <div className="flex flex-col gap-y-6 xl:flex-row xl:gap-x-6 xl:gap-y-0">
        {false && (
          <div className="flex-1">
            <LoadingWrapper
              loading={fetchingDailyState}
              fallback={<Skeleton className="h-64 w-full" />}
            >
              <AnlyticAcceptance
                dailyStats={dailyStats}
                dateRange={dateRange}
              />
            </LoadingWrapper>
          </div>
        )}
        <div style={{ flex: 3 }}>
          <LoadingWrapper
            loading={fetchingDailyState || fetchingYearlyStats}
            fallback={<Skeleton className="h-64 w-full" />}
          >
            
          </LoadingWrapper>
        </div>
      </div> */}
    </div>
  )
}
