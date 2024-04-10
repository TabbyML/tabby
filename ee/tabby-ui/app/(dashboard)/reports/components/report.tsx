'use client'

import { useState } from 'react'
import { sum } from 'lodash-es'
import moment from 'moment'
import numeral from 'numeral'
import { DateRange } from 'react-day-picker'
import { useQuery } from 'urql'
import seedrandom from 'seedrandom'
import { eachDayOfInterval } from 'date-fns'

import { Language } from '@/lib/gql/generates/graphql'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import DatePickerWithRange from '@/components/ui/date-range-picker'
import {
  IconActivity,
  IconCheck,
  IconCode,
  IconUsers
} from '@/components/ui/icons'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { queryDailyStats, queryDailyStatsInPastYear } from '../query'
import type { DailyStats } from '../types/stats'
import { useAllMembers } from '../use-all-members'
import { AnnualActivity } from './annual-activity'
import { DailyActivity } from './daily-activity'

const INITIAL_DATE_RANGE = 14
const KEY_SELECT_ALL = 'all'

function StatsSummary({
  dailyStats
}: {
  dailyStats: DailyStats[] | undefined
}) {
  const totalCompletions = sum(dailyStats?.map(stats => stats.completions))
  const totalAcceptances = sum(dailyStats?.map(stats => stats.selects))
  const acceptRate =
    totalAcceptances === 0
      ? 0
      : ((totalAcceptances / totalCompletions) * 100).toFixed(2)
  return (
    <div className="flex w-full items-center justify-center space-x-6 xl:justify-start">
      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Accept Rate</CardTitle>
          <IconActivity className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{acceptRate}%</div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total completions
          </CardTitle>
          <IconCode className="text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {numeral(totalCompletions).format('0,0')}
          </div>
        </CardContent>
      </Card>

      <Card className="flex flex-1 flex-col justify-between self-stretch bg-primary-foreground/30 md:block">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total acceptances
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

export function Report({
  sample
}: {
  sample: boolean
}) {
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
  let dailyStats: DailyStats[] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: dateRange.from!,
      end: dateRange.to || dateRange.from!
    })
    dailyStats = daysBetweenRange.map(
      date => {
        const rng = seedrandom(moment(date).format('YYYY-MM-DD') + selectedMember + selectedLanguage)
        const selects = Math.ceil(rng() * 20)
        const completions = selects + Math.floor(rng() * 10)
        return {
          start: moment(date).startOf('day').toDate(),
          end:  moment(date).endOf('day').toDate(),
          completions,
          selects
        }
      }
    )
  } else {
    dailyStats = dailyStatsData?.dailyStats.map(
      item => ({
        start: item.start,
        end: item.end,
        completions: item.completions ,
        selects: item.selects
      })
    )
  } 

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    }
  })
  let yearlyStats: DailyStats[] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: moment().toDate(),
      end: moment().subtract(365, 'days').toDate()
    })
    yearlyStats = daysBetweenRange.map(
      date => {
        const rng = seedrandom(moment(date).format('YYYY-MM-DD') + selectedMember + selectedLanguage)
        const selects = Math.ceil(rng() * 20)
        const completions = selects + Math.floor(rng() * 10)
        return {
          start: moment(date).startOf('day').toDate(),
          end:  moment(date).endOf('day').toDate(),
          completions,
          selects
        }
      }
    )
  } else {
    yearlyStats = yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))
  }

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
      <div className="mb-4 flex flex-col items-center justify-between gap-y-2 md:flex-row md:items-end md:gap-y-0">
        <SubHeader className="mb-0">
          Statistics around Tabby IDE / Extensions
        </SubHeader>

        <LoadingWrapper
          loading={fetchingDailyState}
          fallback={<Skeleton className="h-8 w-32" />}
        >
          <Select
            defaultValue={KEY_SELECT_ALL}
            onValueChange={setSelectedMember}
          >
            <SelectTrigger className="h-auto w-auto border-none py-0 shadow-none">
              <div className="flex h-6 items-center">
                <IconUsers className="mr-1" />
                <p className="mr-1.5">Member:</p>
                <div className="w-[80px] overflow-hidden text-ellipsis text-left">
                  <SelectValue />
                </div>
              </div>
            </SelectTrigger>
            <SelectContent align="end">
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
        </LoadingWrapper>
      </div>

      <LoadingWrapper
        loading={fetchingYearlyStats}
        fallback={<Skeleton className="mb-8 h-48" />}
      >
        <div className="mb-8">
          <h1 className="mb-2 text-xl font-semibold">Activity</h1>
          <AnnualActivity yearlyStats={yearlyStats} />
        </div>
      </LoadingWrapper>

      <LoadingWrapper
        loading={fetchingDailyState}
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
          <div className="-mb-2 flex flex-col justify-between gap-y-1 md:flex-row md:items-end md:gap-y-0">
            <h1 className="text-xl font-semibold">Usage</h1>

            <div className="flex items-center gap-x-3">
              <Select
                defaultValue={KEY_SELECT_ALL}
                onValueChange={(value: 'all' | Language) =>
                  setSelectedLanguage(value)
                }
              >
                <SelectTrigger className="w-[180px]">
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

          <StatsSummary dailyStats={dailyStats} />

          <DailyActivity dailyStats={dailyStats} dateRange={dateRange} />
        </div>
      </LoadingWrapper>
    </div>
  )
}
