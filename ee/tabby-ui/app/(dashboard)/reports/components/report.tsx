'use client'

import { useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { eachDayOfInterval } from 'date-fns'
import { sum } from 'lodash-es'
import moment from 'moment'
import numeral from 'numeral'
import { DateRange } from 'react-day-picker'
import seedrandom from 'seedrandom'
import { useQuery } from 'urql'

import {
  DailyStatsInPastYearQuery,
  DailyStatsQuery,
  Language
} from '@/lib/gql/generates/graphql'
import { useAllMembers } from '@/lib/hooks/use-all-members'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import { getLanguageDisplayName } from '@/lib/language-utils'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
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
  IconCode,
  IconUsers
} from '@/components/ui/icons'
import {
  Popover,
  PopoverContent,
  PopoverTrigger
} from '@/components/ui/popover'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import DateRangePicker from '@/components/date-range-picker'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { AnnualActivity } from './annual-activity'
import { DailyActivity } from './daily-activity'

const KEY_SELECT_ALL = 'all'
const DEFAULT_DATE_RANGE = '-14d'

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

export function Report() {
  const searchParams = useSearchParams()
  const [members] = useAllMembers()
  const isDemoMode = useIsDemoMode()
  const [dateRange, setDateRange] = useState<DateRange>({
    from: moment().add(parseInt(DEFAULT_DATE_RANGE, 10), 'day').toDate(),
    to: moment().toDate()
  })
  const [selectedMember, setSelectedMember] = useState(KEY_SELECT_ALL)
  const [selectedLanguage, setSelectedLanguage] = useState<Language[]>([])

  const sample = isDemoMode || searchParams.get('sample') === 'true'

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: moment(dateRange.from).startOf('day').utc().format(),
      end: moment(dateRange.to).endOf('day').utc().format(),
      users: selectedMember === KEY_SELECT_ALL ? undefined : [selectedMember]
    }
  })
  let dailyStats: DailyStatsQuery['dailyStats'] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: dateRange.from!,
      end: dateRange.to || dateRange.from!
    })
    dailyStats = daysBetweenRange.map(date => {
      const languages = [Language.Typescript, Language.Python, Language.Rust]
      const rng = seedrandom(
        moment(date).format('YYYY-MM-DD') + selectedMember + selectedLanguage
      )
      const selects = Math.ceil(rng() * 20)
      const completions = Math.ceil(selects / 0.35)
      return {
        start: moment(date).utc().format(),
        end: moment(date).add(1, 'day').utc().format(),
        completions,
        selects,
        views: completions,
        language: languages[selects % languages.length]
      }
    })
  } else {
    dailyStats = dailyStatsData?.dailyStats.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects,
      views: item.views,
      language: item.language
    }))
  }
  dailyStats = dailyStats?.filter(stats => {
    if (selectedLanguage.length === 0) return true
    return selectedLanguage.includes(stats.language)
  })

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    }
  })
  let yearlyStats: DailyStatsInPastYearQuery['dailyStatsInPastYear'] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: moment().toDate(),
      end: moment().subtract(365, 'days').toDate()
    })
    yearlyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(
        moment(date).format('YYYY-MM-DD') + selectedMember + selectedLanguage
      )
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 10)
      return {
        start: moment(date).format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        end: moment(date).add(1, 'day').format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        completions,
        selects,
        views: completions
      }
    })
  } else {
    yearlyStats = yearlyStatsData?.dailyStatsInPastYear.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects,
      views: item.views
    }))
  }

  return (
    <div className="w-[calc(100vw-2rem)] md:w-auto 2xl:mx-auto 2xl:max-w-5xl">
      <div className="mb-4 flex flex-col items-center justify-between gap-y-2 lg:flex-row lg:items-end lg:gap-y-0">
        <SubHeader className="mb-0">
          Statistics around Tabby IDE / Extensions
        </SubHeader>

        <LoadingWrapper
          loading={fetchingDailyState}
          fallback={<Skeleton className="h-6 w-32" />}
        >
          <Select
            defaultValue={KEY_SELECT_ALL}
            onValueChange={setSelectedMember}
          >
            <SelectTrigger className="h-auto w-auto border-none py-0 shadow-none">
              <div className="flex h-6 items-center">
                <IconUsers className="mr-[0.45rem]" />
                <div className="w-[190px] overflow-hidden text-ellipsis text-left">
                  <SelectValue />
                </div>
              </div>
            </SelectTrigger>
            <SelectContent align="end">
              <SelectGroup>
                <SelectItem value={KEY_SELECT_ALL}>All members</SelectItem>
                {members.map(member => (
                  <SelectItem value={member.id} key={member.id}>
                    {member.name || member.email}
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
          <h1 className="mb-2 text-center text-xl font-semibold md:text-start">
            Activity
          </h1>
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
          <div className="-mb-2 flex flex-col items-center justify-between gap-y-1 md:flex-row md:gap-y-0">
            <h1 className="text-xl font-semibold">Usage</h1>

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

          <StatsSummary dailyStats={dailyStats} />

          <DailyActivity dailyStats={dailyStats} dateRange={dateRange} />
        </div>
      </LoadingWrapper>
    </div>
  )
}
