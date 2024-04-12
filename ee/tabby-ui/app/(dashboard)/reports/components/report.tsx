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
import { toProgrammingLanguageDisplayName } from '@/lib/language-utils'
import { queryDailyStats, queryDailyStatsInPastYear } from '@/lib/tabby/query'
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import DatePickerWithRange from '@/components/ui/date-range-picker'
import {
  IconActivity,
  IconCheck,
  IconCode,
  IconUsers,
  IconChevronUpDown
} from '@/components/ui/icons'


import {
	Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command"
import {
	Popover,
	PopoverContent,
	PopoverTrigger,
} from "@/components/ui/popover"
import { Skeleton } from '@/components/ui/skeleton'
import LoadingWrapper from '@/components/loading-wrapper'
import { SubHeader } from '@/components/sub-header'

import { useAllMembers } from '../use-all-members'
import { AnnualActivity } from './annual-activity'
import { DailyActivity } from './daily-activity'

const INITIAL_DATE_RANGE = 14
const KEY_SELECT_ALL = 'all'

function StatsSummary({
  dailyStats
}: {
  dailyStats?: DailyStatsQuery['dailyStats']
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
            Total Completions
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

type OptionType<T> = {
	label: string;
	value: T;
}

function MultipSelectionContent<T> ({
  title,
  options,
  selected,
  onChange
}: {
  title: string;
  options: OptionType<T>[];
  selected: T[];
  onChange: React.Dispatch<React.SetStateAction<T[]>>;
}) {
  return (
    <Command>
      <CommandInput placeholder={title} />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>

        <CommandGroup>
          {options.map((option) => {
            const isSelected = selected.includes(option.value)
            return (
              <CommandItem
                key={option.value as string}
                onSelect={() => {
                  const newSelect = [...selected]
                  if (isSelected) {
                    const idx = newSelect.findIndex(item => item === option.value)
                    if (idx !== -1) newSelect.splice(idx, 1)
                  } else {
                    newSelect.push(option.value)
                  }
                  onChange(newSelect)
                }}
                className="!pointer-events-auto cursor-pointer !opacity-100"
              >
                <div
                  className={cn(
                    "mr-2 flex h-4 w-4 items-center justify-center rounded-sm border border-primary",
                    isSelected
                      ? "bg-primary text-primary-foreground"
                      : "opacity-50 [&_svg]:invisible"
                  )}
                >
                  <IconCheck className={cn("h-4 w-4")} />
                </div>
                <span>{option.label}</span>
              </CommandItem>
            )
          })}
        </CommandGroup>
        {selected.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup>
              <CommandItem
                onSelect={() => onChange([])}
                className="!pointer-events-auto cursor-pointer justify-center text-center !opacity-100"
              >
                Clear filters
              </CommandItem>
            </CommandGroup>
          </>
        )}
      </CommandList>
    </Command>
  )
}

export function Report() {
  const searchParams = useSearchParams()
  const sample = searchParams.get('sample') === 'true'
  const [members] = useAllMembers()
  const [dateRange, setDateRange] = useState<DateRange>({
    from: moment().subtract(INITIAL_DATE_RANGE, 'day').toDate(),
    to: moment().toDate()
  })
  const [selectedMember, setSelectedMember] = useState<string[]>([])
  const [selectedLanguage, setSelectedLanguage] = useState<Language[]>([]) 

  // Query stats of selected date range
  const [{ data: dailyStatsData, fetching: fetchingDailyState }] = useQuery({
    query: queryDailyStats,
    variables: {
      start: moment(dateRange.from).startOf('day').utc().format(),
      end: moment(dateRange.to).endOf('day').utc().format(),
      users: selectedMember.length === 0 ? undefined : selectedMember,
      languages: selectedLanguage.length === 0 ? undefined : selectedLanguage
    }
  })
  let dailyStats: DailyStatsQuery['dailyStats'] | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: dateRange.from!,
      end: dateRange.to || dateRange.from!
    })
    dailyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(
        moment(date).format('YYYY-MM-DD') + selectedMember + selectedLanguage
      )
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 10)
      return {
        start: moment(date).startOf('day').toDate(),
        end: moment(date).endOf('day').toDate(),
        completions,
        selects
      }
    })
  } else {
    dailyStats = dailyStatsData?.dailyStats.map(item => ({
      start: item.start,
      end: item.end,
      completions: item.completions,
      selects: item.selects
    }))
  }

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: queryDailyStatsInPastYear,
    variables: {
      // TODO: check if it is a bug in API, giving all members return nothing
      users: (selectedMember.length === 0 || selectedMember.length === members.length) ? undefined : selectedMember
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
        start: moment(date).startOf('day').toDate(),
        end: moment(date).endOf('day').toDate(),
        completions,
        selects
      }
    })
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
          <Popover>
            <PopoverTrigger asChild>
              <div className="flex h-6 items-center text-sm">
                <IconUsers className="mr-1" />
                <p className="mr-1">Members:</p>
                <div
                  className="block w-20 cursor-pointer rounded-sm font-normal"
                >
                  {selectedMember.length === 0 &&
                    <span>All</span>
                  }
                  {selectedMember.length === 1 &&
                    <p className="w-full overflow-hidden text-ellipsis">{members.find(m => m.id === selectedMember[0])?.email}</p>
                  }
                  {selectedMember.length > 1 &&
                    <span>{selectedMember.length} selected</span>
                  }
                </div>
                <IconChevronUpDown className="h-3 w-3" />
              </div>
            </PopoverTrigger>
            <PopoverContent className="w-[200px] p-0" align="end">
              <MultipSelectionContent
                title="Member"
                options={members.map(member => ({ label: member.email, value: member.id }))}
                selected={selectedMember}
                onChange={setSelectedMember} />
            </PopoverContent>
          </Popover>
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
            <Popover>
              <PopoverTrigger asChild>
                <div className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed">
                  <span className="mr-1.5 text-muted-foreground">
                    Language:
                  </span>
                  <div className="w-[80px]">
                    {selectedLanguage.length === 0 &&
                      <p className="w-full overflow-hidden text-ellipsis">All</p>
                    }
                    {selectedLanguage.length === 1 &&
                      <p className="w-full overflow-hidden text-ellipsis">
                        {toProgrammingLanguageDisplayName(selectedLanguage[0])}
                      </p>
                    }
                    {selectedLanguage.length > 1 &&
                      <span className="px-1">
                        {selectedLanguage.length} selected
                      </span>
                    }
                  </div>
                  <IconChevronUpDown className="h-3 w-3" />
                </div>
              </PopoverTrigger>
              <PopoverContent className="w-[200px] p-0" align="end">
                <MultipSelectionContent
                  title="Language"
                  options={Object.entries(Language).sort((_, b) => (b[1] === Language.Other ? -1 : 0)).map(([key, value]) => ({ label: key, value }))}
                  selected={selectedLanguage}
                  onChange={setSelectedLanguage} />
              </PopoverContent>
            </Popover>
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
