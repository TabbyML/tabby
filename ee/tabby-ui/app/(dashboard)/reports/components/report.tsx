'use client'

import { useState } from 'react'
import { useSearchParams } from 'next/navigation'
import { eachDayOfInterval } from 'date-fns'
import moment from 'moment'
import seedrandom from 'seedrandom'
import { useQuery } from 'urql'

import {
  ChatDailyStatsInPastYearQuery,
  DailyStatsInPastYearQuery
} from '@/lib/gql/generates/graphql'
import { useAllMembers } from '@/lib/hooks/use-all-members'
import { useIsDemoMode } from '@/lib/hooks/use-server-info'
import {
  chatDailyStatsInPastYearQuery,
  dailyStatsInPastYearQuery
} from '@/lib/tabby/query'
import { ArrayElementType } from '@/lib/types'
import { IconUsers } from '@/components/ui/icons'
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

import { AnnualActivity } from './annual-activity'
import { ChatDailyActivity } from './chat-daily-activity'
import { CompletionDailyActivity } from './completion-daily-activity'
import { KEY_SELECT_ALL } from './constants'

export function Report() {
  const searchParams = useSearchParams()
  const [members, fetchingMembers] = useAllMembers()
  const isDemoMode = useIsDemoMode()
  const [selectedMember, setSelectedMember] = useState(KEY_SELECT_ALL)
  const sample = isDemoMode || searchParams.get('sample') === 'true'

  // query chat yearly stats
  const [{ data: chatYearlyStatsData, fetching: fetchingChatYearlyStats }] =
    useQuery({
      query: chatDailyStatsInPastYearQuery,
      variables: {
        users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
      }
    })

  // Query yearly stats
  const [{ data: yearlyStatsData, fetching: fetchingYearlyStats }] = useQuery({
    query: dailyStatsInPastYearQuery,
    variables: {
      users: selectedMember === KEY_SELECT_ALL ? undefined : selectedMember
    }
  })
  let yearlyStats:
    | Array<
        | ArrayElementType<DailyStatsInPastYearQuery['dailyStatsInPastYear']>
        | ArrayElementType<
            ChatDailyStatsInPastYearQuery['chatDailyStatsInPastYear']
          >
      >
    | undefined
  if (sample) {
    const daysBetweenRange = eachDayOfInterval({
      start: moment().toDate(),
      end: moment().subtract(365, 'days').toDate()
    })
    yearlyStats = daysBetweenRange.map(date => {
      const rng = seedrandom(moment(date).format('YYYY-MM-DD') + selectedMember)
      const selects = Math.ceil(rng() * 20)
      const completions = selects + Math.floor(rng() * 10)
      return {
        __typename: 'CompletionStats',
        start: moment(date).format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        end: moment(date).add(1, 'day').format('YYYY-MM-DD[T]HH:mm:ss[Z]'),
        completions,
        selects,
        views: completions
      }
    })
  } else {
    yearlyStats = [
      ...(yearlyStatsData?.dailyStatsInPastYear || []),
      ...(chatYearlyStatsData?.chatDailyStatsInPastYear || [])
    ]
  }

  return (
    <div className="w-[calc(100vw-2rem)] md:w-auto 2xl:mx-auto 2xl:max-w-5xl">
      <div className="mb-4 flex flex-col items-center justify-between gap-y-2 lg:flex-row lg:items-end lg:gap-y-0">
        <SubHeader className="mb-0">
          Statistics around Tabby IDE / Extensions
        </SubHeader>

        <LoadingWrapper
          loading={fetchingMembers}
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

      {/* Yearly */}
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

      {/* completions */}
      <CompletionDailyActivity
        sample={sample}
        selectedMember={selectedMember}
      />

      {/* chats */}
      <ChatDailyActivity sample={sample} selectedMember={selectedMember} />
    </div>
  )
}
